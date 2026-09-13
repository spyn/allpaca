use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::response::sse::{Event, Sse};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

use crate::api::AppState;

const DEFAULT_CAP: usize = 2048;

#[derive(Debug, Clone, Serialize)]
pub struct LogLine {
    pub seq: u64,
    pub ts_ms: u64,
    pub level: String,
    pub target: String,
    pub message: String,
}

struct Inner {
    seq: u64,
    lines: VecDeque<LogLine>,
    cap: usize,
}

#[derive(Clone)]
pub struct LogRing {
    inner: Arc<Mutex<Inner>>,
}

impl LogRing {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                seq: 0,
                lines: VecDeque::with_capacity(cap.min(DEFAULT_CAP).max(1)),
                cap: cap.max(1),
            })),
        }
    }

    pub fn push(&self, level: &str, target: &str, message: String) {
        let Ok(mut g) = self.inner.lock() else {
            return;
        };
        g.seq += 1;
        let line = LogLine {
            seq: g.seq,
            ts_ms: now_ms(),
            level: level.to_string(),
            target: target.to_string(),
            message,
        };
        if g.lines.len() >= g.cap {
            g.lines.pop_front();
        }
        g.lines.push_back(line);
    }

    pub fn snapshot(&self) -> Vec<LogLine> {
        self.inner
            .lock()
            .map(|g| g.lines.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn after(&self, seq: u64) -> Vec<LogLine> {
        self.inner
            .lock()
            .map(|g| g.lines.iter().filter(|l| l.seq > seq).cloned().collect())
            .unwrap_or_default()
    }

    pub fn layer(&self) -> LogRingLayer {
        LogRingLayer {
            ring: self.clone(),
        }
    }
}

#[derive(Clone)]
pub struct LogRingLayer {
    ring: LogRing,
}

impl<S> Layer<S> for LogRingLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        if visitor.message.is_empty() {
            return;
        }
        self.ring.push(
            meta.level().as_str(),
            meta.target(),
            visitor.message,
        );
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
            if self.message.starts_with('"') && self.message.ends_with('"') && self.message.len() >= 2
            {
                self.message = self.message[1..self.message.len() - 1].to_string();
            }
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Deserialize)]
pub struct LogsQuery {
    pub after: Option<u64>,
}

#[derive(Serialize)]
pub struct LogsBody {
    pub lines: Vec<LogLine>,
}

pub async fn logs_snapshot(
    State(state): State<AppState>,
    Query(q): Query<LogsQuery>,
) -> Json<LogsBody> {
    let lines = match q.after {
        Some(seq) => state.logs.after(seq),
        None => state.logs.snapshot(),
    };
    Json(LogsBody { lines })
}

pub async fn logs_stream(
    State(state): State<AppState>,
    Query(q): Query<LogsQuery>,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let mut after = q.after.unwrap_or(0);
    Sse::new(async_stream::stream! {
        loop {
            let lines = state.logs.after(after);
            for line in lines {
                after = after.max(line.seq);
                if let Ok(data) = serde_json::to_string(&line) {
                    yield Ok(Event::default().data(data));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        }
    })
}
