//! Debug dump of real attention queries and keys, for the latent sparse
//! attention study (board P20) and its example `lsa_probe`.
//!
//! Off unless `ALLPACA_LSA_DUMP` names a directory. With it set, every
//! full-attention layer of the device forward appends, per layer `L`:
//!
//! - `L<L>.k.f32`: one row of `n_kv_heads * head_dim` floats per cached key
//!   (normed and rotated, as the attention reads it);
//! - `L<L>.v.f32`: the values, only with `ALLPACA_LSA_DUMP_V=1`;
//! - `L<L>.q.f32`: one row of `n_heads * head_dim` floats per captured query;
//! - `L<L>.qpos.u32`: the position of each captured query.
//!
//! A decode step captures its query. A prefill forward captures the rows
//! whose position plus one is a multiple of `ALLPACA_LSA_DUMP_STRIDE`
//! (default 1024). A forward that starts at or before a key already written
//! (a new request, a rewind) starts the layer's files over, so the directory
//! holds the last request.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use allpaca_types::{AllpacaError, FatalError};

/// Key rows fetched from the device per copy.
const KEY_ROWS_PER_COPY: usize = 4096;

pub(crate) struct LsaDump {
    dir: PathBuf,
    stride: usize,
    values: bool,
    /// Key rows written per layer.
    written: Mutex<HashMap<usize, usize>>,
}

fn io_err(what: &str, e: std::io::Error) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(format!("lsa dump: {what}: {e}")))
}

/// The dump, when `ALLPACA_LSA_DUMP` is set.
pub(crate) fn active() -> Option<&'static LsaDump> {
    static DUMP: OnceLock<Option<LsaDump>> = OnceLock::new();
    DUMP.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("ALLPACA_LSA_DUMP")?);
        let stride = std::env::var("ALLPACA_LSA_DUMP_STRIDE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|s| *s > 0)
            .unwrap_or(1024);
        let values = std::env::var("ALLPACA_LSA_DUMP_V").is_ok_and(|v| v == "1");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!("lsa dump: cannot create {}: {e}", dir.display());
            return None;
        }
        tracing::warn!(
            "lsa dump: writing attention queries and keys to {} (prefill stride {stride}, values {values})",
            dir.display()
        );
        Some(LsaDump {
            dir,
            stride,
            values,
            written: Mutex::new(HashMap::new()),
        })
    })
    .as_ref()
}

fn as_bytes(floats: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any bit pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(floats.as_ptr().cast(), floats.len() * 4) }
}

impl LsaDump {
    fn open(&self, layer: usize, what: &str, fresh: bool) -> Result<File, AllpacaError> {
        let path = self.dir.join(format!("L{layer}.{what}"));
        let mut options = OpenOptions::new();
        options.create(true);
        if fresh {
            options.write(true).truncate(true);
        } else {
            options.append(true);
        }
        options.open(&path).map_err(|e| io_err(&path.display().to_string(), e))
    }

    /// Record one attention call of `layer`: `n_rows` queries whose first
    /// is at position `start_pos`, over `n_kv` cached keys.
    ///
    /// `fetch_q(row, out)` reads one query row, `fetch_k(first, out)` and
    /// `fetch_v(first, out)` read whole key and value rows from `first` on.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture(
        &self,
        layer: usize,
        start_pos: usize,
        n_rows: usize,
        n_kv: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        fetch_q: &dyn Fn(usize, &mut [f32]) -> Result<(), AllpacaError>,
        fetch_k: &dyn Fn(usize, &mut [f32]) -> Result<(), AllpacaError>,
        fetch_v: &dyn Fn(usize, &mut [f32]) -> Result<(), AllpacaError>,
    ) -> Result<(), AllpacaError> {
        let kv_dim = n_kv_heads * head_dim;
        let mut written = self.written.lock().unwrap_or_else(|e| e.into_inner());
        let have = written.get(&layer).copied();
        let fresh = have.is_none_or(|w| start_pos < w);
        let mut done = if fresh { 0 } else { have.unwrap_or(0) };
        if fresh {
            let meta = format!(
                "{{\"n_heads\":{n_heads},\"n_kv_heads\":{n_kv_heads},\"head_dim\":{head_dim},\"stride\":{},\"values\":{}}}\n",
                self.stride, self.values
            );
            std::fs::write(self.dir.join("meta.json"), meta).map_err(|e| io_err("meta.json", e))?;
            for what in ["q.f32", "qpos.u32"] {
                self.open(layer, what, true)?;
            }
        }
        let mut k_file = self.open(layer, "k.f32", fresh)?;
        let mut v_file = if self.values { Some(self.open(layer, "v.f32", fresh)?) } else { None };
        let mut rows = vec![0.0f32; KEY_ROWS_PER_COPY.min(n_kv.saturating_sub(done)).max(1) * kv_dim];
        while done < n_kv {
            let n = KEY_ROWS_PER_COPY.min(n_kv - done);
            let rows = &mut rows[..n * kv_dim];
            fetch_k(done, rows)?;
            k_file.write_all(as_bytes(rows)).map_err(|e| io_err("keys", e))?;
            if let Some(v_file) = v_file.as_mut() {
                fetch_v(done, rows)?;
                v_file.write_all(as_bytes(rows)).map_err(|e| io_err("values", e))?;
            }
            done += n;
        }
        written.insert(layer, done);

        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut q_file = None;
        for row in 0..n_rows {
            let pos = start_pos + row;
            if n_rows > 1 && (pos + 1) % self.stride != 0 {
                continue;
            }
            fetch_q(row, &mut q)?;
            let (q_file, pos_file) = match q_file.as_mut() {
                Some(files) => files,
                None => q_file.insert((self.open(layer, "q.f32", false)?, self.open(layer, "qpos.u32", false)?)),
            };
            q_file.write_all(as_bytes(&q)).map_err(|e| io_err("queries", e))?;
            pos_file.write_all(&(pos as u32).to_le_bytes()).map_err(|e| io_err("positions", e))?;
        }
        Ok(())
    }
}
