//! MoE pin-prefer keep-set TUI (`allpaca-server --tune`).

use std::io::{self, stdout};
use std::path::PathBuf;
use std::time::Duration;

use allpaca_core::{
    available_groups, available_profiles, cache_bytes_for, discover_models, estimate_budget,
    expand_groups, keep_experts_from_ids, keep_ids_for_entry, load_catalog_beside, probe_checkpoint,
    DeviceInventory, DiscoveredModel, ExpertId, KeepsetCatalog, MoeProbe, ResourceSnapshot,
    TuneBudget, TuneGroup, TuneProfile, CTX_STEPS,
};
use allpaca_types::{Config, TuneEntry};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;
use ratatui::Terminal;

use crate::host::HostSampler;

enum Screen {
    Picker,
    Tune,
}

struct App {
    screen: Screen,
    models: Vec<DiscoveredModel>,
    picker_idx: usize,
    probe: Option<MoeProbe>,
    catalog: Option<KeepsetCatalog>,
    groups: Vec<TuneGroup>,
    profiles: Vec<TuneProfile>,
    selected_groups: Vec<String>,
    group_idx: usize,
    profile_idx: usize,
    ctx_idx: usize,
    status: String,
    config: Config,
    config_path: PathBuf,
    resources: ResourceSnapshot,
}

pub fn run(
    model: Option<PathBuf>,
    config: Config,
    config_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = config_path.unwrap_or_else(|| PathBuf::from("allpaca.toml"));
    let resources = resource_snapshot(&config);
    let models = discover_models(&config);
    let mut app = App {
        screen: Screen::Picker,
        models,
        picker_idx: 0,
        probe: None,
        catalog: None,
        groups: Vec::new(),
        profiles: Vec::new(),
        selected_groups: Vec::new(),
        group_idx: 0,
        profile_idx: 0,
        ctx_idx: 0,
        status: String::new(),
        config,
        config_path,
        resources,
    };
    if let Some(path) = model {
        let spec = path.to_string_lossy().into_owned();
        app.open_model(&spec, &path)?;
        app.screen = Screen::Tune;
    } else if let Some(path) = app.config.model.path.clone() {
        let spec = path.to_string_lossy().into_owned();
        if path.exists() {
            app.open_model(&spec, &path)?;
            app.screen = Screen::Tune;
        }
    }
    if matches!(app.screen, Screen::Picker) && app.models.is_empty() {
        return Err("no models found; pass --model or put GGUF files in ./models".into());
    }

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = app.event_loop(&mut terminal);
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    result
}

fn resource_snapshot(config: &Config) -> ResourceSnapshot {
    let host = HostSampler::new().snapshot();
    let inv = DeviceInventory::discover(config);
    let gpu = inv.devices.iter().find(|d| d.is_gpu());
    ResourceSnapshot {
        ram_total_bytes: host.ram_total_bytes,
        ram_used_bytes: host.ram_used_bytes,
        gpu_name: gpu
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "cpu".into()),
        gpu_memory_bytes: gpu.map(|d| d.memory_bytes).unwrap_or(0),
    }
}

impl App {
    fn open_model(&mut self, spec: &str, path: &std::path::Path) -> Result<(), String> {
        let probe = probe_checkpoint(spec, path)?;
        self.catalog = load_catalog_beside(path);
        self.groups = available_groups(&probe, self.catalog.as_ref());
        self.profiles = available_profiles(&probe, self.catalog.as_ref());
        if let Some(saved) = self.config.tune_for_spec(spec).cloned() {
            self.selected_groups = saved.groups.clone();
            if saved.ctx_tokens > 0 {
                if let Some(i) = CTX_STEPS.iter().position(|&c| c == saved.ctx_tokens) {
                    self.ctx_idx = i;
                }
            }
            if let Some(i) = self.profiles.iter().position(|p| p.id == saved.profile) {
                self.profile_idx = i;
            }
        }
        self.group_idx = 0;
        self.probe = Some(probe);
        self.status.clear();
        Ok(())
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            self.resources = resource_snapshot(&self.config);
            terminal.draw(|f| self.draw(f))?;
            if !event::poll(Duration::from_millis(250))? {
                continue;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match self.screen {
                Screen::Picker => {
                    if self.handle_picker(key.code)? {
                        break;
                    }
                }
                Screen::Tune => {
                    if self.handle_tune(key.code)? {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_picker(&mut self, code: KeyCode) -> Result<bool, Box<dyn std::error::Error>> {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.models.is_empty() {
                    self.picker_idx = (self.picker_idx + 1) % self.models.len();
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if !self.models.is_empty() {
                    self.picker_idx = self
                        .picker_idx
                        .checked_sub(1)
                        .unwrap_or(self.models.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(m) = self.models.get(self.picker_idx).cloned() {
                    match self.open_model(&m.spec, &m.path) {
                        Ok(()) => self.screen = Screen::Tune,
                        Err(e) => self.status = e,
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_tune(&mut self, code: KeyCode) -> Result<bool, Box<dyn std::error::Error>> {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.models.is_empty() {
                    return Ok(true);
                }
                self.screen = Screen::Picker;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.groups.is_empty() {
                    self.group_idx = (self.group_idx + 1) % self.groups.len();
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if !self.groups.is_empty() {
                    self.group_idx = self
                        .group_idx
                        .checked_sub(1)
                        .unwrap_or(self.groups.len() - 1);
                }
            }
            KeyCode::Left => {
                self.ctx_idx = self.ctx_idx.saturating_sub(1);
            }
            KeyCode::Right => {
                self.ctx_idx = (self.ctx_idx + 1).min(CTX_STEPS.len() - 1);
            }
            KeyCode::Tab => {
                if !self.profiles.is_empty() {
                    self.profile_idx = (self.profile_idx + 1) % self.profiles.len();
                    self.apply_profile();
                }
            }
            KeyCode::Enter => self.apply_profile(),
            KeyCode::Char(' ') => self.toggle_group(),
            KeyCode::Char('s') | KeyCode::Char('S') => self.save()?,
            _ => {}
        }
        Ok(false)
    }

    fn apply_profile(&mut self) {
        if let Some(p) = self.profiles.get(self.profile_idx).cloned() {
            self.selected_groups = p.groups.clone();
            if let Some(i) = CTX_STEPS.iter().position(|&c| c == p.ctx_tokens) {
                self.ctx_idx = i;
            }
            self.status = format!("applied profile {}", p.id);
        }
    }

    fn toggle_group(&mut self) {
        let Some(g) = self.groups.get(self.group_idx) else {
            return;
        };
        let id = g.id.clone();
        if let Some(i) = self.selected_groups.iter().position(|x| x == &id) {
            self.selected_groups.remove(i);
        } else {
            self.selected_groups.push(id);
        }
    }

    fn pin_ids(&self) -> Vec<ExpertId> {
        let Some(probe) = &self.probe else {
            return Vec::new();
        };
        let catalog = self.catalog.as_ref();
        let entry = TuneEntry {
            groups: self.selected_groups.clone(),
            keep_experts: Vec::new(),
            ..TuneEntry::default()
        };
        let ids = keep_ids_for_entry(&entry, &probe.shape, catalog);
        if ids.is_empty() {
            expand_groups(&self.selected_groups, &probe.shape)
        } else {
            ids
        }
    }

    fn cache_bytes(&self) -> usize {
        let Some(probe) = &self.probe else {
            return 0;
        };
        let frac = self
            .profiles
            .get(self.profile_idx)
            .map(|p| p.cache_fraction)
            .unwrap_or(0.4);
        cache_bytes_for(probe, &self.pin_ids(), frac, &self.resources)
    }

    fn budget(&self) -> Option<TuneBudget> {
        let probe = self.probe.as_ref()?;
        Some(estimate_budget(
            probe,
            &self.pin_ids(),
            CTX_STEPS[self.ctx_idx],
            self.cache_bytes() as u64,
            &self.resources,
        ))
    }

    fn save(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(probe) = self.probe.as_ref() else {
            self.status = "no model selected".into();
            return Ok(());
        };
        if !probe.shape.is_moe() {
            self.status = "dense model: no routed experts to pin".into();
            return Ok(());
        }
        let ids = self.pin_ids();
        let profile = self
            .profiles
            .get(self.profile_idx)
            .map(|p| p.id.clone())
            .unwrap_or_else(|| "custom".into());
        let entry = TuneEntry {
            profile,
            groups: self.selected_groups.clone(),
            ctx_tokens: CTX_STEPS[self.ctx_idx],
            cache_bytes: self.cache_bytes(),
            keep_experts: keep_experts_from_ids(&ids, probe.shape.n_layers),
        };
        Config::write_tune_file(
            &self.config_path,
            &probe.spec,
            &entry,
            Some(&probe.path),
        )?;
        self.config.upsert_tune(&probe.spec, entry);
        self.config.model.path = Some(probe.path.clone());
        self.status = format!("saved {}", self.config_path.display());
        Ok(())
    }

    fn draw(&self, f: &mut Frame<'_>) {
        match self.screen {
            Screen::Picker => self.draw_picker(f),
            Screen::Tune => self.draw_tune(f),
        }
    }

    fn draw_picker(&self, f: &mut Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1), Constraint::Length(2)])
            .split(f.area());
        f.render_widget(header_block(&self.resources), chunks[0]);
        let items: Vec<ListItem> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let mark = if i == self.picker_idx { "▸ " } else { "  " };
                ListItem::new(format!(
                    "{mark}{}  {}",
                    m.label,
                    format_bytes(m.bytes)
                ))
            })
            .collect();
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Select a model "),
        );
        f.render_widget(list, chunks[1]);
        let help = if self.status.is_empty() {
            "↑↓ choose · enter open · q quit".to_string()
        } else {
            self.status.clone()
        };
        f.render_widget(Paragraph::new(help), chunks[2]);
    }

    fn draw_tune(&self, f: &mut Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(8),
                Constraint::Length(3),
            ])
            .split(f.area());
        f.render_widget(header_block(&self.resources), chunks[0]);
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(chunks[1]);
        self.draw_groups(f, body[0]);
        self.draw_budget(f, body[1]);
        let help = if self.status.is_empty() {
            "↑↓ group · space toggle · tab/enter profile · ←→ context · s save · q back".into()
        } else {
            self.status.clone()
        };
        f.render_widget(Paragraph::new(help), chunks[2]);
    }

    fn draw_groups(&self, f: &mut Frame<'_>, area: Rect) {
        let title = self
            .probe
            .as_ref()
            .map(|p| {
                if p.shape.is_moe() {
                    format!(
                        " {} · {} experts × {} layers ",
                        p.name, p.shape.n_experts, p.shape.n_layers
                    )
                } else {
                    format!(" {} · dense (no routed experts) ", p.name)
                }
            })
            .unwrap_or_else(|| " groups ".into());
        let items: Vec<ListItem> = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let on = self.selected_groups.iter().any(|s| s == &g.id);
                let cursor = if i == self.group_idx { "▸" } else { " " };
                let mark = if on { "●" } else { "○" };
                ListItem::new(format!(
                    "{cursor} {mark} {:<18} {:>6} experts",
                    g.title, g.expert_count
                ))
            })
            .collect();
        let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title));
        f.render_widget(list, area);
    }

    fn draw_budget(&self, f: &mut Frame<'_>, area: Rect) {
        let ctx = CTX_STEPS[self.ctx_idx];
        let profile = self
            .profiles
            .get(self.profile_idx)
            .map(|p| p.title.as_str())
            .unwrap_or("—");
        let mut lines = vec![
            Line::from(format!("profile  {profile}")),
            Line::from(format!("context  {ctx} tokens")),
        ];
        if let Some(b) = self.budget() {
            let total = b.total_experts.max(1);
            let frac = b.pin_count as f64 / total as f64;
            lines.push(Line::from(format!(
                "experts  {} / {}  ({:.0}%)",
                b.pin_count,
                b.total_experts,
                frac * 100.0
            )));
            lines.push(Line::from(format!(
                "pin      {}",
                format_bytes(b.expert_pin_bytes)
            )));
            lines.push(Line::from(format!("KV       {}", format_bytes(b.kv_bytes))));
            lines.push(Line::from(format!(
                "dense    {}",
                format_bytes(b.dense_bytes)
            )));
            lines.push(Line::from(format!(
                "resident {}",
                format_bytes(b.resident_bytes)
            )));
            let verdict = if b.fits {
                Span::styled("FITS", Style::default().add_modifier(Modifier::BOLD))
            } else {
                Span::styled("OVER", Style::default().add_modifier(Modifier::BOLD))
            };
            let free = if b.free_after >= 0 {
                format!("+{}", format_bytes(b.free_after as u64))
            } else {
                format!("-{}", format_bytes((-b.free_after) as u64))
            };
            lines.push(Line::from(vec![
                Span::raw(format!("room     {free}  ")),
                verdict,
            ]));
            lines.push(Line::from(""));
            lines.push(Line::from(
                "Bars are residency (how much of the selection fits), not routing coverage.",
            ));
        }
        let p = Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(" Budget "));
        f.render_widget(p, area);
    }
}

fn header_block(res: &ResourceSnapshot) -> Paragraph<'static> {
    let ram_free = res.ram_free_bytes();
    let gpu = if res.gpu_memory_bytes > 0 {
        format!(
            "{} · {}",
            res.gpu_name,
            format_bytes(res.gpu_memory_bytes)
        )
    } else {
        "no GPU".into()
    };
    Paragraph::new(format!(
        "allpaca tune  ·  RAM {} free / {}  ·  {gpu}",
        format_bytes(ram_free),
        format_bytes(res.ram_total_bytes)
    ))
    .block(Block::default().borders(Borders::ALL))
}

fn format_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let x = n as f64;
    if x >= GIB {
        format!("{:.1} GiB", x / GIB)
    } else if x >= MIB {
        format!("{:.1} MiB", x / MIB)
    } else if x >= KIB {
        format!("{:.1} KiB", x / KIB)
    } else {
        format!("{n} B")
    }
}
