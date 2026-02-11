//! TUI (Terminal User Interface) Module
//!
//! Provides a real-time monitoring dashboard for Worker processes.

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Worker display information
#[derive(Clone, Debug)]
pub struct WorkerDisplayInfo {
    pub worker_id: String,
    pub device_id: u32,
    pub status: String,
    pub restart_count: u32,
    pub uptime: Duration,
    pub memory_used: Option<u64>,
    pub memory_total: Option<u64>,
}

/// TUI Application State
pub struct TuiApp {
    pub workers: Vec<WorkerDisplayInfo>,
    pub log_lines: Vec<String>,
    pub should_quit: bool,
}

impl Default for TuiApp {
    fn default() -> Self {
        Self {
            workers: vec![],
            log_lines: vec![],
            should_quit: false,
        }
    }
}

impl TuiApp {
    /// Create a new TUI app
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a worker to display
    pub fn add_worker(&mut self, info: WorkerDisplayInfo) {
        self.workers.push(info);
    }

    /// Update a worker's status
    pub fn update_worker(&mut self, worker_id: &str, status: String) {
        if let Some(worker) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) {
            worker.status = status;
        }
    }

    /// Add a log line
    pub fn add_log_line(&mut self, line: String) {
        self.log_lines.push(line);
        // Keep only last 10 log lines
        if self.log_lines.len() > 10 {
            self.log_lines.remove(0);
        }
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> (usize, usize, usize) {
        let total = self.workers.len();
        let running = self
            .workers
            .iter()
            .filter(|w| w.status == "Running")
            .count();
        let failed = self
            .workers
            .iter()
            .filter(|w| w.status.contains("Failed"))
            .count();
        (total, running, failed)
    }
}

/// Terminal UI handler
pub struct Tui {
    app: Arc<Mutex<TuiApp>>,
}

impl Tui {
    /// Create a new TUI
    pub fn new() -> Self {
        Self {
            app: Arc::new(Mutex::new(TuiApp::new())),
        }
    }

    /// Get shared app state
    pub fn app(&self) -> Arc<Mutex<TuiApp>> {
        Arc::clone(&self.app)
    }

    /// Run the TUI
    pub fn run(&self) -> io::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Run the main loop
        let result = self.run_app(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    /// Main event loop
    fn run_app<B: Backend>(&self, terminal: &mut Terminal<B>) -> io::Result<()> {
        let tick_rate = Duration::from_millis(200);
        let mut last_tick = Instant::now();

        loop {
            // Render
            let app = self.app.lock().unwrap();
            terminal.draw(|f| {
                draw_ui(f, &app);
            })?;

            // Check for events
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            self.app.lock().unwrap().should_quit = true;
                            return Ok(());
                        }
                        KeyCode::Char('r') => {
                            // Reload/refresh
                        }
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }

            // Check if should quit
            if self.app.lock().unwrap().should_quit {
                return Ok(());
            }
        }
    }
}

/// Draw the UI
fn draw_ui(f: &mut ratatui::Frame, app: &TuiApp) {
    let size = f.size();

    // Create main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(3),  // Header
                Constraint::Min(10),    // Workers table
                Constraint::Length(8),  // Logs
                Constraint::Length(2),  // Footer
            ]
            .as_ref(),
        )
        .split(size);

    // Draw header
    draw_header(f, chunks[0], app);

    // Draw workers table
    draw_workers_table(f, chunks[1], app);

    // Draw recent logs
    draw_logs(f, chunks[2], app);

    // Draw footer
    draw_footer(f, chunks[3]);
}

/// Draw the header with summary statistics
fn draw_header(f: &mut ratatui::Frame, area: Rect, app: &TuiApp) {
    let (total, running, failed) = app.get_summary();

    let header_text = vec![
        Line::from(vec![
            Span::styled(
                "╔ RustInfer Worker Spawner Monitor ",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw("╗"),
        ]),
        Line::from(vec![
            Span::raw(format!("  Workers: "),),
            Span::styled(
                format!("{}", total),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" Total  │  ",)),
            Span::styled(
                format!("{}", running),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" Running  │  ",)),
            Span::styled(
                format!("{}", failed),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" Failed")),
        ]),
    ];

    let header = Paragraph::new(header_text).alignment(Alignment::Left);
    f.render_widget(header, area);
}

/// Draw the workers table
fn draw_workers_table(f: &mut ratatui::Frame, area: Rect, app: &TuiApp) {
    let header = "│ ID        │ GPU │ Status  │ Restarts │ Uptime   │";
    let separator = "├───────────────────────────────────────────────────────────┤";

    let mut lines = vec![
        Line::from(Span::styled(
            header,
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::raw(separator)),
    ];

    // Add worker rows
    for worker in &app.workers {
        let status_color = match worker.status.as_str() {
            "Running" => Color::Green,
            "Waiting" => Color::Cyan,
            "Stopped" => Color::Yellow,
            "Failed" => Color::Red,
            _ => Color::Gray,
        };

        let uptime_str = format_duration(worker.uptime);
        let row_text = format!(
            "│ {:<9} │ {:>3} │ {:<7} │ {:<8} │ {:<8} │",
            worker.worker_id, worker.device_id, worker.status, worker.restart_count, uptime_str
        );

        lines.push(Line::from(Span::styled(
            row_text,
            Style::default().fg(status_color),
        )));
    }

    let table = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Workers"))
        .alignment(Alignment::Left);

    f.render_widget(table, area);
}

/// Draw recent logs
fn draw_logs(f: &mut ratatui::Frame, area: Rect, app: &TuiApp) {
    let log_lines: Vec<Line> = app
        .log_lines
        .iter()
        .rev()
        .take(6)
        .map(|line| Line::from(Span::raw(line.clone())))
        .collect();

    let logs = Paragraph::new(log_lines)
        .block(Block::default().borders(Borders::ALL).title("Recent Logs"))
        .alignment(Alignment::Left);

    f.render_widget(logs, area);
}

/// Draw the footer with instructions
fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    let footer_text = vec![Line::from(vec![
        Span::styled("  q", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" quit  │  "),
        Span::styled("r", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" reload  │  "),
        Span::styled("k", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" kill worker  │  "),
        Span::styled("↑↓", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" navigate"),
    ])];

    let footer = Paragraph::new(footer_text)
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(footer, area);
}

/// Format duration as human-readable string
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let mins = secs / 60;
    let hours = mins / 60;
    let days = hours / 24;

    if days > 0 {
        format!("{}d {:02}h", days, hours % 24)
    } else if hours > 0 {
        format!("{}h {:02}m", hours, mins % 60)
    } else if mins > 0 {
        format!("{}m {:02}s", mins, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_app_creation() {
        let app = TuiApp::new();
        assert_eq!(app.workers.len(), 0);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_add_worker() {
        let mut app = TuiApp::new();
        let worker = WorkerDisplayInfo {
            worker_id: "worker-0".to_string(),
            device_id: 0,
            status: "Running".to_string(),
            restart_count: 0,
            uptime: Duration::from_secs(100),
            memory_used: Some(2_000_000_000),
            memory_total: Some(8_000_000_000),
        };
        app.add_worker(worker);
        assert_eq!(app.workers.len(), 1);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(120)), "2m 00s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 01m");
    }
}
