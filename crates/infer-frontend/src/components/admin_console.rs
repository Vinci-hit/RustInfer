use dioxus::prelude::*;
use crate::api::client::ApiClient;
use crate::state::metrics::SystemMetrics;

#[component]
pub fn AdminConsole() -> Element {
    let mut metrics = use_signal(|| None::<SystemMetrics>);
    let mut metrics_history = use_signal(|| Vec::<SystemMetrics>::new());
    let mut auto_refresh = use_signal(|| true);
    let api_client = use_signal(|| ApiClient::new());

    // Poll every 1 second for admin console
    use_future(move || async move {
        loop {
            if auto_refresh.read().to_owned() {
                if let Ok(sys_metrics) = api_client.read().get_system_metrics().await {
                    metrics.set(Some(sys_metrics.clone()));

                    // Keep last 60 data points (1 minute of history)
                    metrics_history.write().push(sys_metrics);
                    if metrics_history.read().len() > 60 {
                        metrics_history.write().remove(0);
                    }
                }
            }
            gloo_timers::future::TimeoutFuture::new(1000).await;
        }
    });

    rsx! {
        div {
            class: "min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6",

            // Header with glassmorphism
            div {
                class: "sticky top-0 z-10 backdrop-blur-md bg-slate-900/70 border-b border-slate-800/50 mb-8 py-4 -mx-6 px-6",
                div {
                    class: "max-w-7xl mx-auto flex items-center justify-between",
                    div {
                        h1 {
                            class: "text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent",
                            "RustInfer Admin Console"
                        }
                        p { class: "text-slate-400 mt-1", "Real-time inference performance and cache metrics" }
                    }
                    div {
                        class: "flex items-center gap-4",
                        button {
                            class: "px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all",
                            onclick: move |_| {
                                auto_refresh.toggle();
                            },
                            {
                                if *auto_refresh.read() { "⏸ Pause" } else { "▶ Resume" }
                            }
                        }
                        div {
                            class: "flex items-center gap-2 text-sm",
                            span {
                                class: {
                                    if *auto_refresh.read() { "w-2 h-2 rounded-full bg-green-500 animate-pulse" } else { "w-2 h-2 rounded-full bg-slate-600" }
                                }
                            }
                            span { class: "text-slate-400", "Live" }
                        }
                    }
                }
            }

            if let Some(m) = metrics.read().as_ref() {
                div { class: "space-y-6 pb-8",

                    // Key Metrics Row - Large cards with gradients
                    div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4",
                        GradientCard {
                            title: "Total Requests",
                            value: format!("{}", m.engine.as_ref().map(|e| e.total_requests).unwrap_or(0)),
                            subtitle: format!("{} completed", m.engine.as_ref().map(|e| e.completed_requests).unwrap_or(0)),
                            gradient: "from-blue-500/20 to-cyan-500/20",
                            border_color: "border-blue-500/30",
                            text_color: "text-blue-400",
                            icon: "📊"
                        }

                        GradientCard {
                            title: "Cache Hit Rate",
                            value: format!("{:.1}%", m.cache.as_ref().map(|c| c.hit_rate * 100.0).unwrap_or(0.0)),
                            subtitle: format!("{} hits",
                                m.cache.as_ref().map(|c| c.hits).unwrap_or(0)
                            ),
                            gradient: "from-emerald-500/20 to-green-500/20",
                            border_color: "border-emerald-500/30",
                            text_color: "text-emerald-400",
                            icon: "⚡"
                        }

                        GradientCard {
                            title: "Tokens Generated",
                            value: format_large_number(m.engine.as_ref().map(|e| e.total_tokens_generated).unwrap_or(0)),
                            subtitle: "total tokens",
                            gradient: "from-purple-500/20 to-pink-500/20",
                            border_color: "border-purple-500/30",
                            text_color: "text-purple-400",
                            icon: "🔢"
                        }

                        GradientCard {
                            title: "Queue Status",
                            value: format!("{}", m.engine.as_ref().map(|e| e.queue_size).unwrap_or(0)),
                            subtitle: format!("of {} capacity", m.engine.as_ref().map(|e| e.queue_capacity).unwrap_or(0)),
                            gradient: "from-amber-500/20 to-orange-500/20",
                            border_color: "border-amber-500/30",
                            text_color: "text-amber-400",
                            icon: "📋"
                        }
                    }

                    // Detailed Metrics Section
                    div { class: "grid grid-cols-1 lg:grid-cols-2 gap-6",

                        // RadixCache Metrics Panel
                        MetricPanel {
                            title: "RadixCache Metrics",
                            icon: "🔄",
                            color: "emerald",

                            if let Some(cache) = &m.cache {
                                div { class: "space-y-6",
                                    // Hit Rate Gauge with improved design
                                    div {
                                        class: "flex flex-col items-center",
                                        div { class: "relative w-40 h-40",
                                            svg {
                                                class: "transform -rotate-90",
                                                view_box: "0 0 100 100",

                                                // Background circle with gradient
                                                defs {
                                                    linearGradient {
                                                        id: "emerald-gradient",
                                                        x1: "0%",
                                                        y1: "0%",
                                                        x2: "100%",
                                                        y2: "100%",
                                                        stop {
                                                            offset: "0%",
                                                            "stop-color": "#10b981",
                                                            "stop-opacity": "1"
                                                        }
                                                        stop {
                                                            offset: "100%",
                                                            "stop-color": "#059669",
                                                            "stop-opacity": "1"
                                                        }
                                                    }
                                                }
                                                circle {
                                                    cx: "50",
                                                    cy: "50",
                                                    r: "45",
                                                    fill: "none",
                                                    stroke: "#1e293b",
                                                    stroke_width: "10"
                                                }

                                                // Progress circle
                                                circle {
                                                    cx: "50",
                                                    cy: "50",
                                                    r: "45",
                                                    fill: "none",
                                                    stroke: "url(#emerald-gradient)",
                                                    stroke_width: "10",
                                                    stroke_dasharray: "{(2.0 * 3.14159 * 45.0) * (cache.hit_rate * 100.0 / 100.0)} {(2.0 * 3.14159 * 45.0)}",
                                                    stroke_linecap: "round",
                                                    class: "transition-all duration-700 ease-out"
                                                }
                                            }

                                            // Center content
                                            div {
                                                class: "absolute inset-0 flex flex-col items-center justify-center",
                                                div { class: "text-4xl font-bold text-emerald-400", "{cache.hit_rate * 100.0:.1}" }
                                                div { class: "text-sm text-slate-400", "Hit Rate %" }
                                            }
                                        }
                                    }

                                    // Cache Stats Grid
                                    div { class: "grid grid-cols-2 gap-3",
                                        StatBox {
                                            label: "Hits",
                                            value: format!("{}", cache.hits),
                                            color: "emerald"
                                        }
                                        StatBox {
                                            label: "Misses",
                                            value: format!("{}", cache.misses),
                                            color: "rose"
                                        }
                                        StatBox {
                                            label: "Evictions",
                                            value: format!("{}", cache.evictions),
                                            color: "amber"
                                        }
                                        StatBox {
                                            label: "Nodes",
                                            value: format!("{}", cache.node_count),
                                            color: "blue"
                                        }
                                    }

                                    // Memory Usage Progress Bars
                                    div { class: "space-y-3",
                                        // Total Occupancy
                                        ProgressBar {
                                            label: "Total Occupancy",
                                            value: cache.total_cached,
                                            max: cache.total_capacity,
                                            color: "blue",
                                            show_percentage: true
                                        }

                                        // Protected vs Evictable
                                        div { class: "grid grid-cols-2 gap-4",
                                            div {
                                                ProgressBar {
                                                    label: "Protected",
                                                    value: cache.protected_size,
                                                    max: cache.total_cached,
                                                    color: "purple",
                                                    show_percentage: true
                                                }
                                            }
                                            div {
                                                ProgressBar {
                                                    label: "Evictable",
                                                    value: cache.evictable_size,
                                                    max: cache.total_cached,
                                                    color: "emerald",
                                                    show_percentage: true
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                div { class: "text-center text-slate-500 py-12", "No cache metrics available" }
                            }
                        }

                        // Engine Performance Panel
                        MetricPanel {
                            title: "Engine Performance",
                            icon: "⚙️",
                            color: "blue",

                            if let Some(engine) = &m.engine {
                                div { class: "space-y-6",
                                    // Timing Metrics
                                    div { class: "grid grid-cols-3 gap-4",
                                        TimingCard {
                                            label: "Avg Queue",
                                            value_ms: engine.avg_queue_time_ms,
                                            color: "blue"
                                        }
                                        TimingCard {
                                            label: "Avg Prefill",
                                            value_ms: engine.avg_prefill_time_ms,
                                            color: "emerald"
                                        }
                                        TimingCard {
                                            label: "Avg Decode",
                                            value_ms: engine.avg_decode_time_ms,
                                            color: "purple"
                                        }
                                    }

                                    // Request Statistics
                                    div {
                                        div { class: "text-sm text-slate-400 mb-3 font-medium", "Request Statistics" }
                                        div { class: "grid grid-cols-2 gap-3",
                                            StatBox {
                                                label: "Completed",
                                                value: format!("{}", engine.completed_requests),
                                                color: "emerald"
                                            }
                                            StatBox {
                                                label: "Failed",
                                                value: format!("{}", engine.failed_requests),
                                                color: "rose"
                                            }
                                            StatBox {
                                                label: "Concurrent",
                                                value: format!("{}", engine.concurrent_requests),
                                                color: "amber"
                                            }
                                            StatBox {
                                                label: "Success Rate",
                                                value: if engine.total_requests > 0 {
                                                    format!("{:.1}%", (engine.completed_requests as f64 / engine.total_requests as f64) * 100.0)
                                                } else {
                                                    "N/A".to_string()
                                                },
                                                color: "emerald"
                                            }
                                        }
                                    }

                                    // Throughput Indicator
                                    if engine.total_tokens_generated > 0 {
                                        div {
                                            class: "bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-lg p-4",
                                            div { class: "flex items-center justify-between",
                                                div {
                                                    span { class: "text-slate-400 text-sm", "Throughput" }
                                                    div { class: "text-2xl font-bold text-blue-400 mt-1",
                                                        "{(engine.total_tokens_generated as f64 / (engine.avg_prefill_time_ms + engine.avg_decode_time_ms).max(1.0) * 1000.0):.1}"
                                                    }
                                                    div { class: "text-xs text-slate-500", "tokens/second" }
                                                }
                                                div { class: "text-4xl", "🚀" }
                                            }
                                        }
                                    }
                                }
                            } else {
                                div { class: "text-center text-slate-500 py-12", "No engine metrics available" }
                            }
                        }
                    }

                    // System Resources Section
                    div { class: "grid grid-cols-1 md:grid-cols-3 gap-6",
                        // CPU Gauge
                        ResourceGauge {
                            title: "CPU Usage",
                            icon: "💻",
                            value: m.cpu.utilization_percent as f64,
                            max: 100.0,
                            subtitle: format!("{} cores", m.cpu.core_count),
                            color: if m.cpu.utilization_percent > 80.0 { "rose" }
                                   else if m.cpu.utilization_percent > 50.0 { "amber" }
                                   else { "emerald" }
                        }

                        // Memory Gauge
                        ResourceGauge {
                            title: "Memory Usage",
                            icon: "💾",
                            value: (m.memory.used_mb as f64 / m.memory.total_mb as f64) * 100.0,
                            max: 100.0,
                            subtitle: format!("{} / {} MB", m.memory.used_mb, m.memory.total_mb),
                            color: if m.memory.used_mb * 100 / m.memory.total_mb > 80 { "rose" }
                                   else if m.memory.used_mb * 100 / m.memory.total_mb > 50 { "amber" }
                                   else { "emerald" }
                        }

                        // GPU Gauge (if available)
                        if let Some(gpu) = &m.gpu {
                            ResourceGauge {
                                title: "GPU Usage",
                                icon: "🎮",
                                value: gpu.utilization_percent as f64,
                                max: 100.0,
                                subtitle: format!("{} / {} MB", gpu.memory_used_mb, gpu.memory_total_mb),
                                color: if gpu.utilization_percent > 80.0 { "rose" }
                                       else if gpu.utilization_percent > 50.0 { "amber" }
                                       else { "emerald" }
                            }
                        } else {
                            div {
                                class: "bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-6 flex items-center justify-center",
                                div { class: "text-center text-slate-500", "No GPU detected" }
                            }
                        }
                    }
                }
            } else {
                // Loading state
                div {
                    class: "flex items-center justify-center h-96",
                    div { class: "text-center",
                        div {
                            class: "text-8xl mb-6 animate-bounce",
                            "⏳"
                        }
                        div { class: "text-xl text-slate-400", "Loading metrics..." }
                    }
                }
            }
        }
    }
}

// ==================== Component: Gradient Card ====================
#[component]
fn GradientCard(
    title: String,
    value: String,
    subtitle: String,
    gradient: String,
    border_color: String,
    text_color: String,
    icon: String,
) -> Element {
    rsx! {
        div {
            class: "bg-gradient-to-br {gradient} border {border_color} rounded-xl p-5 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-purple-500/10",

            div { class: "flex items-center justify-between mb-3",
                div { class: "text-sm text-slate-300 font-medium", "{title}" }
                div { class: "text-3xl drop-shadow-lg", "{icon}" }
            }
            div { class: "text-4xl font-bold {text_color} mb-2", "{value}" }
            div { class: "text-xs text-slate-400", "{subtitle}" }
        }
    }
}

// ==================== Component: Metric Panel ====================
#[component]
fn MetricPanel(title: String, icon: String, color: String, children: Element) -> Element {
    let color_class = match color.as_str() {
        "emerald" => "text-emerald-400",
        "blue" => "text-blue-400",
        "purple" => "text-purple-400",
        _ => "text-slate-400",
    };

    rsx! {
        div {
            class: "bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-slate-600/50 transition-all duration-300",

            div { class: "flex items-center gap-3 mb-5 border-b border-slate-700 pb-4",
                div { class: "text-3xl", "{icon}" }
                h3 { class: "text-xl font-semibold", "{title}" }
            }

            {children}
        }
    }
}

// ==================== Component: Stat Box ====================
#[component]
fn StatBox(label: String, value: String, color: String) -> Element {
    let text_color = match color.as_str() {
        "emerald" => "text-emerald-400",
        "blue" => "text-blue-400",
        "purple" => "text-purple-400",
        "rose" => "text-rose-400",
        "amber" => "text-amber-400",
        _ => "text-slate-400",
    };

    let bg_color = match color.as_str() {
        "emerald" => "bg-emerald-500/10",
        "blue" => "bg-blue-500/10",
        "purple" => "bg-purple-500/10",
        "rose" => "bg-rose-500/10",
        "amber" => "bg-amber-500/10",
        _ => "bg-slate-500/10",
    };

    rsx! {
        div {
            class: "{bg_color} rounded-lg p-3 transition-all hover:scale-105 duration-200",
            div { class: "text-xs text-slate-400 mb-1", "{label}" }
            div { class: "text-xl font-bold {text_color}", "{value}" }
        }
    }
}

// ==================== Component: Progress Bar ====================
#[component]
fn ProgressBar(label: String, value: usize, max: usize, color: String, show_percentage: bool) -> Element {
    let percentage = if max > 0 {
        ((value as f64 / max as f64) * 100.0).min(100.0)
    } else {
        0.0
    };

    let bg_color = match color.as_str() {
        "blue" => "bg-blue-500",
        "emerald" => "bg-emerald-500",
        "purple" => "bg-purple-500",
        "rose" => "bg-rose-500",
        "amber" => "bg-amber-500",
        _ => "bg-slate-500",
    };

    let percentage_text = if show_percentage {
        format!("({:.1}%)", percentage)
    } else {
        String::new()
    };

    rsx! {
        div { class: "w-full",
            div { class: "flex justify-between text-xs text-slate-400 mb-2",
                span { "{label}" }
                span { "{value} / {max} {percentage_text}" }
            }
            div { class: "w-full bg-slate-700 rounded-full h-3 overflow-hidden shadow-inner",
                div {
                    class: "{bg_color} h-3 rounded-full transition-all duration-700 ease-out shadow-lg",
                    style: "width: {percentage}%"
                }
            }
        }
    }
}

// ==================== Component: Timing Card ====================
#[component]
fn TimingCard(label: String, value_ms: f64, color: String) -> Element {
    let text_color = match color.as_str() {
        "blue" => "text-blue-400",
        "emerald" => "text-emerald-400",
        "purple" => "text-purple-400",
        _ => "text-slate-400",
    };

    let bg_gradient = match color.as_str() {
        "blue" => "from-blue-500/10 to-cyan-500/10 border-blue-500/20",
        "emerald" => "from-emerald-500/10 to-green-500/10 border-emerald-500/20",
        "purple" => "from-purple-500/10 to-pink-500/10 border-purple-500/20",
        _ => "from-slate-500/10 to-slate-600/10 border-slate-500/20",
    };

    rsx! {
        div {
            class: "bg-gradient-to-br {bg_gradient} border rounded-lg p-4 text-center transition-all hover:scale-105 duration-200",
            div { class: "text-xs text-slate-400 mb-1", "{label}" }
            div { class: "text-3xl font-bold {text_color}", "{value_ms:.1}" }
            div { class: "text-xs text-slate-500 mt-1", "milliseconds" }
        }
    }
}

// ==================== Component: Resource Gauge ====================
#[component]
fn ResourceGauge(title: String, icon: String, value: f64, max: f64, subtitle: String, color: String) -> Element {
    let percentage = (value / max * 100.0).min(100.0);

    let stroke_color = match color.as_str() {
        "emerald" => "#10b981",
        "amber" => "#f59e0b",
        "rose" => "#f43f5e",
        _ => "#64748b",
    };

    let circumference = 2.0 * 3.14159 * 45.0;
    let progress = circumference * (percentage / 100.0);

    rsx! {
        div {
            class: "bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-slate-600/50 transition-all duration-300",

            div { class: "flex flex-col items-center space-y-4",
                // SVG Gauge
                div { class: "relative w-32 h-32",
                    svg {
                        class: "transform -rotate-90",
                        view_box: "0 0 100 100",

                        // Background circle
                        circle {
                            cx: "50",
                            cy: "50",
                            r: "45",
                            fill: "none",
                            stroke: "#1e293b",
                            stroke_width: "10"
                        }

                        // Progress circle
                        circle {
                            cx: "50",
                            cy: "50",
                            r: "45",
                            fill: "none",
                            stroke: "{stroke_color}",
                            stroke_width: "10",
                            stroke_dasharray: "{progress} {circumference}",
                            stroke_linecap: "round",
                            class: "transition-all duration-700 ease-out"
                        }
                    }

                    // Center text
                    div {
                        class: "absolute inset-0 flex flex-col items-center justify-center",
                        div { class: "text-3xl font-bold", style: "color: {stroke_color}", "{value:.0}" }
                        div { class: "text-xs text-slate-400 mt-1", "%" }
                    }
                }

                // Title and subtitle
                div { class: "text-center",
                    div { class: "flex items-center justify-center gap-2 mb-1",
                        div { class: "text-xl", "{icon}" }
                        div { class: "font-semibold text-slate-200", "{title}" }
                    }
                    div { class: "text-sm text-slate-400", "{subtitle}" }
                }
            }
        }
    }
}

// ==================== Helper: Format large numbers ====================
fn format_large_number(num: u64) -> String {
    if num >= 1_000_000_000 {
        format!("{:.2}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.2}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.2}K", num as f64 / 1_000.0)
    } else {
        format!("{}", num)
    }
}
