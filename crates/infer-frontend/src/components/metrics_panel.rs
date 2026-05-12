use dioxus::prelude::*;
use crate::api::client::ApiClient;
use crate::state::metrics::SystemMetrics;

/// SVG 圆形进度环组件
fn progress_ring(percent: f32, color: &str, size: u32) -> String {
    let radius = (size as f32 - 8.0) / 2.0;
    let circumference = 2.0 * std::f32::consts::PI * radius;
    let offset = circumference - (percent / 100.0) * circumference;
    let center = size as f32 / 2.0;

    format!(
        r#"<svg width="{size}" height="{size}" class="transform -rotate-90">
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="currentColor" stroke-width="4" class="text-white/5"/>
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" class="transition-all duration-700 ease-out"/>
        </svg>"#
    )
}

#[component]
pub fn MetricsPanel() -> Element {
    let mut metrics = use_signal(|| None::<SystemMetrics>);
    let mut is_connected = use_signal(|| false);

    // Poll every 2 seconds
    use_future(move || async move {
        let client = ApiClient::default();
        loop {
            match client.get_metrics().await {
                Ok(sys_metrics) => {
                    metrics.set(Some(sys_metrics));
                    is_connected.set(true);
                }
                Err(_) => {
                    is_connected.set(false);
                }
            }
            gloo_timers::future::TimeoutFuture::new(2000).await;
        }
    });

    rsx! {
        div {
            class: "glass-panel rounded-2xl p-5 h-full flex flex-col overflow-y-auto",

            // Header
            div {
                class: "flex items-center justify-between mb-5",
                h2 {
                    class: "text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider",
                    "System"
                }
                div {
                    class: if is_connected() {
                        "w-2 h-2 rounded-full bg-[var(--color-success)]"
                    } else {
                        "w-2 h-2 rounded-full bg-[var(--color-error)]"
                    }
                }
            }

            if let Some(m) = metrics.read().as_ref() {
                div { class: "space-y-5 flex-1",

                    // CPU card
                    if let Some(cpu) = &m.cpu {
                        div {
                            class: "metric-card",

                            div { class: "flex items-center justify-between mb-3",
                                span { class: "text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider", "CPU" }
                                span { class: "text-xs text-[var(--color-text-muted)]", "{cpu.core_count} cores" }
                            }

                            div { class: "flex items-center gap-4",
                                div {
                                    class: "relative",
                                    dangerous_inner_html: "{progress_ring(cpu.utilization_percent, \"oklch(0.65 0.15 230)\", 56)}"
                                }
                                div {
                                    div { class: "text-2xl font-bold text-[var(--color-info)]",
                                        "{cpu.utilization_percent:.0}%"
                                    }
                                    div { class: "text-xs text-[var(--color-text-muted)]", "utilization" }
                                }
                            }
                        }
                    }

                    // Memory card
                    if let Some(mem) = &m.memory {
                        {
                            let mem_percent = if mem.total_mb > 0 {
                                (mem.used_mb as f32 / mem.total_mb as f32) * 100.0
                            } else {
                                0.0
                            };
                            rsx! {
                                div {
                                    class: "metric-card",

                                    div { class: "flex items-center justify-between mb-3",
                                        span { class: "text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider", "Memory" }
                                        span { class: "text-xs text-[var(--color-text-muted)]", "{mem.used_mb} / {mem.total_mb} MB" }
                                    }

                                    div { class: "flex items-center gap-4",
                                        div {
                                            class: "relative",
                                            dangerous_inner_html: "{progress_ring(mem_percent, \"oklch(0.72 0.19 145)\", 56)}"
                                        }
                                        div {
                                            div { class: "text-2xl font-bold text-[var(--color-success)]",
                                                "{mem_percent:.0}%"
                                            }
                                            div { class: "text-xs text-[var(--color-text-muted)]", "used" }
                                        }
                                    }

                                    // Memory bar
                                    div { class: "mt-3 h-1.5 rounded-full bg-white/5 overflow-hidden",
                                        div {
                                            class: "h-full rounded-full bg-gradient-to-r from-emerald-500 to-green-400 transition-all duration-700",
                                            style: "width: {mem_percent}%"
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // GPU card
                    if let Some(gpu) = &m.gpu {
                        {
                            let vram_percent = if gpu.memory_total_mb > 0 {
                                (gpu.memory_used_mb as f32 / gpu.memory_total_mb as f32) * 100.0
                            } else {
                                0.0
                            };
                            rsx! {
                                div {
                                    class: "metric-card",

                                    div { class: "flex items-center justify-between mb-3",
                                        span { class: "text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider", "GPU" }
                                        if let Some(temp) = gpu.temperature_celsius {
                                            span { class: "text-xs text-orange-400", "{temp:.0}°C" }
                                        }
                                    }

                                    div { class: "flex items-center gap-4",
                                        div {
                                            class: "relative",
                                            dangerous_inner_html: "{progress_ring(gpu.utilization_percent, \"oklch(0.65 0.18 300)\", 56)}"
                                        }
                                        div {
                                            div { class: "text-2xl font-bold text-purple-400",
                                                "{gpu.utilization_percent:.0}%"
                                            }
                                            div { class: "text-xs text-[var(--color-text-muted)]", "compute" }
                                        }
                                    }

                                    // VRAM bar
                                    div { class: "mt-3",
                                        div { class: "flex justify-between text-[10px] text-[var(--color-text-muted)] mb-1",
                                            span { "VRAM" }
                                            span { "{gpu.memory_used_mb} / {gpu.memory_total_mb} MB" }
                                        }
                                        div { class: "h-1.5 rounded-full bg-white/5 overflow-hidden",
                                            div {
                                                class: "h-full rounded-full bg-gradient-to-r from-purple-500 to-pink-400 transition-all duration-700",
                                                style: "width: {vram_percent}%"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Uptime
                    if let Some(uptime) = m.uptime_secs {
                        div {
                            class: "metric-card",
                            div { class: "flex items-center justify-between",
                                span { class: "text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider", "Uptime" }
                                span { class: "text-sm font-mono text-[var(--color-text-secondary)]",
                                    {format_uptime(uptime)}
                                }
                            }
                        }
                    }
                }
            } else {
                // Loading state
                div { class: "flex-1 flex flex-col items-center justify-center gap-3",
                    div { class: "w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin-slow" }
                    p { class: "text-xs text-[var(--color-text-muted)]", "Connecting..." }
                }
            }
        }
    }
}

fn format_uptime(secs: u64) -> String {
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}", minutes, seconds)
    }
}
