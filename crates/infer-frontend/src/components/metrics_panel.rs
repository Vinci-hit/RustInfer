use dioxus::prelude::*;
use crate::api::client::ApiClient;
use crate::state::metrics::SystemMetrics;

pub fn MetricsPanel() -> Element {
    let mut metrics = use_signal(|| None::<SystemMetrics>);
    let api_client = use_signal(|| ApiClient::new());

    // Poll every 2 seconds
    use_future(move || async move {
        loop {
            if let Ok(sys_metrics) = api_client.read().get_system_metrics().await {
                metrics.set(Some(sys_metrics));
            }
            gloo_timers::future::TimeoutFuture::new(2000).await;
        }
    });

    rsx! {
        div {
            class: "bg-gray-800 rounded-lg p-4 shadow-lg h-full",

            h2 { class: "text-xl font-bold mb-4 border-b border-gray-700 pb-2", "System Metrics" }

            if let Some(m) = metrics.read().as_ref() {
                div { class: "space-y-4",
                    // CPU
                    div {
                        div { class: "text-sm text-gray-400 mb-1", "CPU Usage" }
                        div { class: "text-3xl font-bold text-blue-400", "{m.cpu.utilization_percent:.1}%" }
                        div { class: "text-xs text-gray-500", "{m.cpu.core_count} cores" }
                    }

                    // Memory
                    div {
                        div { class: "text-sm text-gray-400 mb-1", "Memory" }
                        div { class: "text-3xl font-bold text-green-400",
                            "{m.memory.used_mb} MB"
                        }
                        div { class: "text-xs text-gray-500",
                            "{m.memory.used_mb * 100 / m.memory.total_mb}% of {m.memory.total_mb} MB"
                        }
                    }

                    // GPU
                    if let Some(gpu) = &m.gpu {
                        div {
                            div { class: "text-sm text-gray-400 mb-1", "GPU Usage" }
                            div { class: "text-3xl font-bold text-purple-400", "{gpu.utilization_percent:.1}%" }

                            div { class: "text-xs text-gray-500 mt-1",
                                "Memory: {gpu.memory_used_mb} / {gpu.memory_total_mb} MB"
                            }

                            if let Some(temp) = gpu.temperature_celsius {
                                div { class: "text-xs text-gray-500",
                                    "Temp: {temp:.1}°C"
                                }
                            }
                        }
                    }
                }
            } else {
                div { class: "text-gray-500 text-center py-8",
                    div { class: "animate-pulse", "Loading metrics..." }
                }
            }
        }
    }
}
