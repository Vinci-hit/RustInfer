use super::icon::Icon;
use crate::{
    api::client::ApiClient,
    state::workspace::{Connection, Workspace},
};
use dioxus::core::Task;
use dioxus::prelude::*;

#[component]
pub fn MetricsPanel() -> Element {
    let workspace = use_context::<Workspace>();
    let mut metrics = use_signal(|| None::<crate::state::metrics::SystemMetrics>);
    let mut failed = use_signal(|| false);
    let mut poll_task = use_signal(|| None::<Task>);
    let endpoint = use_memo(move || workspace.settings.read().api_base_url.clone());
    use_effect(move || {
        let base = endpoint();
        let connected = matches!((workspace.connection)(), Connection::Online);
        if let Some(task) = poll_task.take() {
            task.cancel();
        }
        metrics.set(None);
        failed.set(false);
        if connected {
            let task = spawn(async move {
                let client = ApiClient::new(&base);
                loop {
                    match client.get_metrics().await {
                        Ok(data) => {
                            metrics.set(Some(data));
                            failed.set(false);
                        }
                        Err(_) => {
                            metrics.set(None);
                            failed.set(true);
                        }
                    }
                    gloo_timers::future::TimeoutFuture::new(5000).await;
                }
            });
            poll_task.set(Some(task));
        }
    });
    rsx! {
        section { class: "inspector-section runtime-section",
            div { class: "section-heading", h3 { "运行状态" } Icon { name: "cpu", size: 16 } }
            if let Some(data) = metrics() {
                if let Some(cpu) = data.cpu { MetricBar { label: "CPU", value: cpu.utilization_percent, detail: format!("{} 核心", cpu.core_count) } }
                if let Some(memory) = data.memory {
                    MetricBar { label: "内存", value: if memory.total_mb > 0 { memory.used_mb as f32 / memory.total_mb as f32 * 100.0 } else { 0.0 }, detail: format!("{:.1} / {:.1} GB", memory.used_mb as f64 / 1024.0, memory.total_mb as f64 / 1024.0) }
                }
                if let Some(gpu) = data.gpu {
                    MetricBar { label: "GPU", value: gpu.utilization_percent, detail: gpu.temperature_celsius.map(|t|format!("{t:.0} °C")).unwrap_or_else(||"计算利用率".into()) }
                    MetricBar { label: "显存", value: if gpu.memory_total_mb > 0 {gpu.memory_used_mb as f32 / gpu.memory_total_mb as f32 * 100.0} else {0.0}, detail: format!("{:.1} / {:.1} GB", gpu.memory_used_mb as f64 / 1024.0, gpu.memory_total_mb as f64 / 1024.0) }
                }
                if let Some(seconds) = data.uptime_secs { div { class: "uptime", span { "已运行" } span { "{seconds / 3600} 时 {(seconds % 3600) / 60} 分" } } }
                p { class: "metrics-hint", span { class: "status-dot online" } "每 5 秒更新" }
            } else {
                div { class: "metrics-empty", Icon { name: "cpu", size: 24 }
                    p { if failed() { "暂时无法获取运行数据" } else if matches!((workspace.connection)(), Connection::Online) { "正在读取运行数据…" } else { "连接后查看实时资源使用" } }
                }
            }
        }
    }
}

#[component]
fn MetricBar(label: &'static str, value: f32, detail: String) -> Element {
    let percent = if value.is_finite() {
        value.clamp(0.0, 100.0)
    } else {
        0.0
    };
    rsx! { div { class: "metric-row", div { strong { "{label}" } span { "{percent:.0}%" } }
        div { class: "metric-track", role: "meter", aria_label: label, aria_valuemin: "0", aria_valuemax: "100", aria_valuenow: "{percent}", span { style: "width: {percent}%" } }
        small { "{detail}" }
    } }
}
