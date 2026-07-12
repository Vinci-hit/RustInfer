use pulldown_cmark::{html, CowStr, Event, Options, Parser, Tag};

const BLOCKED_URL: &str = "#";

/// 将 Markdown 文本转为 HTML
pub fn render_markdown(text: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(text, options).map(sanitize_event);
    let mut html_output = String::new();
    html::push_html(&mut html_output, parser);
    html_output
}

fn sanitize_event(event: Event<'_>) -> Event<'_> {
    match event {
        // Raw model-authored HTML must be displayed as text. Passing either
        // variant through would let scripts and event-handler attributes reach
        // `dangerous_inner_html` in the message component.
        Event::Html(raw) | Event::InlineHtml(raw) => Event::Text(raw),
        Event::Start(Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Link {
            link_type,
            dest_url: safe_url(dest_url),
            title,
            id,
        }),
        Event::Start(Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Image {
            link_type,
            dest_url: safe_url(dest_url),
            title,
            id,
        }),
        event => event,
    }
}

fn safe_url(url: CowStr<'_>) -> CowStr<'_> {
    if is_safe_url(&url) {
        url
    } else {
        CowStr::Borrowed(BLOCKED_URL)
    }
}

fn is_safe_url(url: &str) -> bool {
    // Browsers ignore embedded ASCII whitespace/control characters in parts of
    // URL parsing. Remove them before scheme detection so `java\nscript:` and
    // similar obfuscations cannot bypass the allowlist.
    let normalized: String = url
        .chars()
        .filter(|character| !character.is_ascii_control() && !character.is_ascii_whitespace())
        .flat_map(char::to_lowercase)
        .collect();

    if normalized.is_empty()
        || normalized.starts_with('#')
        || normalized.starts_with('/')
        || normalized.starts_with("./")
        || normalized.starts_with("../")
        || normalized.starts_with('?')
    {
        return true;
    }

    let scheme_end = normalized
        .find([':', '/', '?', '#'])
        .filter(|index| normalized.as_bytes()[*index] == b':');

    match scheme_end {
        Some(index) => matches!(&normalized[..index], "http" | "https" | "mailto" | "tel"),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::{is_safe_url, render_markdown};

    #[test]
    fn escapes_raw_html_from_model_output() {
        let rendered =
            render_markdown("hello <script>alert(1)</script> <img src=x onerror=alert(2)> goodbye");

        assert!(!rendered.contains("<script"));
        assert!(!rendered.contains("<img"));
        assert!(rendered.contains("&lt;script&gt;"));
        assert!(rendered.contains("&lt;img src=x onerror=alert(2)&gt;"));
    }

    #[test]
    fn blocks_active_content_url_schemes() {
        for markdown in [
            "[click](javascript:alert(1))",
            "[click](JaVaScRiPt:alert(1))",
            "[click](javascript&colon;alert(1))",
            "![image](data:image/svg+xml,<svg/onload=alert(1)>)",
            "[click](vbscript:msgbox(1))",
        ] {
            let rendered = render_markdown(markdown);
            assert!(!rendered.to_ascii_lowercase().contains("javascript:"));
            assert!(!rendered.to_ascii_lowercase().contains("data:image"));
            assert!(!rendered.to_ascii_lowercase().contains("vbscript:"));
        }
    }

    #[test]
    fn blocks_obfuscated_active_schemes() {
        assert!(!is_safe_url("java\nscript:alert(1)"));
        assert!(!is_safe_url("\0javascript:alert(1)"));
        assert!(is_safe_url("https://example.com/path?q=hello world"));
    }

    #[test]
    fn preserves_safe_and_relative_links() {
        let rendered = render_markdown(
            "[web](https://example.com) [mail](mailto:team@example.com) [docs](/docs)",
        );

        assert!(rendered.contains("href=\"https://example.com\""));
        assert!(rendered.contains("href=\"mailto:team@example.com\""));
        assert!(rendered.contains("href=\"/docs\""));
    }
}
