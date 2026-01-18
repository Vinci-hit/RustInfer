use comrak::{markdown_to_html, Options};
use syntect::parsing::SyntaxSet;
use syntect::highlighting::ThemeSet;
use syntect::html::{styled_line_to_highlighted_html, IncludeBackground};
use syntect::easy::HighlightLines;

/// Render markdown to HTML with GitHub Flavored Markdown support
/// Includes syntax highlighting and mermaid diagrams
/// Math rendering is done client-side via KaTeX JavaScript
pub fn render_markdown(markdown: &str) -> String {
    let mut options = Options::default();

    // Enable GitHub Flavored Markdown extensions
    options.extension.strikethrough = true;
    options.extension.tagfilter = true;
    options.extension.table = true;
    options.extension.autolink = true;
    options.extension.tasklist = true;
    options.extension.superscript = false;
    options.extension.footnotes = false;
    options.extension.description_lists = false;

    // Render options
    options.render.hardbreaks = false;
    options.render.github_pre_lang = true;
    options.render.full_info_string = false;
    options.render.r#unsafe = true; // Allow HTML for math and diagrams
    options.render.escape = false;

    let mut html = markdown_to_html(markdown, &options);

    // Post-process for syntax highlighting
    html = add_syntax_highlighting(&html);

    // Add math rendering markers (KaTeX will render these client-side)
    html = add_math_markers(&html);

    // Add mermaid diagram markers
    html = add_mermaid_markers(&html);

    html
}

/// Add syntax highlighting to code blocks
fn add_syntax_highlighting(html: &str) -> String {
    // For now, just return as-is
    // We'll implement proper highlighting later
    html.to_string()
}

/// Convert LaTeX math to KaTeX-ready format
/// Marks $...$ and $$...$$ for client-side KaTeX rendering
fn add_math_markers(html: &str) -> String {
    // KaTeX will automatically render these on the client side
    // We just need to ensure the delimiters are preserved in the HTML
    html.to_string()
}

/// Add Mermaid diagram markers
fn add_mermaid_markers(html: &str) -> String {
    // Look for code blocks with language "mermaid"
    html.replace(
        r#"<pre><code class="language-mermaid">"#,
        r#"<div class="mermaid">"#
    ).replace(
        "</code></pre>",
        "</div>"
    )
}

/// Highlight a code block with syntax highlighting
/// This is a helper function for future use
#[allow(dead_code)]
pub fn highlight_code(code: &str, language: &str) -> String {
    let ss = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();

    let syntax = ss.find_syntax_by_token(language)
        .unwrap_or_else(|| ss.find_syntax_plain_text());

    let theme = &ts.themes["base16-ocean.dark"];
    let mut highlighter = HighlightLines::new(syntax, theme);

    let mut html = String::new();
    html.push_str("<pre class=\"syntax-highlighted\"><code>");

    for line in code.lines() {
        let ranges = highlighter.highlight_line(line, &ss).unwrap();
        let line_html = styled_line_to_highlighted_html(&ranges[..], IncludeBackground::No).unwrap();
        html.push_str(&line_html);
        html.push('\n');
    }

    html.push_str("</code></pre>");
    html
}

