// KaTeX initialization and rendering functions

let katexLoaded = false;

export function initKaTeX() {
    if (katexLoaded) return;

    // Load KaTeX from CDN if not already loaded
    if (typeof window.katex === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
        script.integrity = 'sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8';
        script.crossOrigin = 'anonymous';
        script.onload = () => {
            loadAutoRender();
        };
        document.head.appendChild(script);
    } else {
        loadAutoRender();
    }
}

function loadAutoRender() {
    if (typeof window.renderMathInElement === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js';
        script.integrity = 'sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05';
        script.crossOrigin = 'anonymous';
        script.onload = () => {
            katexLoaded = true;
            renderAllMath();
        };
        document.head.appendChild(script);
    } else {
        katexLoaded = true;
        renderAllMath();
    }
}

function renderAllMath() {
    if (!katexLoaded || typeof window.renderMathInElement === 'undefined') {
        setTimeout(renderAllMath, 100);
        return;
    }

    // Render all math in markdown-content elements
    const elements = document.querySelectorAll('.markdown-content');
    elements.forEach(element => {
        try {
            window.renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false,
                errorColor: '#ff6b6b',
                strict: false,
            });
        } catch (error) {
            console.error('KaTeX rendering error:', error);
        }
    });
}

export function renderMathInElement(elementId) {
    if (!katexLoaded) {
        initKaTeX();
        setTimeout(() => renderMathInElement(elementId), 200);
        return;
    }

    const element = document.getElementById(elementId);
    if (element && typeof window.renderMathInElement !== 'undefined') {
        try {
            window.renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                ],
                throwOnError: false,
            });
        } catch (error) {
            console.error('KaTeX rendering error:', error);
        }
    }
}

// Auto-render on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initKaTeX);
} else {
    initKaTeX();
}

// Re-render when new content is added (for dynamically added messages)
const observer = new MutationObserver(() => {
    if (katexLoaded) {
        renderAllMath();
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true,
});
