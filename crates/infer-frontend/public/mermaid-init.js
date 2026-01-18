// Mermaid initialization and rendering functions

let mermaidInitialized = false;

export function initMermaid() {
    if (mermaidInitialized) return;

    // Load Mermaid from CDN if not already loaded
    if (typeof window.mermaid === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js';
        script.onload = () => {
            window.mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                themeVariables: {
                    primaryColor: '#3b82f6',
                    primaryTextColor: '#fff',
                    primaryBorderColor: '#1e40af',
                    lineColor: '#6b7280',
                    secondaryColor: '#10b981',
                    tertiaryColor: '#8b5cf6',
                    background: '#1f2937',
                    mainBkg: '#1f2937',
                    secondBkg: '#374151',
                    tertiaryBkg: '#4b5563',
                },
            });
            mermaidInitialized = true;
        };
        document.head.appendChild(script);
    } else {
        window.mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            themeVariables: {
                primaryColor: '#3b82f6',
                primaryTextColor: '#fff',
                primaryBorderColor: '#1e40af',
                lineColor: '#6b7280',
                secondaryColor: '#10b981',
                tertiaryColor: '#8b5cf6',
                background: '#1f2937',
                mainBkg: '#1f2937',
                secondBkg: '#374151',
                tertiaryBkg: '#4b5563',
            },
        });
        mermaidInitialized = true;
    }
}

export function renderMermaid(elementId) {
    if (!mermaidInitialized || typeof window.mermaid === 'undefined') {
        // Retry after a delay
        setTimeout(() => renderMermaid(elementId), 100);
        return;
    }

    const element = document.getElementById(elementId);
    if (element) {
        try {
            window.mermaid.run({
                nodes: [element],
            });
        } catch (error) {
            console.error('Mermaid rendering error:', error);
            element.innerHTML = `<div class="text-red-400">Error rendering diagram: ${error.message}</div>`;
        }
    }
}
