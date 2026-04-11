// ======================================================================
// NOVA – Markdown Renderer & Text Utilities Module
// Extracted from script.js for maintainability
// ======================================================================

/**
 * Render markdown text to HTML with code block highlighting.
 * Supports: headings, bold, italic, inline code, links, lists, code blocks.
 * @param {string} text - Raw markdown text
 * @returns {string} HTML string
 */
function renderMarkdown(text) {
    if (!text) return '';

    // Split into code-block segments and normal segments
    const parts = [];
    let lastIndex = 0;
    const codeBlockRe = /```(\w*)\n?([\s\S]*?)```/g;
    let match;

    while ((match = codeBlockRe.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parts.push({ type: 'text', content: text.slice(lastIndex, match.index) });
        }
        parts.push({ type: 'code', lang: match[1] || 'plaintext', content: match[2] });
        lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
        parts.push({ type: 'text', content: text.slice(lastIndex) });
    }

    return parts.map(part => {
        if (part.type === 'code') {
            const lang = part.lang || 'plaintext';
            const escaped = part.content
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const highlighted = (typeof hljs !== 'undefined' && hljs.getLanguage(lang))
                ? hljs.highlight(part.content, { language: lang, ignoreIllegals: true }).value
                : escaped;
            return `<div class="code-block-wrapper">
                <div class="code-block-header">
                    <span class="code-lang">${lang}</span>
                    <button class="code-copy-btn" onclick="copyCode(this)" title="Copy code">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        Copy
                    </button>
                </div>
                <pre><code class="hljs language-${lang}">${highlighted}</code></pre>
            </div>`;
        }

        // Process inline text
        let html = part.content
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/^### (.+)$/gm, '<h5 class="md-h5">$1</h5>')
            .replace(/^## (.+)$/gm, '<h4 class="md-h4">$1</h4>')
            .replace(/^# (.+)$/gm, '<h3 class="md-h3">$1</h3>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
            .replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="md-link">$1</a>')
            .replace(/^---+$/gm, '<hr class="md-hr">')
            .replace(/^\d+\.\s+(.+)$/gm, '<li class="ol-item">$1</li>')
            .replace(/^[-•*]\s+(.+)$/gm, '<li class="ul-item">$1</li>');

        // Wrap consecutive list items
        html = html.replace(/(<li class="ul-item">.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
        html = html.replace(/(<li class="ol-item">.*<\/li>\n?)+/g, m => `<ol>${m}</ol>`);

        // Convert newlines to <br> (but not inside tags)
        html = html.replace(/\n/g, '<br>');
        return html;
    }).join('');
}

/**
 * Strip markdown formatting for TTS (text-to-speech) output.
 * @param {string} text - Markdown text
 * @returns {string} Plain text suitable for speech
 */
function stripForSpeech(text) {
    return text
        .replace(/```[\s\S]*?```/g, 'code block')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/\*(.+?)\*/g, '$1')
        .replace(/^[-•*]\s+/gm, '')
        .replace(/^\d+\.\s+/gm, '')
        .replace(/^#{1,3}\s+/gm, '')
        .replace(/<[^>]*>/g, '')
        .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
        .replace(/\n+/g, ' ')
        .trim();
}

/**
 * HTML-escape a string to prevent XSS.
 * @param {string} str - Raw string
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/**
 * Copy code block content to clipboard.
 * @param {HTMLElement} btn - The copy button element
 */
function copyCode(btn) {
    const code = btn.closest('.code-block-wrapper').querySelector('code');
    const text = code.innerText || code.textContent;
    navigator.clipboard.writeText(text).then(() => {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!`;
        btn.style.color = 'var(--core-cyan)';
        setTimeout(() => {
            btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy`;
            btn.style.color = '';
        }, 2000);
    }).catch(() => showToast('Copy failed — please copy manually', 'error'));
}

/**
 * Copy full message content to clipboard.
 * @param {HTMLElement} btn - The copy button element
 */
function copyMessage(btn) {
    const msgBody = btn.closest('.message').querySelector('.msg-body');
    const text = msgBody.innerText || msgBody.textContent;
    navigator.clipboard.writeText(text).then(() => {
        btn.title = 'Copied!';
        btn.style.color = 'var(--core-cyan)';
        showToast('Message copied to clipboard', 'success');
        setTimeout(() => { btn.title = 'Copy reply'; btn.style.color = ''; }, 2000);
    }).catch(() => showToast('Copy failed', 'error'));
}
