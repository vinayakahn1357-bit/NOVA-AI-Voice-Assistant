// ======================================================================
// NOVA – Skeleton Loader & Chat Bubble Utilities Module
// Extracted from script.js for maintainability
// ======================================================================

/**
 * Get a human-readable timestamp string.
 * @returns {string} Formatted time string like "2:30 PM"
 */
function getTimestamp() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Create a thinking/skeleton bubble while NOVA is processing.
 * Shows animated shimmer lines to indicate loading state.
 * @param {HTMLElement} chatBox - The chat container element
 * @returns {HTMLElement} The thinking bubble element
 */
function createThinkingBubble(chatBox) {
    const div = document.createElement('div');
    div.className = 'message nova skeleton-message';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="msg-avatar nova-avatar">N</div>
        <div class="msg-content">
            <div class="msg-header">
                <span class="sender">Nova</span>
                <span class="processing-label">Thinking…</span>
            </div>
            <div class="msg-body skeleton-body">
                <div class="skeleton-line skeleton-line-short"></div>
                <div class="skeleton-line skeleton-line-full"></div>
                <div class="skeleton-line skeleton-line-full"></div>
                <div class="skeleton-line skeleton-line-medium"></div>
                <div class="skeleton-line skeleton-line-short"></div>
            </div>
        </div>`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
}

/**
 * Append a user message bubble to the chat.
 * @param {HTMLElement} chatBox - The chat container
 * @param {string} message - User message text
 */
function appendUserMsg(chatBox, message) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.title = getTimestamp();
    div.innerHTML = `
        <div class="msg-avatar user-avatar">U</div>
        <div class="msg-content">
            <div class="msg-body">${renderMarkdown(message)}</div>
        </div>`;
    chatBox.appendChild(div);
}

/**
 * Append a Nova response bubble (empty, ready for content).
 * @param {HTMLElement} chatBox - The chat container
 * @param {string} [time] - Optional timestamp override
 * @returns {HTMLElement} The body element to fill with content
 */
function appendNovaBubble(chatBox, time) {
    const responseDiv = document.createElement('div');
    responseDiv.className = 'message nova';
    responseDiv.title = time || getTimestamp();
    const bodyEl = document.createElement('div');
    bodyEl.className = 'msg-body';

    const copyBtn = `<button class="msg-copy-btn" onclick="copyMessage(this)" title="Copy reply">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
    </button>`;

    responseDiv.innerHTML = `
        <div class="msg-avatar nova-avatar">N</div>
        <div class="msg-content">
            <div class="msg-header"><span class="sender">Nova</span>${copyBtn}</div>
        </div>`;
    responseDiv.querySelector('.msg-content').appendChild(bodyEl);
    chatBox.appendChild(responseDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return bodyEl;
}

/**
 * Append an error message bubble.
 * @param {HTMLElement} chatBox - The chat container
 * @param {string} [errorMsg] - Error text
 */
function appendErrorBubble(chatBox, errorMsg) {
    const errDiv = document.createElement('div');
    errDiv.className = 'message nova error-msg';
    const msg = errorMsg || 'Connection lost. Is the Flask server running?';
    errDiv.innerHTML = `
        <div class="msg-avatar nova-avatar" style="background:rgba(255,75,92,0.3);border-color:rgba(255,75,92,0.6);">!</div>
        <div class="msg-content">
            <div class="msg-header"><span class="sender" style="color:#ff4b5c;">System Error</span></div>
            <div class="msg-body">${escapeHtml(msg)}</div>
        </div>`;
    chatBox.appendChild(errDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
