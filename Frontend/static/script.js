// =====================================================================
// NOVA – AI Assistant Script  v2.0
// =====================================================================

// ─── Session ID (unique per browser tab, persisted in sessionStorage) ──────
let novaSessionId = sessionStorage.getItem('nova_session_id');
if (!novaSessionId) {
    novaSessionId = 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2);
    sessionStorage.setItem('nova_session_id', novaSessionId);
}

// ─── View Navigation (unified) ─────────────────────────────────────────────────
function showView(viewName) {
    // Hide all views
    document.querySelectorAll('.app-view').forEach(v => {
        v.classList.remove('view-active');
        v.style.display = '';
    });

    // Show the requested view
    const target = document.getElementById('view-' + viewName);
    if (target) target.classList.add('view-active');

    // Update nav bar active state
    document.querySelectorAll('.bn-tab').forEach(t => t.classList.remove('active'));
    const navBtn = document.getElementById('bn-' + viewName);
    if (navBtn) navBtn.classList.add('active');

    // View-specific initialisers
    if (viewName === 'chat') {
        if (typeof loadNovaMemory === 'function') loadNovaMemory();
        if (typeof renderHistoryList === 'function') renderHistoryList();
    }
    if (viewName === 'dashboard') {
        if (typeof loadNovaMemory === 'function') loadNovaMemory();
        if (typeof updateSystemStats === 'function') updateSystemStats();
    }
    if (viewName === 'home') {
        if (typeof initHomeParticles === 'function') initHomeParticles();
    }
    if (viewName === 'voice') {
        if (typeof buildVoiceViewWaveform === 'function') buildVoiceViewWaveform();
        if (typeof initVoiceParticles === 'function') initVoiceParticles();
        if (typeof setVoiceViewState === 'function') setVoiceViewState('idle');
        if (typeof updateVoiceViewTranscript === 'function') updateVoiceViewTranscript('');
        // Clear chat log when opening voice view fresh
        const chatLog = document.getElementById('vv-chat-log');
        if (chatLog) chatLog.innerHTML = '';
        if (typeof patchRecognitionForVoiceView === 'function') patchRecognitionForVoiceView();
    } else {
        if (typeof stopVoiceParticles === 'function') stopVoiceParticles();
    }
}

// ─── Chat Sidebar Toggle (mobile) ────────────────────────────────────────────
function toggleChatSidebar() {
    const sidebar = document.getElementById('gc-sidebar');
    if (sidebar) sidebar.classList.toggle('open');
}

// ─── App State ───────────────────────────────────────────────────────────────
let inputMode = 'text';       // 'text' | 'voice'
let isStreamMode = true;      // use SSE streaming by default
let isGenerating = false;     // true while waiting for / receiving a response

// ─── Utilities ─────────────────────────────────────────────────────────────────

function getTimestamp() {
    return new Date().toLocaleString([], {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}

// ─── Markdown → HTML renderer (enhanced) ─────────────────────────────────────
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

// Convert markdown to plain text for TTS
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

// ─── Copy Code Button ─────────────────────────────────────────────────────────
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

// ─── Copy Message Button ──────────────────────────────────────────────────────
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

// ─── Toast Notification System ────────────────────────────────────────────────
function showToast(message, type = 'info') {
    // type: 'info' | 'success' | 'error' | 'warning'
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
    toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span class="toast-msg">${message}</span>`;
    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => toast.classList.add('toast-show'));

    // Remove after 3.5s
    setTimeout(() => {
        toast.classList.remove('toast-show');
        toast.classList.add('toast-hide');
        setTimeout(() => toast.remove(), 350);
    }, 3500);
}

// ─── Waveform ─────────────────────────────────────────────────────────────────
function createWaveform() {
    const waveformContainer = document.getElementById('waveform');
    if (!waveformContainer) return;
    waveformContainer.innerHTML = '';
    const numBars = 50;
    for (let i = 0; i < numBars; i++) {
        const bar = document.createElement('div');
        bar.className = 'wave-bar';
        const distanceToCenter = Math.abs((numBars / 2) - i);
        const baseHeight = Math.max(5, 45 - (distanceToCenter * 1.8));
        bar.style.animationDelay = `${Math.random() * 0.8}s`;
        bar.style.setProperty('--base-height', `${baseHeight}px`);
        bar.style.setProperty('--target-height', `${baseHeight + Math.random() * 15 + 10}px`);
        waveformContainer.appendChild(bar);
    }
}

// ─── Welcome Screen ───────────────────────────────────────────────────────────
function showWelcomeScreen() {
    // Gemini-style: welcome div is flex, messages area is hidden via :empty CSS
    const ws = document.getElementById('welcome-screen');
    const cb = document.getElementById('chat-box');
    if (ws) { ws.style.display = 'flex'; }
    if (cb) { cb.innerHTML = ''; } // clear messages so :empty CSS hides it
}

function hideWelcomeScreen() {
    const ws = document.getElementById('welcome-screen');
    if (ws) { ws.style.display = 'none'; }
    const cb = document.getElementById('chat-box');
    if (cb) { cb.style.display = 'flex'; }
}

function useChip(prompt) {
    const input = document.getElementById('user-input');
    if (input) {
        input.value = prompt;
        updateCharCounter();
        autoResize(input);
        sendMessage();
    }
}

// ─── History System ────────────────────────────────────────────────────────────
let chatSessions = JSON.parse(localStorage.getItem('nova_history') || '[]');
let currentSessionId = null;

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function saveToHistory(userMsg, novaReply) {
    if (!currentSessionId) {
        currentSessionId = generateId();
        const newSession = {
            id: currentSessionId,
            topic: userMsg.length > 50 ? userMsg.slice(0, 50) + '…' : userMsg,
            timestamp: new Date().toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
            messages: []
        };
        chatSessions.unshift(newSession);
    }

    const session = chatSessions.find(s => s.id === currentSessionId);
    if (session) {
        session.messages.push({ role: 'user', text: userMsg, time: getTimestamp() });
        session.messages.push({ role: 'nova', text: novaReply, time: getTimestamp() });
        // Update preview
        session.lastMsg = novaReply.slice(0, 80);
    }

    localStorage.setItem('nova_history', JSON.stringify(chatSessions));
    renderHistoryList();
}

function renderHistoryList() {
    const list = document.getElementById('history-list');
    if (!list) return;

    if (chatSessions.length === 0) {
        list.innerHTML = '<div class="history-empty">No history yet.<br>Start a conversation!</div>';
        return;
    }

    const clearAllBtn = `<button class="history-clear-all-btn" onclick="event.stopPropagation(); clearAllHistory()">🗑 Clear All</button>`;
    const items = chatSessions.map(session => `
        <div class="history-item ${session.id === currentSessionId ? 'active' : ''}">
            <div class="hi-main" onclick="loadHistorySession('${session.id}')">
                <div class="hi-topic">${escapeHtml(session.topic)}</div>
                <div class="hi-meta">${session.timestamp} · ${Math.ceil(session.messages.length / 2)} msg${session.messages.length > 2 ? 's' : ''}</div>
                <div class="hi-preview">${escapeHtml((session.lastMsg || '').slice(0, 65))}…</div>
            </div>
            <button class="hi-delete-btn" onclick="event.stopPropagation(); deleteHistorySession('${session.id}')" title="Delete">✕</button>
        </div>
    `).join('');

    list.innerHTML = clearAllBtn + items;
}

function escapeHtml(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function deleteHistorySession(sessionId) {
    chatSessions = chatSessions.filter(s => s.id !== sessionId);
    if (currentSessionId === sessionId) {
        currentSessionId = null;
        const chatBox = document.getElementById('chat-box');
        if (chatBox) chatBox.innerHTML = '';
        showWelcomeScreen();
    }
    localStorage.setItem('nova_history', JSON.stringify(chatSessions));
    renderHistoryList();
}

function clearAllHistory() {
    if (!confirm('Clear all chat history? This cannot be undone.')) return;
    chatSessions = [];
    currentSessionId = null;
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.innerHTML = '';
    showWelcomeScreen();
    localStorage.removeItem('nova_history');
    renderHistoryList();
    fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: novaSessionId }) }).catch(() => { });
}

function loadHistorySession(sessionId) {
    const session = chatSessions.find(s => s.id === sessionId);
    if (!session) return;
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = '';
    currentSessionId = sessionId;
    hideWelcomeScreen();

    session.messages.forEach(msg => {
        if (msg.role === 'user') {
            appendUserMsgStatic(chatBox, msg.text, msg.time);
        } else {
            appendNovaMsgStatic(chatBox, msg.text, msg.time);
        }
    });
    chatBox.scrollTop = chatBox.scrollHeight;
    renderHistoryList();
    toggleHistoryPanel();
}

function appendUserMsgStatic(chatBox, text, time) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.title = time || '';
    div.innerHTML = `
        <div class="msg-avatar user-avatar">U</div>
        <div class="msg-content">
            <div class="msg-body">${renderMarkdown(text)}</div>
        </div>`;
    chatBox.appendChild(div);
}

function appendNovaMsgStatic(chatBox, text, time) {
    const div = document.createElement('div');
    div.className = 'message nova';
    div.title = time || '';
    div.innerHTML = `
        <div class="msg-avatar nova-avatar">N</div>
        <div class="msg-content">
            <div class="msg-header">
                <span class="sender">Nova</span>
                <button class="msg-copy-btn" onclick="copyMessage(this)" title="Copy reply">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                </button>
            </div>
            <div class="msg-body">${renderMarkdown(text)}</div>
        </div>`;
    chatBox.appendChild(div);
}

function closeHistoryPanel() {
    const panel = document.getElementById('history-panel');
    if (panel) panel.classList.remove('open');
}

function toggleHistoryPanel() {
    const panel = document.getElementById('history-panel');
    if (!panel) return;
    if (panel.classList.contains('open')) {
        panel.classList.remove('open');
    } else {
        renderHistoryList();
        panel.classList.add('open');
    }
}

function startNewConversation() {
    currentSessionId = null;
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.innerHTML = '';
    showWelcomeScreen();
    stopSpeaking();
    fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: novaSessionId })
    }).catch(() => { });
    renderHistoryList();
    showToast('New conversation started', 'info');
}

// ─── Typewriter Effect ────────────────────────────────────────────────────────
function typewriterEffect(element, rawText, onComplete) {
    element.innerHTML = '';
    element.classList.add('typing-cursor');
    let i = 0;
    // Faster speed — 8ms per char (was 16ms)
    const speed = 8;

    function tick() {
        if (i < rawText.length) {
            element.textContent = rawText.slice(0, i + 1);
            i++;
            setTimeout(tick, speed);
        } else {
            element.innerHTML = renderMarkdown(rawText);
            element.classList.remove('typing-cursor');
            // Run hljs on new code blocks
            element.querySelectorAll('pre code').forEach(block => {
                if (typeof hljs !== 'undefined') hljs.highlightElement(block);
            });
            if (onComplete) onComplete();
        }
    }
    tick();
}

// ─── DOM Helpers ──────────────────────────────────────────────────────────────
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

function createThinkingBubble(chatBox) {
    const div = document.createElement('div');
    div.className = 'message nova';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="msg-avatar nova-avatar">N</div>
        <div class="msg-content">
            <div class="msg-header"><span class="sender">Nova</span> <span class="processing-label">Thinking…</span></div>
            <div class="msg-body"><span class="thinking-dots"><span></span><span></span><span></span></span></div>
        </div>`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
}

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
}

function setSendState(sending) {
    isGenerating = sending;
    const btn = document.getElementById('send-btn');
    if (!btn) return;
    if (sending) {
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ff4b5c" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
        btn.title = 'Stop';
        btn.onclick = stopGeneration;
    } else {
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`;
        btn.title = 'Send (Enter)';
        btn.onclick = sendMessage;
    }
}

let abortController = null;
function stopGeneration() {
    if (abortController) abortController.abort();
    setSendState(false);
    showToast('Generation stopped', 'info');
}

// ─── Send Message (Text Mode — Streaming) ────────────────────────────────────
async function sendMessage() {
    if (isGenerating) return;
    inputMode = 'text';
    const inputField = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const message = inputField.value.trim();
    if (!message) return;

    stopSpeaking();
    hideWelcomeScreen();
    appendUserMsg(chatBox, message);
    inputField.value = '';
    updateCharCounter();
    autoResize(inputField);
    chatBox.scrollTop = chatBox.scrollHeight;

    setSendState(true);
    const typingDiv = createThinkingBubble(chatBox);

    try {
        const replyText = isStreamMode
            ? await sendStream(message, typingDiv, chatBox)
            : await sendFull(message, typingDiv, chatBox);

        if (replyText) saveToHistory(message, replyText);
    } catch (err) {
        typingDiv.remove();
        appendErrorBubble(chatBox, err.message);
    } finally {
        setSendState(false);
        chatBox.scrollTop = chatBox.scrollHeight;
        inputField.focus();
    }
}

// ─── Send from Voice (Voice Mode — WITH TTS, streaming sentence-by-sentence) ──
async function sendVoiceMessage(message) {
    if (isGenerating) return;
    inputMode = 'voice';
    const chatBox = document.getElementById('chat-box');
    hideWelcomeScreen();
    appendUserMsg(chatBox, message);
    chatBox.scrollTop = chatBox.scrollHeight;

    setSendState(true);
    const typingDiv = createThinkingBubble(chatBox);
    let bodyEl = null;

    try {
        const replyText = await sendVoiceStream(message, {
            onToken(fullSoFar, _token) {
                // Remove thinking bubble and create Nova bubble on first token
                if (!bodyEl) {
                    typingDiv.remove();
                    bodyEl = appendNovaBubble(chatBox);
                }
                // Show streamed text in real-time (no typewriter delay)
                bodyEl.textContent = fullSoFar;
                chatBox.scrollTop = chatBox.scrollHeight;
            },
            onDone(fullReply) {
                // Final render with markdown
                if (bodyEl) {
                    bodyEl.innerHTML = renderMarkdown(fullReply);
                    bodyEl.querySelectorAll('pre code').forEach(block => {
                        if (typeof hljs !== 'undefined') hljs.highlightElement(block);
                    });
                }
            },
            onError(err) {
                typingDiv.remove();
                appendErrorBubble(chatBox, err.message);
            },
        });
        if (replyText) {
            saveToHistory(message, replyText);
        }
    } catch (err) {
        if (typingDiv.parentNode) typingDiv.remove();
        appendErrorBubble(chatBox, err.message);
    } finally {
        setSendState(false);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// ─── Full (non-streaming) fetch ───────────────────────────────────────────────
async function sendFull(message, typingDiv, chatBox) {
    abortController = new AbortController();
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': novaSessionId },
        body: JSON.stringify({ message, session_id: novaSessionId }),
        signal: abortController.signal,
    });

    let replyText = "I'm having a moment — please try again.";
    if (response.ok) {
        const data = await response.json();
        replyText = data.reply || replyText;
        updateModelBadge(data.model);
    } else {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Error ${response.status}`);
    }

    typingDiv.remove();
    const bodyEl = appendNovaBubble(chatBox);
    await new Promise(resolve => {
        typewriterEffect(bodyEl, replyText, () => {
            chatBox.scrollTop = chatBox.scrollHeight;
            if (inputMode === 'voice') speak(replyText);
            resolve();
        });
    });
    return replyText;
}

// ─── Streaming (SSE) fetch ────────────────────────────────────────────────────
async function sendStream(message, typingDiv, chatBox) {
    abortController = new AbortController();
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': novaSessionId },
        body: JSON.stringify({ message, session_id: novaSessionId }),
        signal: abortController.signal,
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Error ${response.status}`);
    }

    typingDiv.remove();
    const bodyEl = appendNovaBubble(chatBox);
    bodyEl.classList.add('typing-cursor');

    let fullReply = '';
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const payload = JSON.parse(line.slice(6));
                    if (payload.error) throw new Error(payload.error);
                    if (payload.token) {
                        fullReply += payload.token;
                        // Render raw text as-is during streaming (fast)
                        bodyEl.textContent = fullReply;
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                    if (payload.done) {
                        // Final render with full markdown
                        bodyEl.innerHTML = renderMarkdown(fullReply);
                        bodyEl.querySelectorAll('pre code').forEach(block => {
                            if (typeof hljs !== 'undefined') hljs.highlightElement(block);
                        });
                        // Update model badge if the server told us which sub-provider answered
                        if (payload.model) updateModelBadge(payload.model);
                    }
                } catch (parseErr) {
                    if (parseErr.message !== 'Unexpected token') throw parseErr;
                }
            }
        }
    } finally {
        bodyEl.classList.remove('typing-cursor');
    }

    return fullReply;
}

// ─── Export Chat ──────────────────────────────────────────────────────────────
function exportChat() {
    const session = currentSessionId
        ? chatSessions.find(s => s.id === currentSessionId)
        : null;

    const messages = session ? session.messages : [];
    if (messages.length === 0) {
        showToast('Nothing to export — no messages yet', 'warning');
        return;
    }

    const lines = [`# NOVA Conversation Export`, `> ${session.timestamp || new Date().toLocaleString()}`, ''];
    messages.forEach(msg => {
        const role = msg.role === 'user' ? '**You**' : '**Nova**';
        lines.push(`### ${role}`);
        lines.push(msg.text);
        lines.push('');
    });

    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nova-chat-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Chat exported as Markdown ✓', 'success');
}

// ─── TTS (Text-to-Speech) ─────────────────────────────────────────────────────
let selectedVoice = null;
let isSpeaking = false;
let currentVoicePreset = localStorage.getItem('nova_voice_preset') || 'nova';

// ── Voice personality presets ────────────────────────────────────────────────
const VOICE_PRESETS = {
    nova: {
        name: 'NOVA',
        emoji: '🤖',
        // Natural female voice — let the best-scored voice speak naturally
        rate: 1.0,
        pitch: 1.0,
        volume: 1.0,
        preview: "Hello! I'm Nova, your personal AI assistant. How can I help you today?"
    },
    anya: {
        name: 'Anya',
        emoji: '🎀',
        // Anya — high-pitched, slower, playful curiosity, child-like energy
        rate: 0.82,
        pitch: 1.45,
        volume: 1.0,
        // Extra pitch boost for Zira-type voices
        pitchBoostRobotic: 0.2,
        preview: "Waku waku! Anya is very excited to help you! Heh heh heh!"
    },
    mitsuri: {
        name: 'Mitsuri',
        emoji: '🌸',
        // Mitsuri — warm, slightly breathy, gentle, bubbly and soft
        rate: 0.91,
        pitch: 1.22,
        volume: 1.0,
        preview: "Oh my goodness, of course! I'll do my very best to help you, I promise!"
    }
};

// ── Voice scoring — Indian English preferred, then US/UK ─────────────────────
function scoreVoice(v) {
    const n = v.name;
    const nl = n.toLowerCase();
    // ── Indian English (en-IN) — highest priority ──────────────────────────────
    if (/microsoft.*neerja.*online|microsoft.*swara.*online/i.test(n)) return 105;
    if (/microsoft.*neerja|microsoft.*swara|microsoft.*heera.*online/i.test(n)) return 102;
    if (/google.*india.*female|google.*hindi.*female/i.test(n)) return 100;
    if (/heera/i.test(n)) return 98;
    if (v.lang === 'en-IN' && (nl.includes('female') || nl.includes('woman'))) return 95;
    if (v.lang === 'en-IN') return 88;
    // ── High-quality US/UK fallbacks ──────────────────────────────────────────
    if (/google.*natural|neural|premium|enhanced/i.test(n)) return 80;
    if (/microsoft.*ava.*online|microsoft.*jenny.*online|microsoft.*aria.*online/i.test(n)) return 78;
    if (/microsoft.*ava|microsoft.*jenny|microsoft.*aria|microsoft.*natasha/i.test(n)) return 75;
    if (/samantha/i.test(n)) return 70;
    if (/google uk english female/i.test(n)) return 68;
    if (/google us english/i.test(n) && !nl.includes('male')) return 62;
    if (/google/i.test(n) && nl.includes('female') && v.lang.startsWith('en')) return 58;
    if (v.lang.startsWith('en') && (nl.includes('female') || nl.includes('woman') || nl.includes('girl'))) return 50;
    if (/zira/i.test(n)) return 30;
    if (v.lang.startsWith('en')) return 20;
    return 0;
}

function pickBestFemaleVoice() {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;
    // Score all English voices together — en-IN voices score highest
    const english = voices.filter(v => v.lang.startsWith('en'));
    if (!english.length) return voices[0];
    return english.reduce((best, v) => scoreVoice(v) > scoreVoice(best) ? v : best, english[0]);
}

function loadVoices() {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return;
    selectedVoice = pickBestFemaleVoice();
    console.log('[NOVA TTS] Voice:', selectedVoice?.name, '| Preset:', currentVoicePreset);
}

window.speechSynthesis.onvoiceschanged = loadVoices;
loadVoices();

// ── Apply voice preset to an utterance ───────────────────────────────────────
function applyVoicePreset(utterance, presetKey) {
    const preset = VOICE_PRESETS[presetKey] || VOICE_PRESETS.nova;
    const voiceName = (selectedVoice?.name || '').toLowerCase();
    const isNeural = /neural|online|natural|premium|enhanced/i.test(selectedVoice?.name || '');
    const isRobotic = /zira|david|mark/i.test(selectedVoice?.name || '');

    let rate = preset.rate;
    let pitch = preset.pitch;

    if (presetKey === 'nova') {
        // NOVA: tune based on voice quality as before
        if (isNeural) { rate = 1.0; pitch = 1.0; }
        else if (/samantha|google uk english female/i.test(selectedVoice?.name || '')) { rate = 0.97; pitch = 1.08; }
        else if (/google/i.test(selectedVoice?.name || '')) { rate = 1.0; pitch = 1.05; }
        else { rate = 0.92; pitch = 1.15; }
    } else if (isRobotic && preset.pitchBoostRobotic) {
        // Extra boost for robotic system voices to sound more character-like
        pitch = Math.min(2.0, pitch + preset.pitchBoostRobotic);
    }

    utterance.rate = rate;
    utterance.pitch = pitch;
    utterance.volume = novaVolume;
}

// ── Neural TTS via edge-tts backend ──────────────────────────────────────────
// Map voice presets → Edge TTS voice names
const EDGE_VOICE_MAP = {
    nova: 'en-IN-NeerjaExpressiveNeural', // Natural Indian English female
    anya: 'en-US-AnaNeural',              // Child-like voice (closest to Anya)
    mitsuri: 'en-IN-NeerjaNeural',           // Softer Indian English female
};

// Active HTML5 Audio element (so we can stop/volume-control it)
let currentAudio = null;

// ── Main speak function — uses gapless sentence queue for smooth playback ──────
// For short text (< 200 chars) a single fetch is fine; for longer replies
// we split into sentences and pre-load them with the queue so playback starts
// immediately and subsequent sentences are already decoded before they're needed.
function speak(text) {
    if (!text) return;
    stopSpeaking();

    const plainText = stripForSpeech(text);
    if (!plainText) return;

    // Reset queue state for fresh playback
    _ttsQueueDone = false;
    _ttsBatchCount = 0;
    _pendingSentences = [];
    _nextScheduledTime = 0;
    isSpeaking = true;
    setNovaSpeakingState(true);

    // Split text into sentences and feed into the gapless queue
    const { sentences: initialSentences, remainder } = _extractSentences(plainText);
    const allSentences = [...initialSentences];
    if (remainder.trim().length > 2) allSentences.push(remainder.trim());

    if (allSentences.length === 0) {
        // Single short phrase — enqueue as-is
        _enqueueSentenceTTS(plainText);
    } else {
        for (const sentence of allSentences) {
            _enqueueSentenceTTS(sentence);
        }
    }

    // Mark stream as done (all sentences added)
    _ttsQueueDone = true;
}


function stopSpeaking() {
    // Stop the Web Audio API source node (AudioBufferSourceNode uses .stop())
    if (_currentSource) {
        try { _currentSource.stop(); } catch (e) { /* already stopped */ }
        _currentSource = null;
    }
    // Drain and discard all queued items
    _ttsQueue.length = 0;
    _pendingSentences = [];
    _ttsQueuePlaying = false;
    _ttsQueueDone = false;
    _ttsBatchCount = 0;
    _nextPreloaded = null;
    _nextScheduledTime = 0;   // reset scheduling clock
    currentAudio = null;
    // Stop browser Web Speech API fallback
    window.speechSynthesis.cancel();
    isSpeaking = false;
    setNovaSpeakingState(false);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── GAPLESS TTS QUEUE — Web Audio API + Look-Ahead Pre-fetching ─────────
// ═══════════════════════════════════════════════════════════════════════════
// Strategy:
//   • Split AI reply into sentence groups as tokens stream in
//   • Chunk 1  → sent alone immediately  (fastest possible first-word playback)
//   • Chunk 2  → sent alone immediately  (so user hears something fast)
//   • Chunk 3+ → grouped in pairs        (fewer HTTP round-trips, smoother joins)
//   • While chunk N plays, chunk N+1 is already being fetched (look-ahead)
//   • Web Audio API schedules every decoded buffer sample-accurately
//     so there is zero gap between consecutive chunks
// ═══════════════════════════════════════════════════════════════════════════

let _audioCtx = null;          // lazily created AudioContext
const _ttsQueue = [];          // { buffer, ready, fetching }
let _ttsQueuePlaying = false;
let _ttsQueueDone = false;
let _currentSource = null;
let _nextScheduledTime = 0;
let _pendingSentences = [];
let _ttsBatchCount = 0;
const _TTS_BATCH_SIZE = 2;     // group subsequent sentences in pairs for smoother joins
let _gainNode = null;
let _nextPreloaded = null;     // look-ahead: the next item already queued for fetch

// ── Lazily create / resume the AudioContext ───────────────────────────────
function _getAudioCtx() {
    if (!_audioCtx) {
        _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        _gainNode = _audioCtx.createGain();
        _gainNode.gain.value = novaVolume;
        _gainNode.connect(_audioCtx.destination);
    }
    if (_audioCtx.state === 'suspended') _audioCtx.resume();
    return _audioCtx;
}

// ── Split streaming buffer into completed sentences ───────────────────────
function _extractSentences(buffer) {
    const sentences = [];
    // Match: sentence ending in . ! ? or a line break
    const regex = /[^.!?\n]*[.!?]+(?:\s|$)|[^\n]+\n/g;
    let match, lastIndex = 0;
    while ((match = regex.exec(buffer)) !== null) {
        const sentence = match[0].trim();
        if (sentence.length > 3) sentences.push(sentence);
        lastIndex = regex.lastIndex;
    }
    return { sentences, remainder: buffer.slice(lastIndex) };
}

// ── Fetch one TTS chunk and decode into AudioBuffer ───────────────────────
async function _fetchTTSBuffer(item, text) {
    const plainText = stripForSpeech(text);
    if (!plainText || plainText.length < 3) { item.ready = true; return; }
    const voiceName = EDGE_VOICE_MAP[currentVoicePreset] || EDGE_VOICE_MAP.nova;
    try {
        const resp = await fetch('/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: plainText, voice: voiceName }),
        });
        if (!resp.ok) { item.ready = true; return; }  // mark ready to unblock queue
        const arrayBuf = await resp.arrayBuffer();
        if (!arrayBuf.byteLength) { item.ready = true; return; }
        const ctx = _getAudioCtx();
        item.buffer = await ctx.decodeAudioData(arrayBuf);
        item.ready = true;
    } catch (e) {
        item.ready = true;  // mark ready so we skip gracefully
        console.warn('[TTS Buffer]', e);
    }
}

// ── Pre-fetch the next item in the queue (look-ahead) ────────────────────
function _prefetchNext() {
    if (_ttsQueue.length > 0) {
        // The item at the front of the queue may still be fetching — that's fine,
        // we just want the fetch to be in-flight while we play the current chunk.
        const next = _ttsQueue[0];
        if (!next.fetching) {
            next.fetching = _fetchTTSBuffer(next, next._text || '');
        }
    }
}

// ── Play next item in queue with sample-accurate scheduling ───────────────
function _playNextInQueue() {
    if (_ttsQueue.length === 0) {
        _ttsQueuePlaying = false;
        _currentSource = null;
        if (_ttsQueueDone) {
            // Small grace period (50ms) in case a final chunk is still fetching
            setTimeout(() => {
                if (_ttsQueue.length === 0 && !_ttsQueuePlaying) {
                    isSpeaking = false;
                    setNovaSpeakingState(false);
                }
            }, 50);
        }
        return;
    }

    _ttsQueuePlaying = true;
    const item = _ttsQueue.shift();

    const proceed = () => {
        if (!item.buffer) {
            // No audio (empty/error) — skip to next
            _playNextInQueue();
            return;
        }
        _scheduleBuffer(item.buffer);
        // ── Look-ahead: kick off fetch for the item now at front of queue ──
        _prefetchNext();
    };

    if (item.ready) {
        proceed();
    } else {
        item.fetching.then(proceed);
    }
}

function _scheduleBuffer(audioBuffer) {
    const ctx = _getAudioCtx();
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;

    if (_gainNode) _gainNode.gain.value = novaVolume;
    source.connect(_gainNode || ctx.destination);

    // If _nextScheduledTime is in the past (gap happened), reset to now
    const now = ctx.currentTime;
    // Allow a small tolerance: if next scheduled time is < 30ms ahead, just play now
    const startAt = (_nextScheduledTime > now + 0.03) ? _nextScheduledTime : now;
    source.start(startAt);

    // Next chunk starts sample-accurately at the end of this one
    _nextScheduledTime = startAt + audioBuffer.duration;

    _currentSource = source;
    currentAudio = source;

    source.onended = () => {
        if (_currentSource === source) {
            _currentSource = null;
            currentAudio = null;
            _playNextInQueue();
        }
    };
}

// ── Flush remaining pending sentences into the queue ──────────────────────
function _flushPending() {
    if (_pendingSentences.length === 0) return;
    const combined = _pendingSentences.join(' ');
    _pendingSentences = [];
    _enqueueTTSChunk(combined);
}

// ── Add a decoded chunk to the queue; start playing if idle ──────────────
function _enqueueTTSChunk(text) {
    const plainText = stripForSpeech(text);
    if (!plainText || plainText.length < 3) return;

    const item = { buffer: null, ready: false, fetching: null, _text: plainText };

    // If queue is currently empty and we're playing, kick off fetch immediately
    // so it's ready by the time the current chunk finishes
    const shouldFetchNow = (!_ttsQueuePlaying) || (_ttsQueue.length === 0);
    if (shouldFetchNow) {
        item.fetching = _fetchTTSBuffer(item, plainText);
    }
    // Otherwise, let _prefetchNext() handle it when the current chunk starts playing
    _ttsQueue.push(item);

    if (!_ttsQueuePlaying) {
        isSpeaking = true;
        setNovaSpeakingState(true);
        _playNextInQueue();
    } else if (!shouldFetchNow && _ttsQueue.length === 1) {
        // We just added the first queued item while something is playing —
        // kick off its fetch right now (look-ahead)
        item.fetching = _fetchTTSBuffer(item, plainText);
    }
}

// ── Sentence router: smart grouping for smooth onset + fluid delivery ────
//   Chunk 1  → alone  (instant first-word start)
//   Chunk 2  → alone  (covers the gap while chunk 1 plays)
//   Chunk 3+ → batched in pairs (smooth, fewer round-trips)
function _enqueueSentenceTTS(sentence) {
    _ttsBatchCount++;
    if (_ttsBatchCount <= 2) {
        _enqueueTTSChunk(sentence);
        return;
    }
    _pendingSentences.push(sentence);
    if (_pendingSentences.length >= _TTS_BATCH_SIZE) {
        _flushPending();
    }
}

// ─── Streaming voice pipeline: stream LLM + sentence-by-sentence TTS ─────
async function sendVoiceStream(message, opts = {}) {
    // opts: { onToken, onSentence, onDone, onError, skipTTS }
    const abortCtrl = new AbortController();
    let fullReply = '';
    let sentenceBuffer = '';

    // Reset TTS queue state
    _ttsQueue.length = 0;
    _ttsQueuePlaying = false;
    _ttsQueueDone = false;
    _ttsBatchCount = 0;
    _pendingSentences = [];

    try {
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Session-Id': novaSessionId },
            body: JSON.stringify({ message, session_id: novaSessionId }),
            signal: abortCtrl.signal,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Error ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let sseBuffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            sseBuffer += decoder.decode(value, { stream: true });
            const lines = sseBuffer.split('\n');
            sseBuffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const payload = JSON.parse(line.slice(6));
                    if (payload.error) throw new Error(payload.error);
                    if (payload.token) {
                        fullReply += payload.token;
                        sentenceBuffer += payload.token;
                        if (opts.onToken) opts.onToken(fullReply, payload.token);

                        // Extract completed sentences and enqueue TTS
                        const { sentences, remainder } = _extractSentences(sentenceBuffer);
                        if (sentences.length > 0) {
                            sentenceBuffer = remainder;
                            for (const s of sentences) {
                                if (!opts.skipTTS) _enqueueSentenceTTS(s);
                                if (opts.onSentence) opts.onSentence(s);
                            }
                        }
                    }
                    if (payload.done) {
                        // Flush any remaining text as final sentence
                        const leftover = sentenceBuffer.trim();
                        if (leftover.length > 2) {
                            if (!opts.skipTTS) _enqueueSentenceTTS(leftover);
                            if (opts.onSentence) opts.onSentence(leftover);
                        }
                        sentenceBuffer = '';
                        // Flush any batched sentences that haven't been sent yet
                        if (!opts.skipTTS) _flushPending();
                        _ttsQueueDone = true;
                        // If queue already empty (very short response), mark done
                        if (_ttsQueue.length === 0 && !_ttsQueuePlaying) {
                            isSpeaking = false;
                            setNovaSpeakingState(false);
                        }
                    }
                } catch (parseErr) {
                    if (parseErr.message && !parseErr.message.includes('Unexpected token')) throw parseErr;
                }
            }
        }
    } catch (err) {
        if (opts.onError) opts.onError(err);
        else console.error('[NOVA Voice Stream]', err);
        return null;
    }

    if (opts.onDone) opts.onDone(fullReply);
    return fullReply;
}

// ── Fast speak — instant browser TTS for voice conversations (no network delay)
// Picks the smoothest available voice and uses natural pacing
function speakFast(text) {
    if (!text) return;
    stopSpeaking();

    const plainText = stripForSpeech(text);

    // Pick the smoothest voice: prefer Online/Neural variants
    const voices = window.speechSynthesis.getVoices();
    let smoothVoice = selectedVoice;
    if (voices.length) {
        // Microsoft Online voices (e.g. "Neerja Online", "Jenny Online") are dramatically smoother
        const online = voices.filter(v => /online|neural/i.test(v.name) && v.lang.startsWith('en'));
        if (online.length) {
            // Prefer Indian English Online voices, then any English Online
            const indianOnline = online.filter(v => v.lang === 'en-IN');
            smoothVoice = indianOnline.length ? indianOnline[0] : online[0];
        } else if (!smoothVoice) {
            smoothVoice = pickBestFemaleVoice();
        }
    }

    const utterance = new SpeechSynthesisUtterance(plainText);

    // Smoother, more natural pacing
    utterance.rate = 0.92;    // slightly slower for natural flow
    utterance.pitch = 1.05;   // gentle lift for warmth
    utterance.volume = novaVolume;
    utterance.lang = smoothVoice?.lang || 'en-IN';
    if (smoothVoice) utterance.voice = smoothVoice;

    isSpeaking = true;
    setNovaSpeakingState(true);
    utterance.onend = () => { isSpeaking = false; setNovaSpeakingState(false); };
    utterance.onerror = () => { isSpeaking = false; setNovaSpeakingState(false); };
    window.speechSynthesis.speak(utterance);
}

// ── Voice personality selector ────────────────────────────────────────────────
function selectVoicePreset(key) {
    if (!VOICE_PRESETS[key]) return;
    currentVoicePreset = key;
    localStorage.setItem('nova_voice_preset', key);

    // Update card highlight
    document.querySelectorAll('.voice-preset-card').forEach(c => c.classList.remove('active'));
    const card = document.getElementById('vp-' + key);
    if (card) card.classList.add('active');

    // Update active label
    const lbl = document.getElementById('vp-active-name');
    if (lbl) lbl.textContent = VOICE_PRESETS[key].emoji + ' ' + VOICE_PRESETS[key].name;

    showToast(`Voice set to ${VOICE_PRESETS[key].emoji} ${VOICE_PRESETS[key].name}`, 'success');
}

function previewVoicePreset() {
    const preset = VOICE_PRESETS[currentVoicePreset] || VOICE_PRESETS.nova;
    stopSpeaking();
    if (!selectedVoice) selectedVoice = pickBestFemaleVoice();
    const utt = new SpeechSynthesisUtterance(preset.preview);
    applyVoicePreset(utt, currentVoicePreset);
    utt.lang = selectedVoice?.lang || 'en-US';
    if (selectedVoice) utt.voice = selectedVoice;
    window.speechSynthesis.speak(utt);
}

function initVoicePresetUI() {
    const saved = localStorage.getItem('nova_voice_preset') || 'nova';
    // Highlight the right card
    document.querySelectorAll('.voice-preset-card').forEach(c => c.classList.remove('active'));
    const card = document.getElementById('vp-' + saved);
    if (card) card.classList.add('active');
    // Update active label
    const lbl = document.getElementById('vp-active-name');
    if (lbl && VOICE_PRESETS[saved]) {
        lbl.textContent = VOICE_PRESETS[saved].emoji + ' ' + VOICE_PRESETS[saved].name;
    }
}

function setNovaSpeakingState(active) {
    const waveform = document.getElementById('waveform');
    const statusEl = document.querySelector('.voice-status');
    const hintEl = document.querySelector('.voice-hint');
    const micBtn = document.querySelector('.mic-orb-btn');
    if (active) {
        if (waveform) waveform.classList.add('speaking');
        if (statusEl) statusEl.textContent = '🔊 Nova is speaking…';
        if (hintEl) hintEl.textContent = 'Tap mic to stop Nova';
        if (micBtn) micBtn.classList.add('voice-speaking');
    } else {
        if (waveform) waveform.classList.remove('speaking');
        if (micBtn) micBtn.classList.remove('voice-speaking');
        updateVoiceStatusText();
        // If voice view is active, auto-restart listening so mic stays hot
        if (typeof isVoiceViewActive === 'function' && isVoiceViewActive()) {
            setVoiceViewState('idle');
            // Auto-restart recognition after a brief pause so user can speak again instantly
            setTimeout(() => {
                if (typeof isVoiceViewActive === 'function' && isVoiceViewActive() && recognition && !isListening && !isSpeaking) {
                    voiceBuffer = '';
                    try { recognition.start(); } catch (e) { /* already started */ }
                }
            }, 600);
        }
    }
}

function updateVoiceStatusText() {
    const statusEl = document.querySelector('.voice-status');
    const hintEl = document.querySelector('.voice-hint');
    if (isListening) {
        if (statusEl) statusEl.textContent = '🔴 Listening…';
        if (hintEl) hintEl.textContent = 'Speak now — Nova is hearing you';
    } else {
        if (statusEl) statusEl.textContent = 'Tap mic to talk to Nova';
        if (hintEl) hintEl.textContent = 'Or type below to search silently';
    }
}

// ─── System Stats ─────────────────────────────────────────────────────────────
async function updateSystemStats() {
    try {
        const resp = await fetch('/system');
        if (!resp.ok) return;
        const data = await resp.json();

        const cpu = Math.round(data.cpu);
        const memPct = Math.round(data.memory);
        const memGB = data.memory_used_gb;
        const totalGB = data.memory_total_gb;

        const cpuText = document.getElementById('cpu-value');
        const memText = document.getElementById('memory-value');
        const cpuBar = document.getElementById('cpu-bar');
        const memBar = document.getElementById('memory-bar');
        const cpuPct = document.getElementById('cpu-pct');
        const memPctEl = document.getElementById('mem-pct');

        if (cpuText) cpuText.textContent = cpu + '%';
        if (memText) memText.textContent = (memGB && totalGB) ? `${memGB}GB/${totalGB}GB` : memPct + '%';
        if (cpuBar) cpuBar.style.width = cpu + '%';
        if (memBar) memBar.style.width = memPct + '%';
        if (cpuPct) cpuPct.textContent = cpu + '%';
        if (memPctEl) memPctEl.textContent = memPct + '%';

        // Color thresholds
        if (cpuBar) cpuBar.style.background = cpu > 85 ? 'linear-gradient(90deg,#ff4b5c,#ff6b7a)' : '';
        if (memBar) memBar.style.background = memPct > 85 ? 'linear-gradient(90deg,#ff4b5c,#ff6b7a)' : '';
    } catch { }
}

function updateNetworkStatus() {
    const networkVal = document.getElementById('network-status');
    const netDot = document.getElementById('net-dot');
    if (navigator.onLine) {
        if (networkVal) { networkVal.textContent = 'Active'; networkVal.style.color = 'var(--core-cyan)'; }
        if (netDot) { netDot.style.color = 'var(--core-cyan)'; netDot.style.animation = 'netBlink 2s infinite'; }
    } else {
        if (networkVal) { networkVal.textContent = 'Offline'; networkVal.style.color = '#ff4b5c'; }
        if (netDot) { netDot.style.color = '#ff4b5c'; netDot.style.animation = 'none'; }
    }
}
window.addEventListener('online', updateNetworkStatus);
window.addEventListener('offline', updateNetworkStatus);

// ─── Model Badge ──────────────────────────────────────────────────────────────
function updateModelBadge(model) {
    const badge = document.getElementById('model-badge');
    if (badge && model) badge.textContent = '● ' + model;
    // Sync identity card
    const idModel = document.getElementById('id-model-name');
    if (idModel && model) idModel.textContent = model.charAt(0).toUpperCase() + model.slice(1);
    // Update session count
    const idSessions = document.getElementById('id-sessions');
    if (idSessions) idSessions.textContent = chatSessions.length + ' session' + (chatSessions.length !== 1 ? 's' : '');
}

async function loadAvailableModels() {
    try {
        const r = await fetch('/models');
        if (!r.ok) return;
        const data = await r.json();
        const select = document.getElementById('settings-model');
        if (select && data.models && data.models.length) {
            select.innerHTML = data.models.map(m =>
                `<option value="${m}">${m}</option>`
            ).join('');
        }
    } catch { }
}

// ─── Settings Panel ───────────────────────────────────────────────────────────
const DEFAULT_SETTINGS = {
    model: 'llama-3.3-70b-versatile',
    temperature: 0.75,
    num_predict: 1024,
    system_prompt: '',
    stream_mode: true,
    provider: 'hybrid',
};

function openSettings() {
    loadSettingsIntoUI();
    // Fetch backend live_mode to show/hide Ollama Local option
    fetch('/settings').then(r => r.json()).then(data => {
        novaLiveMode = data.live_mode === true;
        const ollamaLocalBtn = document.getElementById('provider-ollama');
        if (ollamaLocalBtn) {
            ollamaLocalBtn.style.display = novaLiveMode ? 'none' : '';
        }
        // If currently set to local ollama in live mode, switch to hybrid
        if (novaLiveMode && currentProvider === 'ollama') {
            setProvider('hybrid');
        }
    }).catch(() => {});
    document.getElementById('settings-overlay').classList.add('open');
}

function closeSettings() {
    document.getElementById('settings-overlay').classList.remove('open');
}

function closeSettingsOnOverlay(e) {
    if (e.target === document.getElementById('settings-overlay')) closeSettings();
}

function setStreamMode(val) {
    isStreamMode = val;
    document.getElementById('mode-stream').classList.toggle('active', val);
    document.getElementById('mode-full').classList.toggle('active', !val);
}

let currentProvider = 'hybrid';
let novaLiveMode = false; // set from backend /settings live_mode flag

function setProvider(p) {
    // In live mode, never allow Ollama local
    if (novaLiveMode && p === 'ollama') p = 'hybrid';
    currentProvider = p;
    // Guard: provider-ollama is hidden by default in live mode
    const ollamaBtn = document.getElementById('provider-ollama');
    if (ollamaBtn) {
        // Only show Ollama Local button if NOT in live mode
        if (!novaLiveMode) ollamaBtn.style.display = '';
        ollamaBtn.classList.toggle('active', p === 'ollama');
    }
    document.getElementById('provider-ollama-cloud').classList.toggle('active', p === 'ollama_cloud');
    document.getElementById('provider-groq').classList.toggle('active', p === 'groq');
    document.getElementById('provider-hybrid').classList.toggle('active', p === 'hybrid');
    document.getElementById('groq-settings').style.display = p === 'groq' ? 'block' : 'none';
    document.getElementById('ollama-cloud-settings').style.display = p === 'ollama_cloud' ? 'block' : 'none';
    document.getElementById('hybrid-settings').style.display = p === 'hybrid' ? 'block' : 'none';
    // Ollama model dropdown only for local/cloud (hybrid has its own per-column selector)
    document.getElementById('ollama-settings').style.display = (p === 'ollama' || p === 'ollama_cloud') ? 'block' : 'none';
}

function loadSettingsIntoUI() {
    const saved = JSON.parse(localStorage.getItem('nova_settings') || '{}');
    const s = { ...DEFAULT_SETTINGS, ...saved };

    document.getElementById('settings-model').value = s.model;
    const tempEl = document.getElementById('settings-temperature');
    tempEl.value = s.temperature;
    document.getElementById('temp-display').textContent = parseFloat(s.temperature).toFixed(2);
    const tokEl = document.getElementById('settings-tokens');
    tokEl.value = s.num_predict;
    document.getElementById('tokens-display').textContent = s.num_predict;
    document.getElementById('settings-system-prompt').value = s.system_prompt;
    setStreamMode(s.stream_mode !== false);

    // Provider fields — upgrade 'ollama' to 'hybrid' if live mode is active
    const savedProvider = (novaLiveMode && s.provider === 'ollama') ? 'hybrid' : (s.provider || 'hybrid');
    setProvider(savedProvider);
    // Groq
    const groqKeyEl = document.getElementById('settings-groq-key');
    if (groqKeyEl && s.groq_api_key) groqKeyEl.value = s.groq_api_key;
    const groqModelEl = document.getElementById('settings-groq-model');
    if (groqModelEl && s.groq_model) groqModelEl.value = s.groq_model;
    // Ollama Cloud
    const ollamaKeyEl = document.getElementById('settings-ollama-key');
    if (ollamaKeyEl && s.ollama_api_key) ollamaKeyEl.value = s.ollama_api_key;
    const ollamaUrlEl = document.getElementById('settings-ollama-url');
    if (ollamaUrlEl) ollamaUrlEl.value = s.ollama_cloud_url || 'https://api.ollama.com/api/generate';
    // Hybrid fields — if the specific hybrid fields are empty, fall back to
    // the canonical shared keys already saved (so settings round-trip cleanly)
    const savedOllamaKey = s.ollama_api_key || '';
    const savedGroqKey   = s.groq_api_key   || '';
    const hOllamaKey = document.getElementById('settings-hybrid-ollama-key');
    if (hOllamaKey) hOllamaKey.value = s.ollama_api_key || '';
    const hOllamaUrl = document.getElementById('settings-hybrid-ollama-url');
    if (hOllamaUrl) hOllamaUrl.value = s.ollama_cloud_url || 'https://api.ollama.com/api/generate';
    const hOllamaModel = document.getElementById('settings-hybrid-ollama-model');
    if (hOllamaModel && s.hybrid_ollama_model) hOllamaModel.value = s.hybrid_ollama_model;
    const hGroqKey = document.getElementById('settings-hybrid-groq-key');
    if (hGroqKey) hGroqKey.value = s.groq_api_key || '';
    const hGroqModel = document.getElementById('settings-hybrid-groq-model');
    if (hGroqModel && s.hybrid_groq_model) hGroqModel.value = s.hybrid_groq_model;
}

async function saveSettings() {
    const model = document.getElementById('settings-model').value;
    const temperature = parseFloat(document.getElementById('settings-temperature').value);
    const num_predict = parseInt(document.getElementById('settings-tokens').value);
    const system_prompt = document.getElementById('settings-system-prompt').value.trim();
    const provider = currentProvider;

    // Per-provider field reads
    const groq_api_key = document.getElementById('settings-groq-key')?.value || '';
    const groq_model = document.getElementById('settings-groq-model')?.value || 'llama-3.3-70b-versatile';
    const ollama_api_key = document.getElementById('settings-ollama-key')?.value || '';
    const ollama_cloud_url = document.getElementById('settings-ollama-url')?.value || 'https://api.ollama.com/api/generate';
    // Hybrid reads — use shared api keys so they sync across modes
    const h_ollama_key = document.getElementById('settings-hybrid-ollama-key')?.value || '';
    const h_ollama_url = document.getElementById('settings-hybrid-ollama-url')?.value || 'https://api.ollama.com/api/generate';
    const hybrid_ollama_model = document.getElementById('settings-hybrid-ollama-model')?.value || 'mistral';
    const h_groq_key = document.getElementById('settings-hybrid-groq-key')?.value || '';
    const hybrid_groq_model = document.getElementById('settings-hybrid-groq-model')?.value || 'llama-3.3-70b-versatile';

    // Resolve canonical keys — prefer the active-provider's specific field,
    // then fall back to the other field, so whichever one the user filled wins.
    const final_groq_key   = (provider === 'hybrid'
        ? (h_groq_key   || groq_api_key)
        : (groq_api_key || h_groq_key))    || '';
    const final_ollama_key = (provider === 'hybrid'
        ? (h_ollama_key   || ollama_api_key)
        : (ollama_api_key || h_ollama_key)) || '';
    const final_ollama_url = (provider === 'hybrid'
        ? (h_ollama_url   || ollama_cloud_url)
        : (ollama_cloud_url || h_ollama_url)) || 'https://api.ollama.com/api/generate';

    const settings = {
        model, temperature, num_predict, system_prompt, stream_mode: isStreamMode,
        provider, groq_api_key: final_groq_key, groq_model,
        ollama_api_key: final_ollama_key, ollama_cloud_url: final_ollama_url,
        hybrid_ollama_model, hybrid_groq_model
    };
    localStorage.setItem('nova_settings', JSON.stringify(settings));

    // Sync with backend
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model, temperature, num_predict, system_prompt, provider,
                groq_api_key: final_groq_key, groq_model,
                ollama_api_key: final_ollama_key, ollama_cloud_url: final_ollama_url,
                hybrid_ollama_model, hybrid_groq_model,
            }),
        });
    } catch { }

    const activeModel = provider === 'groq' ? groq_model
        : provider === 'hybrid' ? `${hybrid_ollama_model} / ${hybrid_groq_model}`
        : model;
    updateModelBadge(activeModel);
    closeSettings();
    const providerLabel = provider === 'groq' ? '☁️ Groq'
        : provider === 'hybrid' ? '🔀 Hybrid'
        : provider === 'ollama_cloud' ? '🌐 Ollama Cloud'
        : '🖥️ Ollama';
    showToast(`Settings saved! Provider: ${providerLabel} ✓`, 'success');
}

async function resetSettings() {
    localStorage.removeItem('nova_settings');
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: DEFAULT_SETTINGS.model,
                temperature: DEFAULT_SETTINGS.temperature,
                num_predict: DEFAULT_SETTINGS.num_predict,
                system_prompt: '',
                provider: 'groq',
            }),
        });
    } catch { }
    loadSettingsIntoUI();
    showToast('Settings reset to defaults', 'info');
}

async function applySavedSettings() {
    const saved = JSON.parse(localStorage.getItem('nova_settings') || '{}');

    // ── Fetch live_mode from backend to know if Ollama Local is allowed ─────────
    try {
        const r = await fetch('/settings');
        if (r.ok) {
            const backendSettings = await r.json();
            novaLiveMode = backendSettings.live_mode === true;
            const ollamaLocalBtn = document.getElementById('provider-ollama');
            if (ollamaLocalBtn) {
                // Show Ollama Local ONLY if backend explicitly says live_mode=false
                // Default (no backend / fetch fail) = hidden (cloud-safe)
                ollamaLocalBtn.style.display = novaLiveMode ? 'none' : '';
            }
            // If localStorage has ollama (local) provider but we're in live mode, upgrade to hybrid
            if (novaLiveMode && saved.provider === 'ollama') {
                saved.provider = 'hybrid';
                localStorage.setItem('nova_settings', JSON.stringify(saved));
                console.info('[NOVA] Live mode: upgraded local ollama provider to hybrid');
            }
        }
        // If fetch fails (static deploy / no backend), novaLiveMode stays false
        // but button stays hidden because HTML has display:none by default
    } catch { }

    if (!saved.model && !saved.temperature && !saved.provider) return;
    isStreamMode = saved.stream_mode !== false;
    currentProvider = saved.provider || 'hybrid';
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: saved.model || DEFAULT_SETTINGS.model,
                temperature: saved.temperature || DEFAULT_SETTINGS.temperature,
                num_predict: saved.num_predict || DEFAULT_SETTINGS.num_predict,
                system_prompt: saved.system_prompt || '',
                provider: saved.provider || 'groq',
                groq_api_key: saved.groq_api_key || '',
                groq_model: saved.groq_model || 'llama-3.3-70b-versatile',
                ollama_api_key: saved.ollama_api_key || '',
                ollama_cloud_url: saved.ollama_cloud_url || 'https://api.ollama.com/api/generate',
                hybrid_ollama_model: saved.hybrid_ollama_model || 'mistral',
                hybrid_groq_model: saved.hybrid_groq_model || 'llama-3.3-70b-versatile',
            }),
        });
        let activeModel;
        if (currentProvider === 'groq') activeModel = saved.groq_model || 'llama-3.3-70b-versatile';
        else if (currentProvider === 'hybrid') activeModel = `${saved.hybrid_ollama_model || 'mistral'} / ${saved.hybrid_groq_model || 'llama-3.3-70b-versatile'}`;
        else activeModel = saved.model || DEFAULT_SETTINGS.model;
        updateModelBadge(activeModel);
    } catch { }
}

// ─── Input Auto-Resize & Counter ──────────────────────────────────────────────
function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function updateCharCounter() {
    const input = document.getElementById('user-input');
    const counter = document.getElementById('char-counter');
    if (!input || !counter) return;
    const len = input.value.length;
    counter.textContent = len > 0 ? len : '';
    counter.style.color = len > 1800 ? 'var(--core-red)' : len > 1500 ? 'var(--core-gold)' : 'rgba(255,255,255,0.3)';
}

// ─── Voice Recognition ────────────────────────────────────────────────────────
let recognition;
let isListening = false;
let voiceBuffer = '';
let silenceTimer = null;          // auto-stop after silence
const SILENCE_TIMEOUT_MS = 1500; // 1.5s of silence before auto-submit

function setVoiceListeningState(active) {
    isListening = active;
    const allMicBtns = document.querySelectorAll('.mic-orb-btn, .voice-btn');
    const micRing = document.querySelector('.mic-orb-ring');
    if (active) {
        allMicBtns.forEach(btn => btn.classList.add('voice-active'));
        if (micRing) micRing.style.animationDuration = '0.6s';
    } else {
        allMicBtns.forEach(btn => btn.classList.remove('voice-active'));
        if (micRing) micRing.style.animationDuration = '2.5s';
    }
    updateVoiceStatusText();
}

function handleMicClick() {
    if (isSpeaking) { stopSpeaking(); return; }
    if (isListening) { if (recognition) recognition.stop(); return; }
    if (!recognition) { showToast('Speech recognition is not supported in this browser', 'warning'); return; }
    voiceBuffer = '';
    recognition.start();
}

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;       // keep mic open until silence detected
    recognition.interimResults = true;   // receive partial results so we can reset the silence timer
    recognition.lang = 'en-US'; // en-US is faster & more accurate for Web Speech API

    // ── Start: reset buffer + silence timer ──────────────────────────────────
    recognition.onstart = () => {
        setVoiceListeningState(true);
        voiceBuffer = '';
        clearTimeout(silenceTimer);
    };

    // ── Speech result: accumulate finals, reset silence countdown on any result
    recognition.onresult = (event) => {
        // Reset the silence timer every time we receive any speech (interim or final)
        clearTimeout(silenceTimer);

        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                voiceBuffer += event.results[i][0].transcript + ' ';
            }
        }

        // After SILENCE_TIMEOUT_MS with no new speech, auto-stop and process
        silenceTimer = setTimeout(() => {
            if (isListening && recognition) {
                // Update status to give user feedback
                const statusEl = document.querySelector('.voice-status');
                if (statusEl) statusEl.textContent = '⏳ Processing…';
                recognition.stop(); // triggers onend → sendVoiceMessage
            }
        }, SILENCE_TIMEOUT_MS);
    };

    // ── End: clear timer, send whatever was captured ─────────────────────────
    recognition.onend = () => {
        clearTimeout(silenceTimer);
        silenceTimer = null;
        setVoiceListeningState(false);
        const finalText = voiceBuffer.trim();
        if (finalText) {
            sendVoiceMessage(finalText);
        }
        voiceBuffer = '';
    };

    // ── Error handling ───────────────────────────────────────────────────────
    recognition.onerror = (event) => {
        clearTimeout(silenceTimer);
        silenceTimer = null;
        // 'no-speech' fires when the mic opens but user hasn't spoken yet — ignore
        if (event.error === 'no-speech') return;
        setVoiceListeningState(false);
        voiceBuffer = '';
        showToast(`Voice error: ${event.error}`, 'error');
    };
}

// ─── Volume Control ────────────────────────────────────────────────────────────
let novaVolume = 1.0;
let prevVolume = 1.0;
let volPopupOpen = false;
let isDraggingVol = false;

function getVolIcon(vol) {
    if (vol === 0) return '🔇';
    if (vol < 0.4) return '🔉';
    return '🔊';
}

function setVolume(vol) {
    novaVolume = Math.max(0, Math.min(1, parseFloat(vol.toFixed(2))));
    const icon = document.getElementById('vol-icon');
    const fill = document.getElementById('vol-fill');
    const thumb = document.getElementById('vol-thumb');
    const pct = novaVolume * 100;
    if (icon) icon.textContent = getVolIcon(novaVolume);
    if (fill) fill.style.height = pct + '%';
    if (thumb) thumb.style.bottom = 'calc(' + pct + '% - 8px)';
    // Apply to live neural TTS audio if playing
    if (currentAudio) currentAudio.volume = novaVolume;
}

function toggleVolPopup() {
    volPopupOpen = !volPopupOpen;
    const popup = document.getElementById('vol-popup');
    if (popup) popup.classList.toggle('open', volPopupOpen);
}

function toggleMute() {
    if (novaVolume > 0) { prevVolume = novaVolume; setVolume(0); }
    else { setVolume(prevVolume > 0 ? prevVolume : 0.8); }
}

// ─── Initialization ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async function () {
    createWaveform();
    loadVoices();
    updateNetworkStatus();
    updateVoiceStatusText();
    updateSystemStats();
    setInterval(updateSystemStats, 3000);
    renderHistoryList();
    await applySavedSettings();
    await loadAvailableModels();

    // Show welcome if no current session
    if (!currentSessionId || chatSessions.length === 0) {
        showWelcomeScreen();
    } else {
        hideWelcomeScreen();
    }

    // ── Input field (textarea) ──
    const input = document.getElementById('user-input');
    if (input) {
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        input.addEventListener('input', function () {
            autoResize(this);
            updateCharCounter();
        });
    }

    // ── Keyboard: Escape closes panels ──
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeSettings();
            const hp = document.getElementById('history-panel');
            if (hp && hp.classList.contains('open')) hp.classList.remove('open');
        }
    });

    // ── Mic buttons ──
    document.querySelectorAll('.mic-orb-btn, #main-mic-btn').forEach(btn => {
        btn.addEventListener('click', handleMicClick);
    });

    // ── New conversation ──
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) newChatBtn.addEventListener('click', startNewConversation);

    // ── Volume control ──
    const iconBtn = document.getElementById('vol-icon-btn');
    const track = document.getElementById('vol-track');
    const popup = document.getElementById('vol-popup');
    if (iconBtn) {
        let holdTimer = null;
        iconBtn.addEventListener('mousedown', () => {
            holdTimer = setTimeout(() => { holdTimer = null; toggleVolPopup(); }, 400);
        });
        iconBtn.addEventListener('mouseup', () => {
            if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; toggleMute(); }
        });
        iconBtn.addEventListener('mouseleave', () => {
            if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
        });
        iconBtn.addEventListener('touchstart', e => {
            e.preventDefault();
            holdTimer = setTimeout(() => { holdTimer = null; toggleVolPopup(); }, 400);
        }, { passive: false });
        iconBtn.addEventListener('touchend', () => {
            if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; toggleMute(); }
        });
    }
    if (popup) {
        popup.addEventListener('wheel', e => {
            e.preventDefault();
            setVolume(novaVolume + (e.deltaY > 0 ? -0.05 : 0.05));
        }, { passive: false });
    }
    function volFromY(clientY) {
        if (!track) return;
        const rect = track.getBoundingClientRect();
        setVolume(1 - ((clientY - rect.top) / rect.height));
    }
    if (track) {
        track.addEventListener('mousedown', e => { isDraggingVol = true; volFromY(e.clientY); });
        track.addEventListener('touchstart', e => { isDraggingVol = true; volFromY(e.touches[0].clientY); }, { passive: true });
    }
    document.addEventListener('mousemove', e => { if (isDraggingVol) volFromY(e.clientY); });
    document.addEventListener('mouseup', () => { isDraggingVol = false; });
    document.addEventListener('touchmove', e => { if (isDraggingVol) volFromY(e.touches[0].clientY); }, { passive: true });
    document.addEventListener('touchend', () => { isDraggingVol = false; });
    document.addEventListener('click', e => {
        const widget = document.getElementById('vol-widget');
        if (volPopupOpen && widget && !widget.contains(e.target)) {
            volPopupOpen = false;
            if (popup) popup.classList.remove('open');
        }
    });

    setVolume(1.0);
    updateCharCounter();

    // ── Load NOVA Brain stats on startup ──
    loadNovaMemory();
    // Refresh brain panel every 60 seconds (picks up background extractions)
    setInterval(loadNovaMemory, 60_000);

    // ── Restore saved voice personality preset ──
    initVoicePresetUI();
});


// ─── NOVA BRAIN — Memory UI ───────────────────────────────────────────────────

async function loadNovaMemory() {
    try {
        const res = await fetch('/memory');
        if (!res.ok) return;
        const data = await res.json();
        updateBrainPanel(data);
    } catch (e) {
        // Server not running yet — silently skip
    }
}

function updateBrainPanel(data) {
    animateBrainNum('brain-facts', data.facts_count || 0);
    animateBrainNum('brain-days', data.days_active || 0);
    animateBrainNum('brain-convos', data.total_conversations || 0);

    // Memory capacity bar (capped at 100 facts = 100%)
    const pct = Math.min(100, Math.round(((data.facts_count || 0) / 100) * 100));
    const fill = document.getElementById('brain-progress-fill');
    const pctEl = document.getElementById('brain-pct');
    if (fill) fill.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';

    // Interest tags
    const tagsEl = document.getElementById('brain-tags');
    if (tagsEl) {
        const interests = data.top_interests || [];
        tagsEl.innerHTML = interests.length === 0
            ? '<span class="brain-tag-empty">No interests yet</span>'
            : interests.map(t => `<span class="brain-tag">${escapeHtml(t)}</span>`).join('');
    }

    // Badge text based on how many facts learned
    const badge = document.getElementById('brain-badge');
    if (badge) {
        const f = data.facts_count || 0;
        badge.textContent = f === 0 ? 'Learning…'
            : f < 5 ? 'Novice'
                : f < 20 ? 'Growing'
                    : f < 50 ? 'Smart'
                        : 'Advanced';
    }
}

function animateBrainNum(id, target) {
    const el = document.getElementById(id);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;
    const diff = target - current;
    const steps = Math.min(Math.abs(diff), 20);
    const step = diff / steps;
    let val = current, i = 0;
    const timer = setInterval(() => {
        i++;
        val += step;
        el.textContent = Math.round(val);
        if (i >= steps) { el.textContent = target; clearInterval(timer); }
    }, 30);
}

async function resetNovaMemory() {
    if (!confirm("Reset all of Nova's learned memory? She will start fresh. This cannot be undone.")) return;
    try {
        const res = await fetch('/memory/reset', { method: 'POST' });
        if (res.ok) {
            updateBrainPanel({ facts_count: 0, days_active: 0, total_conversations: 0, top_interests: [] });
            showToast("🧠 Nova's memory has been reset", 'info');
        }
    } catch (err) {
        showToast('Could not reach AI engine. Is the Flask server running?', 'error');
    }
};

// ─── 3D Home Scene (Three.js 3D Interactive Logo) ────────────────────────────────
let homeScene, homeCamera, homeRenderer, homeParticles;
let mouseX = 0, mouseY = 0;
let homeAnimId;
let logoGeoData = [];
const windowHalfX = window.innerWidth / 2;
const windowHalfY = window.innerHeight / 2;
const logoParticleCount = 4000;

function initHomeParticles() {
    const canvas = document.getElementById('home-particles');
    if (!canvas) return;

    if (typeof THREE === 'undefined') {
        console.warn('Three.js not loaded. Skipping 3D home logo.');
        return;
    }

    if (homeScene) {
        onWindowResizeHome();
        animateHomeParticles();
        return;
    }

    homeScene = new THREE.Scene();
    // Use narrower FOV so the logo doesn't distort at edges
    homeCamera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 1000);
    homeCamera.position.z = 180;

    homeRenderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    homeRenderer.setPixelRatio(window.devicePixelRatio);
    homeRenderer.setSize(window.innerWidth, window.innerHeight);

    const particleGeometry = new THREE.BufferGeometry();
    const posArray = new Float32Array(logoParticleCount * 3);
    const colorsArray = new Float32Array(logoParticleCount * 3);

    const colorInside = new THREE.Color(0x00e5ff); // Cyan
    const colorOutside = new THREE.Color(0xbc60d3); // Purple
    const colorGold = new THREE.Color(0xe2c077); // Gold

    logoGeoData = [];

    // Procedurally generate a Tech/Neural Sphere Aura
    for(let i = 0; i < logoParticleCount * 3; i+=3) {
        // Sphere distribution
        const radius = 50 + Math.random() * 60; // Inner empty space of 50 for the logo
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        
        let px = radius * Math.sin(phi) * Math.cos(theta);
        let py = radius * Math.sin(phi) * Math.sin(theta);
        let pz = radius * Math.cos(phi);

        posArray[i] = px;
        posArray[i+1] = py;
        posArray[i+2] = pz;

        // Save original positions and initial velocity for physics
        logoGeoData.push({
            ox: px, oy: py, oz: pz,
            vx: 0, vy: 0, vz: 0,
            angle: Math.random() * Math.PI * 2,
            speed: 0.005 + Math.random() * 0.015
        });
        
        // Color based on radius (inner cyan, outer purple)
        const mixRatio = (radius - 50) / 60; 
        let mixedColor = colorInside.clone().lerp(colorOutside, mixRatio);
        
        // Sprinkle gold
        if (Math.random() > 0.95) mixedColor = colorGold;
        
        colorsArray[i] = mixedColor.r;
        colorsArray[i+1] = mixedColor.g;
        colorsArray[i+2] = mixedColor.b;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));

    const particleMaterial = new THREE.PointsMaterial({
        size: 1.8,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: true, // size differs based on distance
        depthWrite: false
    });

    homeParticles = new THREE.Points(particleGeometry, particleMaterial);
    
    // Position it higher up where the old 2D logo used to be.
    homeParticles.position.y = 35;
    
    homeScene.add(homeParticles);

    // Track mouse for physics repel
    document.addEventListener('mousemove', onDocumentMouseMoveHome);
    window.addEventListener('resize', onWindowResizeHome);

    animateHomeParticles();
}

function onDocumentMouseMoveHome(event) {
    mouseX = (event.clientX - windowHalfX);
    mouseY = (event.clientY - windowHalfY);
}

function onWindowResizeHome() {
    if(!homeCamera || !homeRenderer) return;
    homeCamera.aspect = window.innerWidth / window.innerHeight;
    homeCamera.updateProjectionMatrix();
    homeRenderer.setSize(window.innerWidth, window.innerHeight);
}

function animateHomeParticles() {
    const homeView = document.getElementById('view-home');
    if (!homeView || !homeView.classList.contains('view-active')) {
        cancelAnimationFrame(homeAnimId);
        return;
    }
    
    homeAnimId = requestAnimationFrame(animateHomeParticles);
    
    if (homeParticles) {
        // Slow majestic float & rotate
        const time = Date.now() * 0.001;
        homeParticles.rotation.y = Math.sin(time * 0.2) * 0.5;
        homeParticles.rotation.x = Math.sin(time * 0.3) * 0.2;
        
        // Map mouse screen coords to approximate 3D world coords at z=180
        const cursorX = mouseX * 0.12; 
        const cursorY = -mouseY * 0.12 - 35; 
        
        const positions = homeParticles.geometry.attributes.position.array;
        
        for(let i=0; i < logoParticleCount; i++) {
            let data = logoGeoData[i];
            
            // Read current
            let cx = positions[i*3];
            let cy = positions[i*3+1];
            let cz = positions[i*3+2];
            
            // Calculate distance to mouse cursor
            let dx = cx - cursorX;
            let dy = cy - cursorY;
            
            // Push away logic
            let distSq = dx*dx + dy*dy;
            let radiusSq = 1600; // Effect radius ~40 units
            
            if (distSq < radiusSq) {
                let force = (radiusSq - distSq) / radiusSq;
                data.vx += dx * force * 0.06;
                data.vy += dy * force * 0.06;
                data.vz += (Math.random() - 0.5) * force * 0.3; // Pop outwards
            }
            
            // Subtly orbit the particles around their origin to make the sphere feel "alive"
            data.angle += data.speed;
            const orbitX = data.ox + Math.sin(data.angle) * 5;
            const orbitY = data.oy + Math.cos(data.angle) * 5;
            const orbitZ = data.oz + Math.sin(data.angle * 0.5) * 5;

            // Spring back to orbital target
            data.vx += (orbitX - cx) * 0.04;
            data.vy += (orbitY - cy) * 0.04;
            data.vz += (orbitZ - cz) * 0.04;
            
            // Friction/damping
            data.vx *= 0.85;
            data.vy *= 0.85;
            data.vz *= 0.85;
            
            // Write new
            positions[i*3]   += data.vx;
            positions[i*3+1] += data.vy;
            positions[i*3+2] += data.vz;
        }
        
        homeParticles.geometry.attributes.position.needsUpdate = true;
    }
    
    homeRenderer.render(homeScene, homeCamera);
}

// ─── CENTER PANEL — Particle canvas ──────────────────────────────────────────
(function initParticles() {
    const canvas = document.getElementById('cp-particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const particles = Array.from({ length: 55 }, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.5 + 0.3,
        dx: (Math.random() - 0.5) * 0.35,
        dy: (Math.random() - 0.5) * 0.35,
        alpha: Math.random() * 0.5 + 0.1,
        hue: Math.random() > 0.5 ? '0,229,255' : '188,96,211'
    }));

    function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            p.x += p.dx; p.y += p.dy;
            if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
            if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${p.hue},${p.alpha})`;
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }
    draw();
})();

// ─── CENTER PANEL — Live data nodes ──────────────────────────────────────────
let _cpStartTime = Date.now();
let _cpSessions = 0;
let _cpLastLatency = null;

function updateOrbNodes() {
    // UPTIME
    const el = document.getElementById('node-uptime');
    if (el) {
        const s = Math.floor((Date.now() - _cpStartTime) / 1000);
        const m = Math.floor(s / 60), sec = s % 60;
        el.textContent = String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
    }
    // MODEL — pull from settings store if available
    const modelEl = document.getElementById('node-model');
    if (modelEl) {
        const ms = document.getElementById('id-model-name');
        if (ms && ms.textContent) modelEl.textContent = ms.textContent.toUpperCase();
    }
    // SESSIONS
    const sessEl = document.getElementById('node-sessions');
    if (sessEl) {
        const keys = Object.keys(localStorage).filter(k => k.startsWith('nova_session_') || k.startsWith('history_'));
        sessEl.textContent = Math.max(keys.length, _cpSessions);
    }
    // LATENCY
    if (_cpLastLatency !== null) {
        const latEl = document.getElementById('node-latency');
        if (latEl) latEl.textContent = _cpLastLatency + 'ms';
    }
}

// Hook into loadNovaMemory to update the MEMORY node
const _origLoadNovaMemory = loadNovaMemory;
window.loadNovaMemory = async function () {
    await _origLoadNovaMemory();
    const memEl = document.getElementById('node-mem');
    const factsEl = document.getElementById('brain-facts');
    if (memEl && factsEl) memEl.textContent = factsEl.textContent + ' FACTS';
};

// Uptime + node refresh loop
setInterval(updateOrbNodes, 1000);
updateOrbNodes();

// ═══════════════════════════════════════════════════════════════════════════
// ─── HOME PAGE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════


// ─── Home Page Particle Canvas ────────────────────────────────────────────────
let homeParticlesInitialized = false;
function initHomeParticles() {
    if (homeParticlesInitialized) return;
    homeParticlesInitialized = true;

    const canvas = document.getElementById('home-particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const NUM_DOTS = 80;
    const dots = Array.from({ length: NUM_DOTS }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.5 + 0.2
    }));

    const LINE_DIST = 120;

    function animate() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Move dots
        dots.forEach(d => {
            d.x += d.vx;
            d.y += d.vy;
            if (d.x < 0) d.x = canvas.width;
            if (d.x > canvas.width) d.x = 0;
            if (d.y < 0) d.y = canvas.height;
            if (d.y > canvas.height) d.y = 0;
        });

        // Draw connections
        for (let i = 0; i < dots.length; i++) {
            for (let j = i + 1; j < dots.length; j++) {
                const dx = dots[i].x - dots[j].x;
                const dy = dots[i].y - dots[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < LINE_DIST) {
                    const opacity = (1 - dist / LINE_DIST) * 0.25;
                    ctx.strokeStyle = `rgba(0, 180, 255, ${opacity})`;
                    ctx.lineWidth = 0.5;
                    ctx.beginPath();
                    ctx.moveTo(dots[i].x, dots[i].y);
                    ctx.lineTo(dots[j].x, dots[j].y);
                    ctx.stroke();
                }
            }
        }

        // Draw dots
        dots.forEach(d => {
            ctx.beginPath();
            ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 210, 255, ${d.alpha})`;
            ctx.fill();
        });

        requestAnimationFrame(animate);
    }
    animate();
}

// ─── Weather Widget ───────────────────────────────────────────────────────────
const WEATHER_CODES = {
    0: ['☀️', 'Clear Sky'],
    1: ['🌤️', 'Mainly Clear'], 2: ['⛅', 'Partly Cloudy'], 3: ['☁️', 'Overcast'],
    45: ['🌫️', 'Foggy'], 48: ['🌫️', 'Icy Fog'],
    51: ['🌦️', 'Light Drizzle'], 53: ['🌦️', 'Drizzle'], 55: ['🌧️', 'Heavy Drizzle'],
    61: ['🌧️', 'Light Rain'], 63: ['🌧️', 'Rain'], 65: ['🌧️', 'Heavy Rain'],
    71: ['🌨️', 'Light Snow'], 73: ['❄️', 'Snow'], 75: ['❄️', 'Heavy Snow'],
    80: ['🌦️', 'Showers'], 81: ['🌧️', 'Rain Showers'], 82: ['⛈️', 'Violent Showers'],
    95: ['⛈️', 'Thunderstorm'], 96: ['⛈️', 'Hail Storm'], 99: ['⛈️', 'Heavy Hail']
};

async function fetchWeather() {
    try {
        // Use IP geolocation to get approximate location
        let lat = 12.97, lon = 77.59; // Default: Bangalore
        try {
            const geoRes = await fetch('https://ipapi.co/json/');
            if (geoRes.ok) {
                const geoData = await geoRes.json();
                if (geoData.latitude) { lat = geoData.latitude; lon = geoData.longitude; }
            }
        } catch (e) { /* use default */ }

        const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,apparent_temperature,weather_code&temperature_unit=celsius`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('weather API failed');
        const data = await res.json();
        const cur = data.current;
        const wc = cur.weather_code;
        const [icon, desc] = WEATHER_CODES[wc] || ['🌡️', 'Unknown'];

        document.getElementById('weather-icon').textContent = icon;
        document.getElementById('weather-temp').textContent = `${Math.round(cur.temperature_2m)}°`;
        document.getElementById('weather-feel').textContent = `Feels ${Math.round(cur.apparent_temperature)}°C`;
        document.getElementById('weather-desc').textContent = desc;
    } catch (e) {
        document.getElementById('weather-desc').textContent = 'Weather unavailable';
    }
}

// ─── Mini Calendar ────────────────────────────────────────────────────────────
function buildMiniCal() {
    const cal = document.getElementById('mini-cal');
    if (!cal) return;
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth();
    const today = now.getDate();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const firstDay = new Date(year, month, 1).getDay(); // 0=Sun

    const dayNames = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
    let html = dayNames.map(d => `<div class="mc-day-header">${d}</div>`).join('');

    // Empty cells before day 1
    for (let i = 0; i < firstDay; i++) {
        html += `<div class="mc-day empty"></div>`;
    }
    for (let d = 1; d <= daysInMonth; d++) {
        html += `<div class="mc-day${d === today ? ' today' : ''}">${d}</div>`;
    }
    cal.innerHTML = html;

    // Update event count based on day of week
    const eventEl = document.getElementById('event-count');
    const dayOfWeek = now.getDay();
    const events = (dayOfWeek === 0 || dayOfWeek === 6) ? 0 : Math.floor(Math.random() * 4) + 1;
    if (eventEl) eventEl.textContent = `${events} EVENT${events !== 1 ? 'S' : ''}`;
}

// ─── Task List ────────────────────────────────────────────────────────────────
function updateTaskCount() {
    const list = document.getElementById('task-list');
    const countEl = document.getElementById('task-count');
    if (!list || !countEl) return;
    const total = list.querySelectorAll('.task-cb').length;
    const done = list.querySelectorAll('.task-cb:checked').length;
    countEl.textContent = `${total - done} LEFT`;
}

function addTask() {
    const input = document.getElementById('task-input-field');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;

    const list = document.getElementById('task-list');
    const li = document.createElement('li');
    li.className = 'task-item';
    li.innerHTML = `
        <input type="checkbox" class="task-cb" onchange="updateTaskCount()">
        <span class="task-text">${escapeHtml(text)}</span>`;
    list.appendChild(li);
    input.value = '';
    updateTaskCount();
}

function addTaskOnEnter(e) {
    if (e.key === 'Enter') { e.preventDefault(); addTask(); }
}

// ─── News Headlines ───────────────────────────────────────────────────────────
const NEWS_ICONS = ['🌍', '💡', '🔬', '📊', '🚀', '🎯', '⚡', '🌐'];

async function fetchNews() {
    const newsEl = document.getElementById('news-list');
    if (!newsEl) return;

    const RSS_SOURCES = [
        'https://feeds.bbci.co.uk/news/world/rss.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/World.xml'
    ];
    const apiUrl = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(RSS_SOURCES[0])}&count=3`;

    try {
        const res = await fetch(apiUrl);
        if (!res.ok) throw new Error('News feed failed');
        const data = await res.json();
        if (data.status !== 'ok' || !data.items?.length) throw new Error('No items');

        newsEl.innerHTML = data.items.slice(0, 3).map((item, i) => {
            const icon = NEWS_ICONS[i % NEWS_ICONS.length];
            const title = escapeHtml((item.title || '').slice(0, 90));
            const link = escapeHtml(item.link || '#');
            return `
            <div class="news-item" onclick="window.open('${link}', '_blank')">
                <div class="news-thumb">${icon}</div>
                <div class="news-text">
                    <div class="news-title">${title}</div>
                    <div class="news-source">BBC World</div>
                </div>
            </div>`;
        }).join('');
    } catch (e) {
        newsEl.innerHTML = `
            <div class="news-item">
                <div class="news-thumb">🌍</div>
                <div class="news-text"><div class="news-title">World leaders convene for climate summit</div><div class="news-source">Global News</div></div>
            </div>
            <div class="news-item">
                <div class="news-thumb">🔬</div>
                <div class="news-text"><div class="news-title">New AI model breaks benchmark records</div><div class="news-source">Tech Today</div></div>
            </div>
            <div class="news-item">
                <div class="news-thumb">🚀</div>
                <div class="news-text"><div class="news-title">Space agency announces lunar mission date</div><div class="news-source">Space Wire</div></div>
            </div>`;
    }
}

// ─── Home page initializer ────────────────────────────────────────────────────
function initHomePage() {
    initHomeParticles();
    buildMiniCal();
    fetchWeather();
    fetchNews();
    updateTaskCount();

    // Set greeting based on time of day
    const hour = new Date().getHours();
    let greeting = 'WELCOME BACK.';
    if (hour < 5) greeting = 'GOOD NIGHT.';
    else if (hour < 12) greeting = 'GOOD MORNING.';
    else if (hour < 17) greeting = 'GOOD AFTERNOON.';
    else if (hour < 21) greeting = 'GOOD EVENING.';
    const greetEl = document.getElementById('home-greeting');
    if (greetEl) greetEl.textContent = greeting;
}

// ─── Bootstrap home page on DOM ready ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initHomePage();
    showView('home');
});


// ═══════════════════════════════════════════════════════════════
// VOICE VIEW — Full-screen voice assistant mode
// ═══════════════════════════════════════════════════════════════

// ─── Neural Network Particle Canvas ──────────────────────────────────────────
let voiceParticleCanvas = null;
let voiceParticleCtx = null;
let voiceParticleAF = null;
let voiceNodes = [];
const VP_NODE_COUNT = 55;
const VP_CONNECT_DIST = 140;

function initVoiceParticles() {
    const canvas = document.getElementById('voice-particles');
    if (!canvas) return;
    voiceParticleCanvas = canvas;
    voiceParticleCtx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    voiceNodes = Array.from({ length: VP_NODE_COUNT }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.8 + 0.6,
        alpha: Math.random() * 0.5 + 0.3,
    }));

    function draw() {
        const ctx = voiceParticleCtx;
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        voiceNodes.forEach(n => {
            n.x += n.vx; n.y += n.vy;
            if (n.x < 0 || n.x > W) n.vx *= -1;
            if (n.y < 0 || n.y > H) n.vy *= -1;
        });
        for (let i = 0; i < voiceNodes.length; i++) {
            for (let j = i + 1; j < voiceNodes.length; j++) {
                const a = voiceNodes[i], b = voiceNodes[j];
                const dx = a.x - b.x, dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < VP_CONNECT_DIST) {
                    const alpha = (1 - dist / VP_CONNECT_DIST) * 0.22;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = `rgba(0, 180, 255, ${alpha})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }
        voiceNodes.forEach(n => {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 229, 255, ${n.alpha})`;
            ctx.fill();
        });
        voiceParticleAF = requestAnimationFrame(draw);
    }
    if (voiceParticleAF) cancelAnimationFrame(voiceParticleAF);
    draw();
}

function stopVoiceParticles() {
    if (voiceParticleAF) { cancelAnimationFrame(voiceParticleAF); voiceParticleAF = null; }
}

// ─── Voice View Waveform ──────────────────────────────────────────────────────
function buildVoiceViewWaveform() {
    const container = document.getElementById('vv-waveform');
    if (!container) return;
    container.innerHTML = '';
    const count = 48;
    for (let i = 0; i < count; i++) {
        const bar = document.createElement('div');
        bar.className = 'vv-wave-bar';
        const center = Math.abs(i - count / 2);
        const baseH = Math.max(4, 26 - center * 0.85);
        const peakH = baseH + Math.random() * 18 + 8;
        const dur = (Math.random() * 0.8 + 0.8).toFixed(2);
        const delay = (Math.random() * 0.8).toFixed(2);
        bar.style.setProperty('--base-h', `${baseH}px`);
        bar.style.setProperty('--peak-h', `${peakH}px`);
        bar.style.setProperty('--dur', `${dur}s`);
        bar.style.setProperty('--delay', `-${delay}s`);
        container.appendChild(bar);
    }
}

// ─── Voice View State Machine ─────────────────────────────────────────────────
let voiceViewState = 'idle';

function setVoiceViewState(state) {
    voiceViewState = state;
    const orb = document.getElementById('vv-orb-wrap');
    const waveform = document.getElementById('vv-waveform');
    const statusEl = document.getElementById('vv-status-text');
    const hintEl = document.getElementById('vv-hint-text');
    if (!orb) return;

    orb.classList.remove('listening', 'speaking');
    if (waveform) waveform.classList.remove('wv-active', 'wv-speaking');
    if (statusEl) statusEl.classList.remove('listening', 'speaking');

    switch (state) {
        case 'idle':
            if (statusEl) { statusEl.textContent = 'TAP TO SPEAK'; }
            if (hintEl) hintEl.style.opacity = '1';
            break;
        case 'listening':
            orb.classList.add('listening');
            if (waveform) waveform.classList.add('wv-active');
            if (statusEl) { statusEl.textContent = 'LISTENING\u2026'; statusEl.classList.add('listening'); }
            if (hintEl) hintEl.style.opacity = '0.4';
            break;
        case 'processing':
            if (statusEl) { statusEl.textContent = 'PROCESSING\u2026'; }
            if (hintEl) hintEl.style.opacity = '0.4';
            break;
        case 'speaking':
            orb.classList.add('speaking');
            if (waveform) waveform.classList.add('wv-speaking');
            if (statusEl) { statusEl.textContent = 'NOVA IS SPEAKING\u2026'; statusEl.classList.add('speaking'); }
            if (hintEl) hintEl.style.opacity = '0.4';
            break;
    }
}

function updateVoiceViewTranscript(text) {
    const el = document.getElementById('vv-transcript');
    if (!el) return;
    if (text) {
        el.textContent = '\u201c' + text + '\u201d';
        el.classList.add('visible');
    }
    else {
        el.classList.remove('visible');
        el.textContent = '';
    }
}

function isVoiceViewActive() {
    return document.getElementById('view-voice')?.classList.contains('view-active');
}

// ─── Mic button on voice view ─────────────────────────────────────────────────
function handleVoiceViewMic() {
    if (isSpeaking) { stopSpeaking(); setVoiceViewState('idle'); return; }
    if (isListening) { if (recognition) recognition.stop(); setVoiceViewState('idle'); return; }
    if (!recognition) { showToast('Speech recognition not supported in this browser', 'warning'); return; }
    // Show instant visual feedback while browser negotiates the mic
    setVoiceViewState('listening');
    voiceBuffer = '';
    recognition.start();
}

// ─── Patch speech recognition to update Voice View state ─────────────────────
let _vvPatched = false;
function patchRecognitionForVoiceView() {
    if (_vvPatched || !recognition) return;
    _vvPatched = true;

    const origStart = recognition.onstart;
    const origResult = recognition.onresult;
    const origEnd = recognition.onend;
    const origError = recognition.onerror;

    recognition.onstart = function (e) {
        if (origStart) origStart.call(this, e);
        if (isVoiceViewActive()) setVoiceViewState('listening');
    };

    recognition.onresult = function (e) {
        if (origResult) origResult.call(this, e);
        if (isVoiceViewActive()) {
            // Build full transcript from ALL results to avoid duplication
            let fullTranscript = '';
            for (let i = 0; i < e.results.length; i++) {
                fullTranscript += e.results[i][0].transcript;
            }
            fullTranscript = fullTranscript.trim();
            if (fullTranscript) updateVoiceViewTranscript(fullTranscript);
        }
    };

    recognition.onend = function (e) {
        const hadText = voiceBuffer.trim();
        if (origEnd) origEnd.call(this, e);
        if (isVoiceViewActive()) {
            if (hadText) {
                setVoiceViewState('processing');
                // sendVoiceMessage will navigate to chat
            } else {
                setVoiceViewState('idle');
                updateVoiceViewTranscript('');
            }
        }
    };

    recognition.onerror = function (e) {
        if (origError) origError.call(this, e);
        if (isVoiceViewActive()) { setVoiceViewState('idle'); updateVoiceViewTranscript(''); }
    };
}

// ─── Voice View: in-page response handler (streaming + sentence TTS) ─────────
// Override sendVoiceMessage: stay on voice view, stream LLM tokens, speak sentences as they arrive
const _vvOrigSendVoiceMessage = (typeof sendVoiceMessage === 'function') ? sendVoiceMessage : null;
sendVoiceMessage = async function (message) {
    if (!isVoiceViewActive()) {
        if (_vvOrigSendVoiceMessage) await _vvOrigSendVoiceMessage(message);
        return;
    }

    const chatLog = document.getElementById('vv-chat-log');

    // Create a message pair container
    const pair = document.createElement('div');
    pair.className = 'vv-chat-pair';

    // Add user bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'vv-chat-user';
    userBubble.textContent = message;
    pair.appendChild(userBubble);

    // Add processing indicator for Nova
    const novaBubble = document.createElement('div');
    novaBubble.className = 'vv-chat-nova processing';
    novaBubble.textContent = 'Thinking\u2026';
    pair.appendChild(novaBubble);

    if (chatLog) {
        chatLog.appendChild(pair);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Hide the live transcript now
    updateVoiceViewTranscript('');
    setVoiceViewState('processing');

    try {
        const replyText = await sendVoiceStream(message, {
            onToken(fullSoFar, _token) {
                // Show streaming text in Nova bubble (live update)
                novaBubble.className = 'vv-chat-nova';
                const display = fullSoFar.length > 400 ? '\u2026' + fullSoFar.slice(-400) : fullSoFar;
                novaBubble.textContent = display;
                if (chatLog) chatLog.scrollTop = chatLog.scrollHeight;
            },
            onSentence(_sentence) {
                // TTS is auto-queued by the pipeline; update visual state
                if (!isSpeaking) setVoiceViewState('speaking');
            },
            onDone(fullReply) {
                // Final display — show full reply (trimmed for readability)
                const displayReply = fullReply.length > 400 ? fullReply.slice(0, 400) + '\u2026' : fullReply;
                novaBubble.textContent = displayReply;
                if (chatLog) chatLog.scrollTop = chatLog.scrollHeight;
            },
            onError(err) {
                novaBubble.className = 'vv-chat-nova';
                novaBubble.textContent = 'Could not reach NOVA \u2014 is the server running?';
                setVoiceViewState('idle');
                showToast('Could not reach NOVA \u2014 is the server running?', 'error');
            },
        });

        if (replyText) {
            saveToHistory(message, replyText);
            setVoiceViewState('speaking');
        } else {
            setVoiceViewState('idle');
        }

    } catch (err) {
        novaBubble.className = 'vv-chat-nova';
        novaBubble.textContent = 'Could not reach NOVA \u2014 is the server running?';
        setVoiceViewState('idle');
        showToast('Could not reach NOVA \u2014 is the server running?', 'error');
    }
};
