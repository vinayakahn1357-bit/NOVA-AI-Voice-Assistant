function getTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Generate the Audio Waveform bars dynamically (Symmetrical matching the design)
function createWaveform() {
    const waveformContainer = document.getElementById('waveform');
    if (!waveformContainer) return;
    waveformContainer.innerHTML = '';
    const numBars = 50;

    // We want a shape that is tall in the middle and short on edges
    for (let i = 0; i < numBars; i++) {
        let bar = document.createElement('div');
        bar.className = 'wave-bar';

        // Calculate a bell curve kind of height base
        let distanceToCenter = Math.abs((numBars / 2) - i);
        let maxBase = 45;
        let baseHeight = Math.max(5, maxBase - (distanceToCenter * 1.8));

        // Randomize the animation delay so they don't move exactly together
        bar.style.animationDelay = `${Math.random() * 0.8}s`;

        // Specific inline heights with custom property for the CSS keyframe
        bar.style.setProperty('--base-height', `${baseHeight}px`);
        // Randomize peak target slightly based on its base height
        let targetHeight = baseHeight + (Math.random() * 15 + 10);
        bar.style.setProperty('--target-height', `${targetHeight}px`);

        waveformContainer.appendChild(bar);
    }
}

// =================== HISTORY SYSTEM ===================
// Each session: { id, topic, timestamp, messages: [{role, text}] }
let chatSessions = JSON.parse(localStorage.getItem('nova_history') || '[]');
let currentSessionId = null;

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function saveToHistory(userMsg, novaReply) {
    // If no active session, create one with the first user message as topic
    if (!currentSessionId) {
        currentSessionId = generateId();
        const newSession = {
            id: currentSessionId,
            topic: userMsg.length > 45 ? userMsg.slice(0, 45) + '…' : userMsg,
            timestamp: new Date().toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
            messages: []
        };
        chatSessions.unshift(newSession); // newest first
    }

    const session = chatSessions.find(s => s.id === currentSessionId);
    if (session) {
        session.messages.push({ role: 'user', text: userMsg });
        session.messages.push({ role: 'nova', text: novaReply });
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

    // Clear All button at top
    const clearAllBtn = `<button class="history-clear-all-btn" onclick="clearAllHistory()">🗑 Clear All</button>`;

    const items = chatSessions.map(session => `
        <div class="history-item">
            <div class="hi-main" onclick="loadHistorySession('${session.id}')">
                <div class="hi-topic">${session.topic}</div>
                <div class="hi-meta">${session.timestamp} · ${Math.ceil(session.messages.length / 2)} exchange${session.messages.length > 2 ? 's' : ''}</div>
                <div class="hi-preview">▪ ${session.messages.length > 0 ? session.messages[session.messages.length - 1].text.slice(0, 60) + '…' : ''}</div>
            </div>
            <button class="hi-delete-btn" onclick="deleteHistorySession('${session.id}')" title="Delete">✕</button>
        </div>
    `).join('');

    list.innerHTML = clearAllBtn + items;
}

function deleteHistorySession(sessionId) {
    chatSessions = chatSessions.filter(s => s.id !== sessionId);
    // If the deleted session was active, reset current session
    if (currentSessionId === sessionId) {
        currentSessionId = null;
        const chatBox = document.getElementById('chat-box');
        if (chatBox) chatBox.innerHTML = '';
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
    localStorage.removeItem('nova_history');
    renderHistoryList();
}

function loadHistorySession(sessionId) {
    const session = chatSessions.find(s => s.id === sessionId);
    if (!session) return;

    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = '';
    currentSessionId = sessionId;

    session.messages.forEach(msg => {
        const div = document.createElement('div');
        if (msg.role === 'user') {
            div.className = 'message user';
            div.innerHTML = `<div class="msg-body">${msg.text}</div>`;
        } else {
            div.className = 'message nova';
            div.innerHTML = `<div class="msg-header"><span class="sender">Nova:</span> Response</div><div class="msg-body">${msg.text}</div>`;
        }
        chatBox.appendChild(div);
    });

    chatBox.scrollTop = chatBox.scrollHeight;
    // Close history panel and show chat
    toggleHistoryPanel();
}

function toggleHistoryPanel() {
    const panel = document.getElementById('history-panel');
    if (!panel) return;
    renderHistoryList();
    panel.classList.toggle('open');
}

// =================== SEND MESSAGE ===================
async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const message = inputField.value.trim();

    if (!message) return;

    // Show USER message
    const userDiv = document.createElement('div');
    userDiv.className = 'message user';
    userDiv.innerHTML = `<div class="msg-body">${message}</div>`;
    chatBox.appendChild(userDiv);

    inputField.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    // Show typing indicator
    const typingDiv = document.createElement("div");
    typingDiv.className = "message nova";
    typingDiv.id = "typing-indicator";
    typingDiv.innerHTML = `
        <div class="msg-header">Nova: <span style="font-weight: normal; color: #fff;">Processing request...</span></div>
        <div class="msg-body">...</div>
    `;
    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        }).catch(err => null);

        let replyText = "Action completed successfully.";

        if (response && response.ok) {
            const data = await response.json();
            replyText = data.reply || replyText;
        }

        setTimeout(() => {
            typingDiv.remove();

            const responseDiv = document.createElement("div");
            responseDiv.className = "message nova";
            responseDiv.innerHTML = `
                <div class="msg-header"><span class="sender">Nova:</span> Response</div>
                <div class="msg-body">${replyText}</div>
            `;
            chatBox.appendChild(responseDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            // Save this exchange to history
            saveToHistory(message, replyText);
        }, 1200);

    } catch (error) {
        typingDiv.remove();
        chatBox.innerHTML += `
        <div class="message nova" style="border-left-color: red;">
            <div class="msg-header"><span class="sender" style="color: red;">System Error</span></div>
            <div class="msg-body">Core offline.</div>
        </div>`;
    }

    chatBox.scrollTop = chatBox.scrollHeight;
    inputField.focus();
}

// Real System Stats from Backend (/system uses psutil)
let lastCpu = 0;
let lastMemPercent = 0;
let totalRamGB = 0;

// Fetch total RAM once on load
async function fetchTotalRam() {
    try {
        const resp = await fetch('/system');
        if (resp.ok) {
            const data = await resp.json();
            // psutil gives percent; estimate total from first read if not stored
            // We'll store it separately once we get a reading
            const memInfo = await fetch('/system_detail').catch(() => null);
            if (memInfo && memInfo.ok) {
                const d = await memInfo.json();
                totalRamGB = d.total_gb || 0;
            }
        }
    } catch { }
}

async function updateSystemStats() {
    try {
        const resp = await fetch('/system');
        if (!resp.ok) throw new Error('fetch failed');
        const data = await resp.json();

        lastCpu = Math.round(data.cpu);
        lastMemPercent = Math.round(data.memory);
        const memGB = data.memory_used_gb !== undefined ? data.memory_used_gb : null;
        const totalGB = data.memory_total_gb !== undefined ? data.memory_total_gb : null;

        const cpuBar = document.getElementById("cpu-bar");
        const memoryBar = document.getElementById("memory-bar");
        const cpuText = document.getElementById("cpu-value");
        const memoryText = document.getElementById("memory-value");

        if (cpuText) cpuText.textContent = lastCpu + "%";
        if (cpuBar) cpuBar.style.width = lastCpu + "%";

        if (memoryText) {
            if (memGB !== null && totalGB !== null) {
                memoryText.textContent = `${memGB.toFixed(1)}GB/${totalGB.toFixed(0)}GB`;
            } else {
                memoryText.textContent = lastMemPercent + "%";
            }
        }
        if (memoryBar) memoryBar.style.width = lastMemPercent + "%";

    } catch (e) {
        // Backend unreachable — keep last known values, no flicker
    }
}

// Network Status Detection
function updateNetworkStatus() {
    const networkVal = document.querySelector('.health-network .stat-value');
    if (!networkVal) return;

    if (navigator.onLine) {
        networkVal.textContent = 'Active';
        networkVal.style.color = 'var(--core-cyan)';
        networkVal.style.textShadow = '0 0 5px rgba(0, 229, 255, 0.5)';
    } else {
        networkVal.textContent = 'Offline';
        networkVal.style.color = '#ff4b5c';
        networkVal.style.textShadow = '0 0 5px rgba(255, 75, 92, 0.5)';
    }
}

window.addEventListener('online', updateNetworkStatus);
window.addEventListener('offline', updateNetworkStatus);

// Initialize on Load
document.addEventListener("DOMContentLoaded", function () {
    createWaveform();
    updateNetworkStatus(); // Run once on load

    // Update lively system stats every 2.5 seconds
    setInterval(updateSystemStats, 2500);

    const input = document.getElementById("user-input");
    if (input) {
        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                sendMessage();
            }
        });
    }
});
// --------------------
// Unified Voice Recognition
// --------------------
let recognition;
let isListening = false;

function setVoiceActiveState(active) {
    isListening = active;
    const statusEl = document.querySelector('.voice-status');
    const hintEl = document.querySelector('.voice-hint');
    // All mic/voice buttons: mic-orb-btn + any .voice-btn in right panel
    const allMicBtns = document.querySelectorAll('.mic-orb-btn, .voice-btn');
    const micRing = document.querySelector('.mic-orb-ring');

    if (active) {
        if (statusEl) statusEl.textContent = '🔴 Listening...';
        if (hintEl) hintEl.textContent = 'Speak now — Nova is hearing you';
        allMicBtns.forEach(btn => btn.classList.add('voice-active'));
        if (micRing) micRing.style.animationDuration = '0.6s';
    } else {
        if (statusEl) statusEl.textContent = "Listening for 'Nova'...";
        if (hintEl) hintEl.textContent = 'How can Nova help you today? (say a command)';
        allMicBtns.forEach(btn => btn.classList.remove('voice-active'));
        if (micRing) micRing.style.animationDuration = '2.5s';
    }
}

function startVoiceRecognition() {
    if (!recognition) {
        alert("Speech recognition not supported in this browser.");
        return;
    }
    if (isListening) {
        recognition.stop();
        return;
    }
    recognition.start();
}

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = false;

    recognition.onstart = function () {
        setVoiceActiveState(true);
    };

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        const inputBox = document.getElementById("user-input");
        if (inputBox) inputBox.value = transcript;
        sendMessage();
    };

    recognition.onend = function () {
        setVoiceActiveState(false);
    };

    recognition.onerror = function (event) {
        console.error("Speech recognition error:", event.error);
        setVoiceActiveState(false);
    };
}

// Wire up ALL voice buttons after DOM is ready
document.addEventListener("DOMContentLoaded", function () {
    // Mic orb button in voice bar
    const micOrbBtn = document.querySelector('.mic-orb-btn');
    if (micOrbBtn) micOrbBtn.addEventListener('click', startVoiceRecognition);

    // Voice Assistant button in Core Capabilities (right panel)
    document.querySelectorAll('.voice-btn').forEach(btn => {
        btn.addEventListener('click', startVoiceRecognition);
    });
});

function speak(text) {
    const speech = new SpeechSynthesisUtterance(text);
    speech.rate = 1;
    speech.pitch = 1;
    speech.lang = "en-US";
    window.speechSynthesis.speak(speech);
}
