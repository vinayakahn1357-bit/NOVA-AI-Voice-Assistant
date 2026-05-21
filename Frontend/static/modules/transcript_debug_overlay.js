/**
 * Transcript Debug Overlay for NOVA Voice System
 *
 * Optional developer HUD that displays real-time voice pipeline diagnostics:
 *   - Raw transcript (before filtering)
 *   - Cleaned transcript (after dedup/filter)
 *   - Confidence score with visual bar
 *   - Live audio energy level
 *   - Noise rejection count
 *   - Wake-word and PTT state
 *   - Device info
 *   - StreamAudioGuard stats
 *   - Event log (last 12 events)
 *
 * Hidden by default. Activated via:
 *   - Ctrl+Shift+D keyboard shortcut
 *   - window.NOVA_DEBUG = true in console
 *   - URL parameter ?debug=1
 *
 * Usage:
 *   const overlay = new NovaDebugOverlay();
 *   overlay.update('rawTranscript', 'hello world hello world');
 *   overlay.update('confidence', 0.87);
 *   overlay.log('noise_reject', 'Low energy: 0.003');
 *   overlay.toggle(); // show/hide
 */

class NovaDebugOverlay {
    constructor() {
        /** @private */ this._visible = false;
        /** @private */ this._container = null;
        /** @private */ this._fields = {};
        /** @private */ this._eventLog = [];
        /** @private */ this._maxEvents = 12;
        /** @private */ this._energyAnimFrame = null;

        // Auto-show if URL param or global flag
        const autoShow = new URLSearchParams(window.location.search).has('debug')
                       || window.NOVA_DEBUG === true;

        this._createDOM();

        if (autoShow) {
            // Defer to let DOM settle
            setTimeout(() => this.show(), 500);
        }

        // Keyboard shortcut: Ctrl+Shift+D
        this._boundKeyHandler = (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.toggle();
            }
        };
        document.addEventListener('keydown', this._boundKeyHandler);
    }

    /** @private */
    _createDOM() {
        // Container
        this._container = document.createElement('div');
        this._container.id = 'nova-debug-overlay';
        this._container.style.cssText = `
            position: fixed;
            bottom: 16px;
            right: 16px;
            width: 340px;
            max-height: 420px;
            background: rgba(10, 14, 26, 0.92);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 12px;
            padding: 0;
            z-index: 99999;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            font-size: 11px;
            color: rgba(255, 255, 255, 0.85);
            display: none;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 1px rgba(0, 229, 255, 0.3);
            user-select: text;
            transition: opacity 0.2s ease;
        `;

        // Header
        const header = document.createElement('div');
        header.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(0, 229, 255, 0.08);
            border-bottom: 1px solid rgba(0, 229, 255, 0.15);
        `;
        header.innerHTML = `
            <span style="color: #00e5ff; font-weight: 600; font-size: 10px; letter-spacing: 1.2px;">
                🔬 NOVA VOICE DEBUG
            </span>
        `;
        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✕';
        closeBtn.style.cssText = `
            background: none; border: none; color: rgba(255,255,255,0.5);
            cursor: pointer; font-size: 13px; padding: 0 2px; line-height: 1;
        `;
        closeBtn.onclick = () => this.hide();
        header.appendChild(closeBtn);
        this._container.appendChild(header);

        // Body
        const body = document.createElement('div');
        body.style.cssText = `
            padding: 8px 12px;
            overflow-y: auto;
            max-height: 360px;
        `;

        // Define fields
        const fieldDefs = [
            { key: 'rawTranscript',     label: 'RAW',        type: 'text' },
            { key: 'cleanedTranscript', label: 'CLEANED',    type: 'text' },
            { key: 'confidence',        label: 'CONFIDENCE', type: 'bar' },
            { key: 'energy',            label: 'ENERGY',     type: 'bar' },
            { key: 'threshold',         label: 'THRESHOLD',  type: 'value' },
            { key: 'speechActive',      label: 'SPEECH',     type: 'badge' },
            { key: 'pttMode',           label: 'PTT',        type: 'badge' },
            { key: 'device',            label: 'DEVICE',     type: 'text' },
            { key: 'guardSessions',     label: 'SESSIONS',   type: 'value' },
            { key: 'guardRestarts',     label: 'RESTARTS',   type: 'value' },
            { key: 'guardErrors',       label: 'ERRORS',     type: 'value' },
            { key: 'rejected',          label: 'REJECTED',   type: 'value' },
        ];

        for (const def of fieldDefs) {
            const row = document.createElement('div');
            row.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 3px;
                min-height: 18px;
            `;

            const label = document.createElement('span');
            label.style.cssText = `
                color: rgba(0, 229, 255, 0.6);
                font-size: 9px;
                letter-spacing: 0.8px;
                min-width: 72px;
                flex-shrink: 0;
            `;
            label.textContent = def.label;
            row.appendChild(label);

            if (def.type === 'bar') {
                // Progress bar + value
                const barWrap = document.createElement('div');
                barWrap.style.cssText = `
                    flex: 1;
                    height: 6px;
                    background: rgba(255,255,255,0.08);
                    border-radius: 3px;
                    overflow: hidden;
                `;
                const barFill = document.createElement('div');
                barFill.style.cssText = `
                    width: 0%;
                    height: 100%;
                    background: linear-gradient(90deg, #00e5ff, #bc60d3);
                    border-radius: 3px;
                    transition: width 0.15s ease;
                `;
                barWrap.appendChild(barFill);
                row.appendChild(barWrap);

                const val = document.createElement('span');
                val.style.cssText = `
                    min-width: 36px;
                    text-align: right;
                    font-size: 10px;
                    color: rgba(255,255,255,0.7);
                `;
                val.textContent = '0.00';
                row.appendChild(val);

                this._fields[def.key] = { bar: barFill, val };
            } else if (def.type === 'badge') {
                const badge = document.createElement('span');
                badge.style.cssText = `
                    font-size: 9px;
                    padding: 1px 6px;
                    border-radius: 4px;
                    background: rgba(255,255,255,0.08);
                    color: rgba(255,255,255,0.5);
                `;
                badge.textContent = '—';
                row.appendChild(badge);
                this._fields[def.key] = { badge };
            } else if (def.type === 'text') {
                const text = document.createElement('span');
                text.style.cssText = `
                    flex: 1;
                    font-size: 10px;
                    color: rgba(255,255,255,0.7);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                `;
                text.textContent = '—';
                row.appendChild(text);
                this._fields[def.key] = { text };
            } else {
                const val = document.createElement('span');
                val.style.cssText = `
                    font-size: 10px;
                    color: rgba(255,255,255,0.7);
                `;
                val.textContent = '—';
                row.appendChild(val);
                this._fields[def.key] = { val };
            }

            body.appendChild(row);
        }

        // Separator
        const sep = document.createElement('div');
        sep.style.cssText = `
            margin: 6px 0;
            border-top: 1px solid rgba(0, 229, 255, 0.1);
        `;
        body.appendChild(sep);

        // Event log header
        const logHeader = document.createElement('div');
        logHeader.style.cssText = `
            color: rgba(0, 229, 255, 0.5);
            font-size: 9px;
            letter-spacing: 0.8px;
            margin-bottom: 4px;
        `;
        logHeader.textContent = 'EVENT LOG';
        body.appendChild(logHeader);

        // Event log container
        this._logContainer = document.createElement('div');
        this._logContainer.style.cssText = `
            max-height: 110px;
            overflow-y: auto;
            font-size: 9px;
            color: rgba(255,255,255,0.45);
            line-height: 1.5;
        `;
        body.appendChild(this._logContainer);

        this._container.appendChild(body);
        document.body.appendChild(this._container);
    }

    /**
     * Update a specific debug field.
     * @param {string} key - Field key (e.g., 'rawTranscript', 'confidence', 'energy')
     * @param {*} value - The value to display
     */
    update(key, value) {
        if (!this._visible) return; // Skip DOM updates when hidden
        const field = this._fields[key];
        if (!field) return;

        if (field.bar) {
            // Progress bar
            const pct = Math.min(100, Math.max(0, (parseFloat(value) || 0) * 100));
            field.bar.style.width = pct + '%';
            if (field.val) field.val.textContent = (parseFloat(value) || 0).toFixed(3);
        } else if (field.badge) {
            const isActive = value === true || value === 'on' || value === 'active';
            field.badge.textContent = isActive ? 'ACTIVE' : 'OFF';
            field.badge.style.background = isActive ? 'rgba(0, 229, 255, 0.2)' : 'rgba(255,255,255,0.08)';
            field.badge.style.color = isActive ? '#00e5ff' : 'rgba(255,255,255,0.5)';
        } else if (field.text) {
            field.text.textContent = String(value || '—');
            field.text.title = String(value || '');
        } else if (field.val) {
            field.val.textContent = String(value ?? '—');
        }
    }

    /**
     * Log an event to the event log.
     * @param {string} eventType - Short event type (e.g., 'noise_reject', 'device_change')
     * @param {string} detail - Event detail text
     */
    log(eventType, detail) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        this._eventLog.unshift({ time: timeStr, type: eventType, detail });
        if (this._eventLog.length > this._maxEvents) {
            this._eventLog.length = this._maxEvents;
        }

        if (!this._visible) return; // Skip DOM updates when hidden
        this._renderLog();
    }

    /** @private */
    _renderLog() {
        if (!this._logContainer) return;
        this._logContainer.innerHTML = this._eventLog.map(e => {
            const typeColors = {
                'noise_reject': '#ff6b7a',
                'speech_start': '#00e5ff',
                'speech_end':   '#bc60d3',
                'device_change': '#e2c077',
                'device_add':   '#7cff6b',
                'device_remove':'#ff6b7a',
                'ptt_press':    '#00e5ff',
                'ptt_release':  '#bc60d3',
                'recovery':     '#e2c077',
                'error':        '#ff4b5c',
                'guard':        '#bc60d3',
            };
            const color = typeColors[e.type] || 'rgba(255,255,255,0.4)';
            return `<div style="margin-bottom:1px;">
                <span style="color:rgba(255,255,255,0.3)">${e.time}</span>
                <span style="color:${color};font-weight:600">${e.type}</span>
                <span>${e.detail || ''}</span>
            </div>`;
        }).join('');
    }

    /**
     * Bulk update from StreamAudioGuard stats.
     * @param {Object} guardStats - Stats from StreamAudioGuard.stats
     */
    updateGuardStats(guardStats) {
        if (!guardStats) return;
        this.update('guardSessions', guardStats.totalSessions || 0);
        this.update('guardRestarts', guardStats.totalRestarts || 0);
        this.update('guardErrors', guardStats.totalErrors || 0);
    }

    /**
     * Bulk update from SpeechActivityThreshold stats.
     * @param {Object} thresholdStats - Stats from SpeechActivityThreshold.stats
     */
    updateThresholdStats(thresholdStats) {
        if (!thresholdStats) return;
        this.update('energy', thresholdStats.currentEnergy || 0);
        this.update('threshold', (thresholdStats.threshold || 0).toFixed(4));
        this.update('speechActive', thresholdStats.speechActive);
        this.update('rejected', thresholdStats.rejectedCount || 0);
    }

    /** Show the overlay. */
    show() {
        this._visible = true;
        if (this._container) this._container.style.display = 'block';
        this._renderLog(); // refresh log
    }

    /** Hide the overlay. */
    hide() {
        this._visible = false;
        if (this._container) this._container.style.display = 'none';
    }

    /** Toggle visibility. */
    toggle() {
        if (this._visible) this.hide();
        else this.show();
    }

    /** @returns {boolean} Whether the overlay is currently visible. */
    get isVisible() { return this._visible; }

    /**
     * Cleanup — remove DOM, event listeners.
     */
    destroy() {
        if (this._boundKeyHandler) {
            document.removeEventListener('keydown', this._boundKeyHandler);
            this._boundKeyHandler = null;
        }
        if (this._container && this._container.parentNode) {
            this._container.parentNode.removeChild(this._container);
        }
        this._container = null;
    }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.NovaDebugOverlay = NovaDebugOverlay;
}
