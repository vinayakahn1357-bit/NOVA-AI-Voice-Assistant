/**
 * Optional Push-to-Talk (PTT) for NOVA Voice System
 *
 * Adds an optional Push-to-Talk mode where the mic is only active while
 * a button is held or a key is pressed. Ideal for:
 *   - Noisy rooms / crowded environments
 *   - Classrooms and public demonstrations
 *   - Reducing accidental activations
 *
 * Activation:
 *   - Settings toggle in Voice section
 *   - Ctrl+Shift+P keyboard shortcut
 *
 * PTT key: Spacebar (when input field is not focused)
 * PTT button: On-screen floating button in voice view
 *
 * Usage:
 *   const ptt = new PushToTalk({
 *     recognition:      recognition,
 *     micPreprocessor:   _novaMicPreprocessor,
 *     onPTTStart: () => console.log('PTT active'),
 *     onPTTStop:  () => console.log('PTT released'),
 *   });
 *   ptt.setEnabled(true);
 *   // ... later:
 *   ptt.destroy();
 */

class PushToTalk {
    /**
     * @param {Object} options
     * @param {Object}   [options.recognition]          SpeechRecognition instance
     * @param {Object}   [options.micPreprocessor]      MicPreprocessor instance
     * @param {string}   [options.pttKey='Space']       Key code for PTT activation
     * @param {number}   [options.debounceMs=150]       Debounce for rapid press/release (ms)
     * @param {Function} [options.onPTTStart]           Called when PTT is pressed
     * @param {Function} [options.onPTTStop]            Called when PTT is released
     * @param {Function} [options.onModeChanged]        Called when PTT mode is toggled (enabled: bool)
     */
    constructor(options = {}) {
        this._recognition     = options.recognition || null;
        this._micPreprocessor = options.micPreprocessor || null;
        this._pttKey          = options.pttKey || 'Space';
        this._debounceMs      = options.debounceMs || 150;

        this._onPTTStart      = options.onPTTStart || (() => {});
        this._onPTTStop       = options.onPTTStop || (() => {});
        this._onModeChanged   = options.onModeChanged || (() => {});

        /** @private */ this._enabled = false;
        /** @private */ this._active = false; // currently holding PTT
        /** @private */ this._debounceTimer = null;
        /** @private */ this._pttButton = null;

        // Restore saved preference
        const savedPref = localStorage.getItem('nova_ptt_enabled');
        if (savedPref === 'true') {
            this._enabled = true;
        }

        // Bind event handlers
        this._boundKeyDown   = (e) => this._handleKeyDown(e);
        this._boundKeyUp     = (e) => this._handleKeyUp(e);
        this._boundBlur      = ()  => this._handleBlur();
        this._boundShortcut  = (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'P') {
                e.preventDefault();
                this.toggleEnabled();
            }
        };

        // Attach listeners
        document.addEventListener('keydown', this._boundKeyDown);
        document.addEventListener('keyup', this._boundKeyUp);
        window.addEventListener('blur', this._boundBlur);
        document.addEventListener('keydown', this._boundShortcut);

        // Create on-screen PTT button for voice view
        this._createPTTButton();

        // Sync UI state
        this._syncUIState();

        console.info('[PushToTalk] Initialized — enabled=%s, key=%s', this._enabled, this._pttKey);
    }

    /** @private */
    _createPTTButton() {
        // Create a floating PTT button for voice view
        this._pttButton = document.createElement('button');
        this._pttButton.id = 'nova-ptt-btn';
        this._pttButton.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
            <span>HOLD TO TALK</span>
        `;
        this._pttButton.style.cssText = `
            position: fixed;
            bottom: 90px;
            left: 50%;
            transform: translateX(-50%);
            display: none;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background: rgba(0, 229, 255, 0.12);
            border: 1.5px solid rgba(0, 229, 255, 0.3);
            border-radius: 24px;
            color: rgba(0, 229, 255, 0.8);
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1.5px;
            cursor: pointer;
            z-index: 9999;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            transition: all 0.2s ease;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
        `;

        // Mouse/touch events for on-screen button
        this._pttButton.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this._startPTT();
        });
        this._pttButton.addEventListener('mouseup', (e) => {
            e.preventDefault();
            this._stopPTT();
        });
        this._pttButton.addEventListener('mouseleave', () => {
            if (this._active) this._stopPTT();
        });

        // Touch support
        this._pttButton.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this._startPTT();
        }, { passive: false });
        this._pttButton.addEventListener('touchend', (e) => {
            e.preventDefault();
            this._stopPTT();
        }, { passive: false });
        this._pttButton.addEventListener('touchcancel', () => {
            if (this._active) this._stopPTT();
        });

        document.body.appendChild(this._pttButton);
    }

    /** @private */
    _handleKeyDown(e) {
        if (!this._enabled) return;
        if (e.code !== this._pttKey) return;
        if (e.repeat) return; // ignore held-key repeats

        // Don't intercept space when typing in an input/textarea
        const tag = document.activeElement?.tagName?.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

        e.preventDefault();
        this._startPTT();
    }

    /** @private */
    _handleKeyUp(e) {
        if (!this._enabled) return;
        if (e.code !== this._pttKey) return;

        // Don't intercept when typing
        const tag = document.activeElement?.tagName?.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

        e.preventDefault();
        this._stopPTT();
    }

    /** @private */
    _handleBlur() {
        // Tab lost focus while holding — auto-release
        if (this._active) {
            this._stopPTT();
        }
    }

    /** @private */
    _startPTT() {
        if (this._active) return;

        // Debounce rapid press/release
        if (this._debounceTimer) return;
        this._debounceTimer = setTimeout(() => {
            this._debounceTimer = null;
        }, this._debounceMs);

        this._active = true;

        // Visual feedback on button
        if (this._pttButton) {
            this._pttButton.style.background = 'rgba(0, 229, 255, 0.3)';
            this._pttButton.style.borderColor = 'rgba(0, 229, 255, 0.7)';
            this._pttButton.style.color = '#00e5ff';
            this._pttButton.style.boxShadow = '0 0 20px rgba(0, 229, 255, 0.3)';
            const span = this._pttButton.querySelector('span');
            if (span) span.textContent = '🔴 LISTENING…';
        }

        // Start recognition
        if (this._recognition) {
            // In PTT mode, use non-continuous (auto-stop when user stops speaking)
            this._recognition.continuous = false;
            try {
                // Ensure mic preprocessor is active
                if (this._micPreprocessor && !this._micPreprocessor.isActive) {
                    this._micPreprocessor.init().catch(() => {});
                }
                // voiceBuffer is reset by recognition.onstart handler in script.js
                this._recognition.start();
            } catch (e) {
                // Already started or other error
                console.warn('[PushToTalk] Could not start recognition:', e.message);
            }
        }

        this._onPTTStart();
        console.info('[PushToTalk] PTT pressed — mic active');
    }

    /** @private */
    _stopPTT() {
        if (!this._active) return;
        this._active = false;

        // Visual feedback
        if (this._pttButton) {
            this._pttButton.style.background = 'rgba(0, 229, 255, 0.12)';
            this._pttButton.style.borderColor = 'rgba(0, 229, 255, 0.3)';
            this._pttButton.style.color = 'rgba(0, 229, 255, 0.8)';
            this._pttButton.style.boxShadow = 'none';
            const span = this._pttButton.querySelector('span');
            if (span) span.textContent = 'HOLD TO TALK';
        }

        // Stop recognition (triggers onend → processes voiceBuffer)
        if (this._recognition) {
            try {
                this._recognition.stop();
            } catch (e) { /* already stopped */ }
            // Restore continuous mode immediately — will be set back to false
            // on the next _startPTT() call
            this._recognition.continuous = true;
        }

        this._onPTTStop();
        console.info('[PushToTalk] PTT released — mic off');
    }

    /**
     * Enable or disable PTT mode.
     * @param {boolean} enabled
     */
    setEnabled(enabled) {
        this._enabled = !!enabled;
        localStorage.setItem('nova_ptt_enabled', this._enabled ? 'true' : 'false');

        // If disabling while PTT is active, stop immediately
        if (!this._enabled && this._active) {
            this._stopPTT();
        }

        // Restore continuous mode when PTT is disabled
        if (!this._enabled && this._recognition) {
            this._recognition.continuous = true;
        }

        this._syncUIState();
        this._onModeChanged(this._enabled);

        console.info('[PushToTalk] Mode %s', this._enabled ? 'ENABLED' : 'DISABLED');
    }

    /**
     * Toggle PTT mode on/off.
     */
    toggleEnabled() {
        this.setEnabled(!this._enabled);
    }

    /** @private */
    _syncUIState() {
        // Show/hide on-screen PTT button based on mode
        if (this._pttButton) {
            // Only show in voice view and when PTT is enabled
            const isVoiceView = document.getElementById('view-voice')?.classList.contains('view-active');
            this._pttButton.style.display = (this._enabled && isVoiceView) ? 'flex' : 'none';
        }

        // Sync settings toggle if it exists
        const toggle = document.getElementById('ptt-toggle');
        if (toggle) {
            toggle.classList.toggle('active', this._enabled);
        }
    }

    /**
     * Update visibility of PTT button (call when view changes).
     */
    syncVisibility() {
        this._syncUIState();
    }

    /**
     * Update references (if recognition/mic are created after this instance).
     * @param {Object} refs - { recognition, micPreprocessor }
     */
    updateRefs(refs) {
        if (refs.recognition !== undefined) this._recognition = refs.recognition;
        if (refs.micPreprocessor !== undefined) this._micPreprocessor = refs.micPreprocessor;
    }

    /** @returns {boolean} Whether PTT mode is enabled */
    get isEnabled() { return this._enabled; }

    /** @returns {boolean} Whether PTT is currently being held */
    get isActive() { return this._active; }

    /**
     * Cleanup — remove DOM, event listeners.
     */
    destroy() {
        if (this._active) this._stopPTT();

        document.removeEventListener('keydown', this._boundKeyDown);
        document.removeEventListener('keyup', this._boundKeyUp);
        window.removeEventListener('blur', this._boundBlur);
        document.removeEventListener('keydown', this._boundShortcut);

        if (this._pttButton && this._pttButton.parentNode) {
            this._pttButton.parentNode.removeChild(this._pttButton);
        }
        this._pttButton = null;

        if (this._debounceTimer) {
            clearTimeout(this._debounceTimer);
            this._debounceTimer = null;
        }
    }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.PushToTalk = PushToTalk;
}
