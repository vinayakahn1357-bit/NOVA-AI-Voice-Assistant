/**
 * Voice Pro Features for NOVA Voice System
 *
 * Premium voice assistant enhancements:
 *   1. Live Audio Waveform — frequency-driven bars from real mic input
 *   2. Smart Interruption (Barge-in) — talk to interrupt Nova mid-speech
 *   3. Sound Effects — programmatic Web Audio chimes for state transitions
 *   4. Live Transcript Bubble — real-time interim words with cursor
 *   5. Voice Commands — local-only commands (stop, louder, repeat, clear)
 *   6. Orb Energy Glow — orb aura driven by mic energy
 *   7. Adaptive Silence Timeout — smart timeout based on speech length
 *
 * All features are event-driven, rAF-gated, and zero-cost when voice view is hidden.
 */

class VoiceProEngine {
    constructor(options = {}) {
        this._activityThreshold = options.activityThreshold || null;
        this._recognition       = options.recognition || null;
        this._onStopSpeaking    = options.onStopSpeaking || (() => {});
        this._onStartListening  = options.onStartListening || (() => {});
        this._getVolume         = options.getVolume || (() => 0.5);
        this._setVolume         = options.setVolume || (() => {});
        this._getLastReply      = options.getLastReply || (() => '');
        this._speakFn           = options.speakFn || (() => {});
        this._showToast         = options.showToast || (() => {});
        this._isSpeaking        = options.isSpeaking || (() => false);
        this._isListening       = options.isListening || (() => false);
        this._isVoiceViewActive = options.isVoiceViewActive || (() => false);

        // ── State ──
        this._rafId = null;
        this._running = false;
        this._bargeInCooldown = false;
        this._lastBargeInTime = 0;
        this._sfxCtx = null;
        this._lastReply = '';
        this._soundEnabled = true;

        // ── Feature 1: Waveform ──
        this._waveBars = [];
        this._waveBarCount = 0;
        this._frequencyData = null;

        // ── Feature 6: Orb glow ──
        this._orbWrap = null;
        this._smoothedEnergy = 0;

        // Init
        this._init();
    }

    _init() {
        // Cache DOM elements
        this._orbWrap = document.getElementById('vv-orb-wrap');
        const waveformEl = document.getElementById('vv-waveform');
        if (waveformEl) {
            this._waveBars = Array.from(waveformEl.querySelectorAll('.vv-wave-bar'));
            this._waveBarCount = this._waveBars.length;
        }

        // Prepare frequency data buffer if analyser exists
        this._syncAnalyser();

        console.info('[VoiceProEngine] Initialized — %d wave bars, sfx=%s',
            this._waveBarCount, this._soundEnabled);
    }

    /** Sync the analyser reference (call after activity threshold is attached) */
    _syncAnalyser() {
        if (this._activityThreshold && this._activityThreshold._analyser) {
            const analyser = this._activityThreshold._analyser;
            this._frequencyData = new Uint8Array(analyser.frequencyBinCount);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 1: LIVE AUDIO WAVEFORM
    // ═══════════════════════════════════════════════════════════════════════

    /** Start the rAF render loop for waveform + orb glow */
    startVisualizer() {
        if (this._running) return;
        this._running = true;
        this._syncAnalyser();
        this._renderFrame();
    }

    /** Stop the rAF render loop */
    stopVisualizer() {
        this._running = false;
        if (this._rafId) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        // Reset bars to idle
        this._resetWaveBars();
        // Reset orb energy
        if (this._orbWrap) this._orbWrap.style.setProperty('--orb-energy', '0');
    }

    /** @private */
    _renderFrame() {
        if (!this._running) return;

        // Only render when voice view is visible
        if (this._isVoiceViewActive()) {
            const analyser = this._activityThreshold?._analyser;

            if (analyser && this._frequencyData) {
                // Read frequency data
                analyser.getByteFrequencyData(this._frequencyData);

                // Feature 1: Drive waveform bars
                this._updateWaveBars(this._frequencyData);

                // Feature 6: Drive orb glow
                this._updateOrbGlow();
            }
        }

        this._rafId = requestAnimationFrame(() => this._renderFrame());
    }

    /** @private */
    _updateWaveBars(freqData) {
        if (this._waveBarCount === 0) return;

        const binCount = freqData.length;
        const binsPerBar = Math.floor(binCount / this._waveBarCount);

        for (let i = 0; i < this._waveBarCount; i++) {
            // Average the frequency bins for this bar
            let sum = 0;
            const start = i * binsPerBar;
            for (let j = start; j < start + binsPerBar && j < binCount; j++) {
                sum += freqData[j];
            }
            const avg = sum / binsPerBar; // 0-255

            // Map to height: min 3px, max 50px
            const height = 3 + (avg / 255) * 47;

            // Direct DOM update (fastest path)
            this._waveBars[i].style.height = height + 'px';
            // Kill CSS animation when we're driving directly
            this._waveBars[i].style.animation = 'none';
        }
    }

    /** @private */
    _resetWaveBars() {
        for (let i = 0; i < this._waveBarCount; i++) {
            this._waveBars[i].style.height = '';
            this._waveBars[i].style.animation = '';
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 2: SMART INTERRUPTION (BARGE-IN)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Call this periodically while Nova is speaking.
     * Requires SUSTAINED high energy (3 consecutive readings ≈ 600ms)
     * to avoid false triggers from background noise, TV, or echo.
     */
    checkBargeIn() {
        if (!this._isSpeaking()) return false;
        if (this._bargeInCooldown) return false;
        if (!this._activityThreshold) return false;
        if (!this._isVoiceViewActive()) return false;

        const energy = this._activityThreshold.getCurrentEnergy();
        const threshold = this._activityThreshold.getThreshold();

        // Require energy MUCH higher than threshold — 5× to filter out
        // background voices, TV, and Nova's own TTS echo
        if (energy > threshold * 5) {
            this._bargeInHits = (this._bargeInHits || 0) + 1;
        } else {
            // Reset counter if energy drops — must be sustained
            this._bargeInHits = 0;
            return false;
        }

        // Require 3 consecutive high-energy readings (~600ms of sustained speech)
        // This filters single spikes from doors, coughs, background TV
        if (this._bargeInHits < 3) return false;

        const now = Date.now();
        // Don't barge in within 3s of TTS starting (avoid picking up Nova's voice)
        if (now - this._ttsStartTime < 3000) return false;

        this._lastBargeInTime = now;
        this._bargeInCooldown = true;
        this._bargeInHits = 0;

        // Stop Nova speaking
        this._onStopSpeaking();
        this._showToast('Interrupted — listening…', 'info');
        this.playSFX('listen');

        // Brief delay for audio to clear, then start listening
        setTimeout(() => {
            this._bargeInCooldown = false;
            this._onStartListening();
        }, 400);

        return true;
    }

    /** Enable barge-in monitoring during TTS playback */
    startBargeInMonitor() {
        if (this._bargeInInterval) return;
        this._bargeInHits = 0;
        this._ttsStartTime = Date.now(); // track when TTS started for cooldown
        this._bargeInInterval = setInterval(() => {
            if (!this._isSpeaking()) {
                this.stopBargeInMonitor();
                return;
            }
            this.checkBargeIn();
        }, 200); // check every 200ms
    }

    /** Stop barge-in monitoring */
    stopBargeInMonitor() {
        if (this._bargeInInterval) {
            clearInterval(this._bargeInInterval);
            this._bargeInInterval = null;
        }
        this._bargeInCooldown = false;
        this._bargeInHits = 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 3: SOUND EFFECTS
    // ═══════════════════════════════════════════════════════════════════════

    /** @private */
    _getSFXCtx() {
        if (!this._sfxCtx) {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (Ctx) this._sfxCtx = new Ctx();
        }
        return this._sfxCtx;
    }

    /**
     * Play a sound effect.
     * @param {'listen'|'process'|'speak'|'error'} type
     */
    playSFX(type) {
        if (!this._soundEnabled) return;
        const ctx = this._getSFXCtx();
        if (!ctx) return;

        // Resume if suspended (browser autoplay policy)
        if (ctx.state === 'suspended') ctx.resume();

        const vol = Math.min(this._getVolume() * 0.15, 0.12); // max 12% volume
        const now = ctx.currentTime;

        switch (type) {
            case 'listen': this._playRisingChime(ctx, now, vol); break;
            case 'process': this._playDoubleBlip(ctx, now, vol); break;
            case 'speak': this._playWarmPulse(ctx, now, vol); break;
            case 'error': this._playDescendingTone(ctx, now, vol); break;
        }
    }

    /** Rising two-tone chime: 440→880Hz, 120ms */
    _playRisingChime(ctx, now, vol) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, now);
        osc.frequency.exponentialRampToValueAtTime(880, now + 0.12);
        gain.gain.setValueAtTime(vol, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.2);
    }

    /** Soft double-blip: 660Hz × 2, 60ms each */
    _playDoubleBlip(ctx, now, vol) {
        for (let i = 0; i < 2; i++) {
            const t = now + i * 0.14;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.value = 660;
            gain.gain.setValueAtTime(vol * 0.8, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
            osc.connect(gain).connect(ctx.destination);
            osc.start(t);
            osc.stop(t + 0.08);
        }
    }

    /** Low warm pulse: 220Hz, 200ms */
    _playWarmPulse(ctx, now, vol) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = 220;
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(vol * 0.6, now + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.25);
    }

    /** Descending tone: 440→220Hz, 150ms */
    _playDescendingTone(ctx, now, vol) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, now);
        osc.frequency.exponentialRampToValueAtTime(220, now + 0.15);
        gain.gain.setValueAtTime(vol, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.22);
    }

    /** Enable or disable sound effects */
    setSoundEnabled(enabled) {
        this._soundEnabled = !!enabled;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 4: LIVE TRANSCRIPT BUBBLE
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Update the live transcript with interim (partial) speech results.
     * Shows real-time words with a blinking cursor.
     *
     * @param {string} interimText - Current interim transcript
     * @param {string} finalText - Accumulated final transcript
     */
    updateLiveTranscript(interimText, finalText) {
        const el = document.getElementById('vv-transcript');
        if (!el) return;

        const combined = (finalText + ' ' + interimText).trim();
        if (combined) {
            // Show with cursor ▌
            el.innerHTML = '\u201c' + this._escapeHtml(combined) +
                '<span class="vv-cursor">▌</span>\u201d';
            el.classList.add('visible');
        } else {
            el.classList.remove('visible');
            el.textContent = '';
        }
    }

    /** @private */
    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 5: VOICE COMMANDS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Check if the transcript is a voice command.
     * Returns true if it was handled (caller should skip sendVoiceMessage).
     *
     * @param {string} text - The final transcript
     * @returns {boolean} Whether a command was matched and executed
     */
    handleVoiceCommand(text) {
        if (!text) return false;
        const cmd = text.toLowerCase().trim().replace(/[.,!?]/g, '');

        // ── Stop command ──
        if (this._matchCmd(cmd, ['stop', 'nova stop', 'stop talking', 'shut up', 'be quiet'])) {
            this._onStopSpeaking();
            this._showToast('⏹️ Stopped', 'info');
            this.playSFX('process');
            return true;
        }

        // ── Volume up ──
        if (this._matchCmd(cmd, ['louder', 'nova louder', 'volume up', 'speak louder', 'increase volume'])) {
            const cur = this._getVolume();
            this._setVolume(Math.min(1.0, cur + 0.2));
            this._showToast('🔊 Volume up: ' + Math.round(Math.min(1.0, cur + 0.2) * 100) + '%', 'info');
            this.playSFX('listen');
            return true;
        }

        // ── Volume down ──
        if (this._matchCmd(cmd, ['softer', 'quieter', 'nova softer', 'nova quieter', 'volume down', 'speak softer', 'decrease volume'])) {
            const cur = this._getVolume();
            this._setVolume(Math.max(0.05, cur - 0.2));
            this._showToast('🔉 Volume down: ' + Math.round(Math.max(0.05, cur - 0.2) * 100) + '%', 'info');
            this.playSFX('listen');
            return true;
        }

        // ── Repeat ──
        if (this._matchCmd(cmd, ['repeat', 'nova repeat', 'say that again', 'repeat that', 'say again'])) {
            const lastReply = this._getLastReply();
            if (lastReply) {
                this._speakFn(lastReply);
                this._showToast('🔁 Repeating…', 'info');
            } else {
                this._showToast('Nothing to repeat', 'warning');
            }
            return true;
        }

        // ── Clear chat ──
        if (this._matchCmd(cmd, ['clear', 'nova clear', 'clear chat', 'clear history', 'clear conversation'])) {
            const chatLog = document.getElementById('vv-chat-log');
            if (chatLog) chatLog.innerHTML = '';
            this._showToast('🗑️ Chat cleared', 'info');
            this.playSFX('process');
            return true;
        }

        // ── Mute ──
        if (this._matchCmd(cmd, ['mute', 'nova mute', 'mute voice'])) {
            this._setVolume(0);
            this._showToast('🔇 Muted — say "NOVA louder" to unmute', 'info');
            return true;
        }

        return false;
    }

    /** @private */
    _matchCmd(input, patterns) {
        return patterns.some(p => {
            // Exact match or starts with pattern
            return input === p || input.startsWith(p + ' ');
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 6: ORB ENERGY GLOW
    // ═══════════════════════════════════════════════════════════════════════

    /** @private — Called each rAF frame */
    _updateOrbGlow() {
        if (!this._orbWrap || !this._activityThreshold) return;

        const energy = this._activityThreshold.getCurrentEnergy();

        // Smooth the energy value to avoid jitter (exponential moving average)
        this._smoothedEnergy = this._smoothedEnergy * 0.7 + energy * 0.3;

        // Clamp to 0-1 range and apply a curve for more dramatic effect
        const mapped = Math.min(1, Math.pow(this._smoothedEnergy * 8, 1.5));

        // Set CSS custom property — CSS handles the visual mapping
        this._orbWrap.style.setProperty('--orb-energy', mapped.toFixed(3));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEATURE 7: ADAPTIVE SILENCE TIMEOUT
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Calculate the optimal silence timeout based on speech characteristics.
     *
     * @param {string} voiceBuffer - Current accumulated transcript
     * @returns {number} Timeout in milliseconds
     */
    getAdaptiveSilenceTimeout(voiceBuffer) {
        if (!voiceBuffer || !voiceBuffer.trim()) return 1500; // default

        const text = voiceBuffer.trim();
        const words = text.split(/\s+/);
        const wordCount = words.length;
        const lastWord = words[words.length - 1].toLowerCase().replace(/[.,!?]/g, '');

        // Trailing conjunction — user clearly has more to say
        const trailingConjunctions = ['and', 'but', 'or', 'also', 'plus', 'then', 'so', 'because', 'since'];
        if (trailingConjunctions.includes(lastWord)) {
            return 3000;
        }

        // Trailing preposition — mid-sentence
        const trailingPrepositions = ['to', 'for', 'with', 'about', 'from', 'in', 'on', 'at', 'of', 'the', 'a', 'an'];
        if (trailingPrepositions.includes(lastWord)) {
            return 2500;
        }

        // Question word at start — user composing a question
        const questionWords = ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'];
        const firstWord = words[0].toLowerCase().replace(/[.,!?]/g, '');
        if (questionWords.includes(firstWord) && wordCount < 8) {
            return 2000;
        }

        // Length-based scaling
        if (wordCount <= 3) return 1200;     // Short command
        if (wordCount <= 10) return 1500;    // Normal
        if (wordCount <= 20) return 2000;    // Medium
        return 2500;                          // Long — user explaining something
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /** Update references after initialization */
    updateRefs(refs) {
        if (refs.activityThreshold !== undefined) this._activityThreshold = refs.activityThreshold;
        if (refs.recognition !== undefined) this._recognition = refs.recognition;
    }

    /** Store last AI reply for repeat command */
    setLastReply(text) {
        this._lastReply = text;
    }

    /** Full cleanup */
    destroy() {
        this.stopVisualizer();
        this.stopBargeInMonitor();
        if (this._sfxCtx && this._sfxCtx.state !== 'closed') {
            this._sfxCtx.close().catch(() => {});
        }
    }
}

// Export
if (typeof window !== 'undefined') {
    window.VoiceProEngine = VoiceProEngine;
}
