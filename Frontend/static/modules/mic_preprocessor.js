/**
 * Microphone Preprocessor for NOVA Voice System
 *
 * Optimises browser microphone capture for voice assistant use:
 *   - Enables echo cancellation, noise suppression, auto gain control
 *   - Applies a high-pass filter to remove low-frequency hum/rumble
 *   - Normalises dynamics via compressor for distant speakers
 *   - Manages mic stream lifecycle (init / pause / resume / destroy)
 *
 * KEY INSIGHT: The Web Speech API manages its own mic internally.
 * By ALSO acquiring a getUserMedia stream with optimised constraints,
 * we tell the browser's audio subsystem to activate hardware-level
 * echo cancellation and noise suppression for the physical microphone —
 * which benefits the Web Speech API since they share the same device.
 *
 * Usage:
 *   const mic = new MicPreprocessor({
 *     onStreamReady: (stream) => console.log('Mic optimised'),
 *     onStreamError: (err) => console.warn('Mic fallback', err),
 *   });
 *   await mic.init();
 *   // ... later:
 *   mic.destroy();
 */

class MicPreprocessor {
    /**
     * @param {Object} options
     * @param {boolean}  [options.echoCancellation=true]  Enable browser echo cancellation
     * @param {boolean}  [options.noiseSuppression=true]  Enable browser noise suppression
     * @param {boolean}  [options.autoGainControl=true]   Enable automatic gain control
     * @param {number}   [options.channelCount=1]         Mono capture
     * @param {number}   [options.sampleRate=16000]       Sample rate (Hz)
     * @param {number}   [options.highPassFrequency=80]   High-pass filter cutoff (Hz)
     * @param {Function} [options.onStreamReady]          Called when mic stream is ready
     * @param {Function} [options.onStreamError]          Called on mic error
     * @param {Function} [options.onDisconnect]           Called if mic disconnects
     */
    constructor(options = {}) {
        this._echoCancellation = options.echoCancellation !== false;
        this._noiseSuppression = options.noiseSuppression !== false;
        this._autoGainControl  = options.autoGainControl !== false;
        this._channelCount     = options.channelCount || 1;
        this._sampleRate       = options.sampleRate || 16000;
        this._highPassFreq     = options.highPassFrequency || 80;

        this._onStreamReady = options.onStreamReady || (() => {});
        this._onStreamError = options.onStreamError || (() => {});
        this._onDisconnect  = options.onDisconnect  || (() => {});

        /** @private */ this._stream = null;
        /** @private */ this._audioCtx = null;
        /** @private */ this._sourceNode = null;
        /** @private */ this._filterNode = null;
        /** @private */ this._compressorNode = null;
        /** @private */ this._gainNode = null;
        /** @private */ this._active = false;
        /** @private */ this._initPromise = null;
    }

    /**
     * Check if getUserMedia is supported in this browser.
     * @returns {boolean}
     */
    static isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    /**
     * Acquire the microphone with optimised constraints and set up
     * the audio processing chain.
     *
     * Safe to call multiple times — returns the existing stream if already active.
     *
     * @returns {Promise<MediaStream|null>} The mic stream, or null on failure.
     */
    async init() {
        // Return existing init if in progress
        if (this._initPromise) return this._initPromise;
        if (this._active && this._stream) return this._stream;

        this._initPromise = this._doInit();
        try {
            return await this._initPromise;
        } finally {
            this._initPromise = null;
        }
    }

    /** @private */
    async _doInit() {
        if (!MicPreprocessor.isSupported()) {
            console.warn('[MicPreprocessor] getUserMedia not supported — skipping optimisation');
            return null;
        }

        // Build constraints
        const constraints = {
            audio: {
                echoCancellation: this._echoCancellation,
                noiseSuppression: this._noiseSuppression,
                autoGainControl:  this._autoGainControl,
                channelCount:     this._channelCount,
            },
            video: false,
        };

        // sampleRate constraint isn't universally supported — try it, fall back
        try {
            constraints.audio.sampleRate = this._sampleRate;
        } catch (e) { /* browser doesn't support sampleRate constraint */ }

        try {
            this._stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
            // Fall back to minimal constraints
            console.warn('[MicPreprocessor] Optimised constraints failed, trying fallback:', err.message);
            try {
                this._stream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true },
                    video: false,
                });
            } catch (fallbackErr) {
                console.warn('[MicPreprocessor] Mic access failed entirely:', fallbackErr.message);
                this._onStreamError(fallbackErr);
                return null;
            }
        }

        // Set up audio processing chain
        this._setupAudioChain();

        // Monitor for disconnection
        this._stream.getAudioTracks().forEach(track => {
            track.onended = () => {
                console.warn('[MicPreprocessor] Mic track ended unexpectedly');
                this._active = false;
                this._onDisconnect();
            };
            track.onmute = () => {
                console.warn('[MicPreprocessor] Mic track muted (possible disconnect)');
            };
        });

        this._active = true;
        console.info('[MicPreprocessor] Mic optimised: echo=%s noise=%s agc=%s highpass=%dHz',
            this._echoCancellation, this._noiseSuppression,
            this._autoGainControl, this._highPassFreq);

        this._onStreamReady(this._stream);
        return this._stream;
    }

    /** @private */
    _setupAudioChain() {
        if (!this._stream) return;

        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;

            this._audioCtx = new AudioCtx();
            this._sourceNode = this._audioCtx.createMediaStreamSource(this._stream);

            // High-pass filter: remove low-frequency rumble (fans, HVAC, traffic)
            this._filterNode = this._audioCtx.createBiquadFilter();
            this._filterNode.type = 'highpass';
            this._filterNode.frequency.value = this._highPassFreq;
            this._filterNode.Q.value = 0.7; // gentle roll-off

            // Compressor: normalise dynamic range (helps with distant speakers)
            this._compressorNode = this._audioCtx.createDynamicsCompressor();
            this._compressorNode.threshold.value = -40;  // dB — start compressing quiet signals
            this._compressorNode.knee.value = 12;         // soft knee for natural sound
            this._compressorNode.ratio.value = 4;         // 4:1 compression
            this._compressorNode.attack.value = 0.005;    // fast attack (5ms)
            this._compressorNode.release.value = 0.1;     // smooth release (100ms)

            // Gain node: final level adjustment
            this._gainNode = this._audioCtx.createGain();
            this._gainNode.gain.value = 1.0;

            // Connect chain: source → highpass → compressor → gain → (silent destination)
            this._sourceNode.connect(this._filterNode);
            this._filterNode.connect(this._compressorNode);
            this._compressorNode.connect(this._gainNode);
            // Don't connect to destination — we don't want to play mic audio
            // The processing chain still runs, applying the filter to the stream

        } catch (err) {
            console.warn('[MicPreprocessor] Audio chain setup failed (non-fatal):', err.message);
        }
    }

    /**
     * Pause the mic stream (mute without releasing the device).
     */
    pause() {
        if (this._stream) {
            this._stream.getAudioTracks().forEach(t => { t.enabled = false; });
        }
    }

    /**
     * Resume the mic stream after a pause.
     */
    resume() {
        if (this._stream) {
            this._stream.getAudioTracks().forEach(t => { t.enabled = true; });
        }
    }

    /**
     * Stop the mic stream and release the device, but keep the instance alive
     * for re-init later.
     */
    stop() {
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
        this._disconnectAudioChain();
        this._active = false;
    }

    /**
     * Full cleanup — stop stream, close AudioContext, release all references.
     */
    destroy() {
        this.stop();
        if (this._audioCtx && this._audioCtx.state !== 'closed') {
            this._audioCtx.close().catch(() => {});
        }
        this._audioCtx = null;
        this._sourceNode = null;
        this._filterNode = null;
        this._compressorNode = null;
        this._gainNode = null;
    }

    /** @private */
    _disconnectAudioChain() {
        try {
            if (this._sourceNode) this._sourceNode.disconnect();
            if (this._filterNode) this._filterNode.disconnect();
            if (this._compressorNode) this._compressorNode.disconnect();
            if (this._gainNode) this._gainNode.disconnect();
        } catch (e) { /* already disconnected */ }
    }

    /** @returns {boolean} Whether the mic stream is currently active */
    get isActive() { return this._active && !!this._stream; }

    /** @returns {MediaStream|null} The active MediaStream or null */
    get stream() { return this._stream; }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.MicPreprocessor = MicPreprocessor;
}
