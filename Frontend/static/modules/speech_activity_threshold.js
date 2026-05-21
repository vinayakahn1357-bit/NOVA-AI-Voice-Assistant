/**
 * Speech Activity Threshold for NOVA Voice System
 *
 * Uses the Web Audio API AnalyserNode to measure real-time audio energy (RMS)
 * from the microphone stream. Provides a speech/silence gate that filters out:
 *   - Fan / HVAC / ambient room hum
 *   - Keyboard and mouse clicks
 *   - Weak background voices (conversations nearby)
 *   - Low-energy noise artifacts
 *
 * The threshold auto-calibrates the ambient noise floor during the first
 * few seconds of each session, then applies a dynamic gate above it.
 *
 * Usage:
 *   const sat = new SpeechActivityThreshold({
 *     onSpeechStart: () => console.log('Speech detected'),
 *     onSilence:     () => console.log('Silence'),
 *   });
 *   sat.attachStream(micStream);
 *   // ...
 *   if (sat.isSpeechActive()) { // process voice }
 *   sat.destroy();
 */

class SpeechActivityThreshold {
    /**
     * @param {Object} options
     * @param {number}   [options.threshold=0.015]          RMS energy threshold (0.0–1.0)
     * @param {number}   [options.holdOpenMs=200]            Keep gate open for this long after last speech (ms)
     * @param {number}   [options.calibrationDurationMs=2000] Auto-calibration window (ms)
     * @param {number}   [options.calibrationMargin=1.8]    Multiplier above noise floor for threshold
     * @param {number}   [options.pollIntervalMs=80]        How often to check energy (ms)
     * @param {number}   [options.fftSize=512]              AnalyserNode FFT size (256–2048)
     * @param {Function} [options.onSpeechStart]            Called when speech energy starts
     * @param {Function} [options.onSilence]                Called when silence resumes
     * @param {Function} [options.onEnergyUpdate]           Called each poll with (energy, threshold, isSpeech)
     */
    constructor(options = {}) {
        this._threshold       = options.threshold || 0.015;
        this._holdOpenMs      = options.holdOpenMs || 200;
        this._calibDuration   = options.calibrationDurationMs || 2000;
        this._calibMargin     = options.calibrationMargin || 1.8;
        this._pollInterval    = options.pollIntervalMs || 80;
        this._fftSize         = options.fftSize || 512;

        this._onSpeechStart   = options.onSpeechStart || (() => {});
        this._onSilence       = options.onSilence || (() => {});
        this._onEnergyUpdate  = options.onEnergyUpdate || (() => {});

        /** @private */ this._audioCtx = null;
        /** @private */ this._analyser = null;
        /** @private */ this._sourceNode = null;
        /** @private */ this._timeDomainData = null;
        /** @private */ this._pollTimer = null;

        /** @private */ this._speechActive = false;
        /** @private */ this._lastSpeechTime = 0;
        /** @private */ this._currentEnergy = 0;

        // Calibration state
        /** @private */ this._calibrating = false;
        /** @private */ this._calibSamples = [];
        /** @private */ this._calibStartTime = 0;
        /** @private */ this._calibrated = false;

        // Stats
        /** @private */ this._rejectedCount = 0;
        /** @private */ this._acceptedCount = 0;
    }

    /**
     * Check if the Web Audio API is available.
     * @returns {boolean}
     */
    static isSupported() {
        return !!(window.AudioContext || window.webkitAudioContext);
    }

    /**
     * Attach to a MediaStream and begin energy monitoring.
     * Safe to call multiple times — detaches previous stream first.
     *
     * @param {MediaStream} stream - The mic MediaStream to monitor
     */
    attachStream(stream) {
        if (!stream || !SpeechActivityThreshold.isSupported()) return;

        // Detach any existing stream
        this.detach();

        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this._audioCtx = new AudioCtx();

            this._analyser = this._audioCtx.createAnalyser();
            this._analyser.fftSize = this._fftSize;
            this._analyser.smoothingTimeConstant = 0.3; // responsive but not jittery

            this._sourceNode = this._audioCtx.createMediaStreamSource(stream);
            this._sourceNode.connect(this._analyser);
            // Don't connect to destination — we only analyse, never play

            this._timeDomainData = new Uint8Array(this._analyser.fftSize);

            // Start auto-calibration
            this._startCalibration();

            // Begin polling
            this._pollTimer = setInterval(() => this._poll(), this._pollInterval);

            console.info('[SpeechActivityThreshold] Attached — threshold=%.4f, fft=%d',
                this._threshold, this._fftSize);

        } catch (err) {
            console.warn('[SpeechActivityThreshold] Failed to attach (non-fatal):', err.message);
        }
    }

    /**
     * Detach from the current stream and stop monitoring.
     */
    detach() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
        try {
            if (this._sourceNode) this._sourceNode.disconnect();
        } catch (e) { /* already disconnected */ }
        if (this._audioCtx && this._audioCtx.state !== 'closed') {
            this._audioCtx.close().catch(() => {});
        }
        this._audioCtx = null;
        this._analyser = null;
        this._sourceNode = null;
        this._timeDomainData = null;
        this._speechActive = false;
        this._calibrating = false;
        this._calibrated = false;
        this._calibSamples = [];
    }

    /**
     * Full cleanup — detach and release all references.
     */
    destroy() {
        this.detach();
    }

    /** @private */
    _startCalibration() {
        this._calibrating = true;
        this._calibrated = false;
        this._calibSamples = [];
        this._calibStartTime = Date.now();
        console.info('[SpeechActivityThreshold] Calibrating ambient noise floor (%dms)…',
            this._calibDuration);
    }

    /** @private */
    _poll() {
        if (!this._analyser || !this._timeDomainData) return;

        // Read time-domain data
        this._analyser.getByteTimeDomainData(this._timeDomainData);

        // Compute RMS energy (0.0 – 1.0)
        let sumSquares = 0;
        for (let i = 0; i < this._timeDomainData.length; i++) {
            const sample = (this._timeDomainData[i] - 128) / 128; // normalise to -1..1
            sumSquares += sample * sample;
        }
        this._currentEnergy = Math.sqrt(sumSquares / this._timeDomainData.length);

        // Calibration phase — collect ambient samples
        if (this._calibrating) {
            this._calibSamples.push(this._currentEnergy);
            if (Date.now() - this._calibStartTime >= this._calibDuration) {
                this._finishCalibration();
            }
        }

        // Determine speech state
        const now = Date.now();
        const aboveThreshold = this._currentEnergy >= this._threshold;

        if (aboveThreshold) {
            this._lastSpeechTime = now;
            if (!this._speechActive) {
                this._speechActive = true;
                this._onSpeechStart();
            }
        } else if (this._speechActive && (now - this._lastSpeechTime) > this._holdOpenMs) {
            // Hold-open timer expired — transition to silence
            this._speechActive = false;
            this._onSilence();
        }

        // Notify energy update
        this._onEnergyUpdate(this._currentEnergy, this._threshold, this._speechActive);
    }

    /** @private */
    _finishCalibration() {
        this._calibrating = false;
        this._calibrated = true;

        if (this._calibSamples.length === 0) return;

        // Compute average ambient noise floor
        const sum = this._calibSamples.reduce((a, b) => a + b, 0);
        const avgNoise = sum / this._calibSamples.length;

        // Set threshold above ambient noise floor (with margin)
        // But never below the minimum default
        const calibratedThreshold = avgNoise * this._calibMargin;
        const minThreshold = 0.008; // absolute minimum
        this._threshold = Math.max(minThreshold, Math.max(this._threshold, calibratedThreshold));

        console.info('[SpeechActivityThreshold] Calibrated: noise=%.4f → threshold=%.4f (%d samples)',
            avgNoise, this._threshold, this._calibSamples.length);

        this._calibSamples = []; // free memory
    }

    /**
     * Check if speech is currently active (energy above threshold).
     * @returns {boolean}
     */
    isSpeechActive() {
        return this._speechActive;
    }

    /**
     * Get the current audio energy level (0.0–1.0).
     * @returns {number}
     */
    getCurrentEnergy() {
        return this._currentEnergy;
    }

    /**
     * Get the current threshold value.
     * @returns {number}
     */
    getThreshold() {
        return this._threshold;
    }

    /**
     * Manually set the energy threshold.
     * @param {number} value - Threshold (0.0–1.0)
     */
    setThreshold(value) {
        this._threshold = Math.max(0, Math.min(1, value));
    }

    /**
     * Record a rejected result (for stats/debug).
     */
    recordRejection() {
        this._rejectedCount++;
    }

    /**
     * Record an accepted result (for stats/debug).
     */
    recordAcceptance() {
        this._acceptedCount++;
    }

    /**
     * Re-calibrate the noise floor (e.g., after device change).
     */
    recalibrate() {
        if (this._analyser) {
            this._startCalibration();
        }
    }

    /**
     * Get activity stats.
     * @returns {Object}
     */
    get stats() {
        return {
            currentEnergy: this._currentEnergy,
            threshold: this._threshold,
            speechActive: this._speechActive,
            calibrated: this._calibrated,
            rejectedCount: this._rejectedCount,
            acceptedCount: this._acceptedCount,
        };
    }

    /** @returns {boolean} Whether the analyser is currently attached and monitoring */
    get isAttached() {
        return !!this._analyser;
    }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.SpeechActivityThreshold = SpeechActivityThreshold;
}
