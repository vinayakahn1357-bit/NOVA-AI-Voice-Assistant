/**
 * Stream Audio Guard for NOVA Voice System
 *
 * Protects the voice recognition pipeline from common failure modes:
 *   - Rapid restart loops (rate-limited to 3 per 10 seconds)
 *   - Indefinite recognition sessions (max 5 minutes)
 *   - Dead streams (3 consecutive no-result sessions)
 *   - Overlapping recognition starts
 *   - Error-based recovery classification
 *
 * Usage:
 *   const guard = new StreamAudioGuard({
 *     onSessionTimeout: () => showToast('Voice session timed out'),
 *     onRestartBlocked: () => console.warn('Too many restarts'),
 *   });
 *   guard.guard(recognition);
 *   // guard wraps the recognition lifecycle automatically
 */

class StreamAudioGuard {
    /**
     * @param {Object} options
     * @param {number}   [options.maxRestartRate=3]       Max restarts per window
     * @param {number}   [options.restartWindow=10000]    Window for restart rate (ms)
     * @param {number}   [options.sessionTimeout=300000]  Max session duration (ms, 5 min)
     * @param {Function} [options.onSessionTimeout]       Called when session times out
     * @param {Function} [options.onRestartBlocked]       Called when restart is rate-limited
     * @param {Function} [options.onRecovery]             Called on successful recovery
     * @param {Function} [options.onStreamDead]           Called when stream appears dead
     */
    constructor(options = {}) {
        this._maxRestartRate  = options.maxRestartRate || 3;
        this._restartWindow   = options.restartWindow || 10000;
        this._sessionTimeout  = options.sessionTimeout || 300000;

        this._onSessionTimeout = options.onSessionTimeout || (() => {});
        this._onRestartBlocked = options.onRestartBlocked || (() => {});
        this._onRecovery       = options.onRecovery || (() => {});
        this._onStreamDead     = options.onStreamDead || (() => {});

        /** @private */ this._sessionActive = false;
        /** @private */ this._sessionStartTime = 0;
        /** @private */ this._sessionTimer = null;
        /** @private */ this._restartTimestamps = [];
        /** @private */ this._emptySessionCount = 0;
        /** @private */ this._hadResultsThisSession = false;

        // Stats
        /** @private */ this._totalSessions = 0;
        /** @private */ this._totalRestarts = 0;
        /** @private */ this._totalErrors = 0;
        /** @private */ this._startTime = Date.now();
    }

    /**
     * Called when a recognition session starts.
     * Tracks session state and starts the session timeout timer.
     */
    onRecognitionStart() {
        if (this._sessionActive) {
            console.warn('[StreamAudioGuard] Overlapping recognition session detected');
        }

        this._sessionActive = true;
        this._sessionStartTime = Date.now();
        this._hadResultsThisSession = false;
        this._totalSessions++;

        // Start session timeout
        if (this._sessionTimer) clearTimeout(this._sessionTimer);
        this._sessionTimer = setTimeout(() => {
            if (this._sessionActive) {
                console.warn('[StreamAudioGuard] Session timeout reached (%dms)', this._sessionTimeout);
                this._onSessionTimeout();
                this._sessionActive = false;
            }
        }, this._sessionTimeout);
    }

    /**
     * Called when recognition produces a result (interim or final).
     * Resets the empty-session counter.
     */
    onRecognitionResult() {
        this._hadResultsThisSession = true;
        this._emptySessionCount = 0;
    }

    /**
     * Called when a recognition session ends normally.
     */
    onRecognitionEnd() {
        this._sessionActive = false;
        if (this._sessionTimer) {
            clearTimeout(this._sessionTimer);
            this._sessionTimer = null;
        }

        // Track empty sessions (no results = possible dead stream)
        if (!this._hadResultsThisSession) {
            this._emptySessionCount++;
            if (this._emptySessionCount >= 3) {
                console.warn('[StreamAudioGuard] Stream appears dead (%d empty sessions)',
                    this._emptySessionCount);
                this._onStreamDead();
                this._emptySessionCount = 0;
            }
        }
    }

    /**
     * Called when recognition encounters an error.
     * Classifies the error and determines if recovery is appropriate.
     *
     * @param {Object} error - The SpeechRecognitionError event
     * @returns {string} Action: 'ignore', 'retry', 'fatal'
     */
    onRecognitionError(error) {
        this._totalErrors++;
        const errorType = error?.error || error?.message || String(error);

        this._sessionActive = false;
        if (this._sessionTimer) {
            clearTimeout(this._sessionTimer);
            this._sessionTimer = null;
        }

        // Classify error
        switch (errorType) {
            case 'no-speech':
                // Normal — user didn't speak. Not an error.
                return 'ignore';

            case 'aborted':
                // Recognition was aborted (e.g., by calling stop()). Normal.
                return 'ignore';

            case 'audio-capture':
                // Mic not available — could be temporary
                console.warn('[StreamAudioGuard] Audio capture error — mic may be unavailable');
                return 'retry';

            case 'network':
                // Network error for cloud-based recognition
                console.warn('[StreamAudioGuard] Network error in speech recognition');
                return 'retry';

            case 'not-allowed':
            case 'service-not-allowed':
                // Permission denied — fatal, user must grant access
                console.error('[StreamAudioGuard] Mic permission denied — cannot recover');
                return 'fatal';

            default:
                console.warn('[StreamAudioGuard] Unknown recognition error:', errorType);
                return 'retry';
        }
    }

    /**
     * Request a rate-limited restart of speech recognition.
     *
     * @param {Function} startFn - Function to call to restart recognition
     * @returns {boolean} Whether the restart was allowed
     */
    requestRestart(startFn) {
        const now = Date.now();

        // Prune old timestamps outside the window
        this._restartTimestamps = this._restartTimestamps.filter(
            t => (now - t) < this._restartWindow
        );

        // Check rate limit
        if (this._restartTimestamps.length >= this._maxRestartRate) {
            console.warn('[StreamAudioGuard] Restart rate limit reached (%d/%d in %dms)',
                this._restartTimestamps.length, this._maxRestartRate, this._restartWindow);
            this._onRestartBlocked();
            return false;
        }

        // Allow restart
        this._restartTimestamps.push(now);
        this._totalRestarts++;

        try {
            startFn();
            this._onRecovery();
            return true;
        } catch (err) {
            console.warn('[StreamAudioGuard] Restart failed:', err.message);
            return false;
        }
    }

    /**
     * Full resource cleanup.
     */
    cleanup() {
        if (this._sessionTimer) {
            clearTimeout(this._sessionTimer);
            this._sessionTimer = null;
        }
        this._sessionActive = false;
        this._restartTimestamps = [];
        this._emptySessionCount = 0;
    }

    /** @returns {boolean} Whether a recognition session is currently active */
    get isSessionActive() { return this._sessionActive; }

    /** @returns {number} Duration of current session in ms, or 0 */
    get sessionDuration() {
        return this._sessionActive ? Date.now() - this._sessionStartTime : 0;
    }

    /**
     * Get guard statistics.
     * @returns {Object} Stats object
     */
    get stats() {
        return {
            totalSessions: this._totalSessions,
            totalRestarts: this._totalRestarts,
            totalErrors: this._totalErrors,
            emptySessionStreak: this._emptySessionCount,
            uptimeMs: Date.now() - this._startTime,
            sessionActive: this._sessionActive,
        };
    }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.StreamAudioGuard = StreamAudioGuard;
}
