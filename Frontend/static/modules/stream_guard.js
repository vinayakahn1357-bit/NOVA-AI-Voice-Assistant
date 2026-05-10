/**
 * Frontend Stream Guard for NOVA (Phase 14)
 * 
 * Protects the SSE streaming connection from:
 *   - Stale/stuck streams
 *   - Memory leaks from accumulated tokens
 *   - Infinite retry loops
 *   - Orphaned EventSource connections
 * 
 * Usage:
 *   const guard = new StreamGuard({
 *     maxDuration: 90000,
 *     onTimeout: () => showError("Response timed out"),
 *     onCleanup: () => resetUI(),
 *   });
 *   guard.start(eventSource);
 *   // ... on complete:
 *   guard.stop();
 */

class StreamGuard {
    constructor(options = {}) {
        this.maxDuration = options.maxDuration || 90000;  // 90s default
        this.maxTokens = options.maxTokens || 50000;      // max chars
        this.heartbeatInterval = options.heartbeatInterval || 15000; // 15s
        this.onTimeout = options.onTimeout || (() => {});
        this.onCleanup = options.onCleanup || (() => {});
        this.onStall = options.onStall || (() => {});

        this._source = null;
        this._timer = null;
        this._heartbeat = null;
        this._lastActivity = 0;
        this._totalChars = 0;
        this._active = false;
        this._startTime = 0;
    }

    /**
     * Start guarding a stream.
     * @param {EventSource|AbortController} source - The stream to guard
     */
    start(source) {
        this.stop();  // clean up any previous stream

        this._source = source;
        this._active = true;
        this._startTime = Date.now();
        this._lastActivity = Date.now();
        this._totalChars = 0;

        // Timeout timer
        this._timer = setTimeout(() => {
            console.warn('[StreamGuard] Stream exceeded max duration:', this.maxDuration + 'ms');
            this.onTimeout();
            this.stop();
        }, this.maxDuration);

        // Heartbeat checker (detect stalled streams)
        this._heartbeat = setInterval(() => {
            const silenceMs = Date.now() - this._lastActivity;
            if (silenceMs > this.heartbeatInterval * 2) {
                console.warn('[StreamGuard] Stream stalled for', silenceMs + 'ms');
                this.onStall();
            }
        }, this.heartbeatInterval);
    }

    /**
     * Record activity (call on each token/chunk received).
     * @param {number} charCount - Characters in this chunk
     */
    activity(charCount = 0) {
        this._lastActivity = Date.now();
        this._totalChars += charCount;

        // Guard against excessive response size
        if (this._totalChars > this.maxTokens) {
            console.warn('[StreamGuard] Max token limit reached:', this._totalChars);
            this.stop();
        }
    }

    /**
     * Stop guarding and clean up all resources.
     */
    stop() {
        if (this._timer) {
            clearTimeout(this._timer);
            this._timer = null;
        }
        if (this._heartbeat) {
            clearInterval(this._heartbeat);
            this._heartbeat = null;
        }

        if (this._source) {
            // Close EventSource
            if (this._source.close) {
                try { this._source.close(); } catch (e) { /* ignore */ }
            }
            // Abort fetch
            if (this._source.abort) {
                try { this._source.abort(); } catch (e) { /* ignore */ }
            }
            this._source = null;
        }

        if (this._active) {
            this._active = false;
            const elapsed = Date.now() - this._startTime;
            console.debug('[StreamGuard] Stream ended:', elapsed + 'ms,', this._totalChars, 'chars');
            try { this.onCleanup(); } catch (e) { /* ignore */ }
        }
    }

    /** Is the stream currently active? */
    get isActive() { return this._active; }

    /** Elapsed time in ms since stream started */
    get elapsed() { return this._active ? Date.now() - this._startTime : 0; }

    /** Total characters received */
    get totalChars() { return this._totalChars; }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.StreamGuard = StreamGuard;
}
