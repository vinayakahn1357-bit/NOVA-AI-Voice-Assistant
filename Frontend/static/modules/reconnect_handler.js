/**
 * Reconnect Handler for NOVA (Phase 14)
 * 
 * Manages automatic reconnection for SSE streams with:
 *   - Exponential backoff (1s, 2s, 4s, 8s, max 30s)
 *   - Max retry limit (5 attempts)
 *   - Connection state tracking
 *   - User notification on reconnect
 *   - Graceful degradation to polling on repeated failure
 *
 * Usage:
 *   const handler = new ReconnectHandler({
 *     maxRetries: 5,
 *     onReconnect: (attempt) => showToast("Reconnecting..."),
 *     onGiveUp: () => showError("Connection lost"),
 *   });
 *   handler.onDisconnect();  // called when SSE disconnects
 */

class ReconnectHandler {
    constructor(options = {}) {
        this.maxRetries = options.maxRetries || 5;
        this.baseDelay = options.baseDelay || 1000;     // 1 second
        this.maxDelay = options.maxDelay || 30000;       // 30 seconds
        this.onReconnect = options.onReconnect || (() => {});
        this.onGiveUp = options.onGiveUp || (() => {});
        this.onSuccess = options.onSuccess || (() => {});

        this._attempts = 0;
        this._timer = null;
        this._state = 'idle';  // idle, reconnecting, connected, failed
    }

    /**
     * Called when a stream disconnects unexpectedly.
     * Initiates the reconnection cycle.
     * @param {Function} reconnectFn - Function to call to attempt reconnection
     */
    onDisconnect(reconnectFn) {
        if (this._state === 'reconnecting') return;

        this._attempts = 0;
        this._state = 'reconnecting';
        this._scheduleReconnect(reconnectFn);
    }

    /**
     * Called when reconnection succeeds.
     */
    connected() {
        this._attempts = 0;
        this._state = 'connected';
        if (this._timer) {
            clearTimeout(this._timer);
            this._timer = null;
        }
        this.onSuccess();
        console.info('[ReconnectHandler] Connection established');
    }

    /**
     * Reset the handler state.
     */
    reset() {
        this._attempts = 0;
        this._state = 'idle';
        if (this._timer) {
            clearTimeout(this._timer);
            this._timer = null;
        }
    }

    /** @private */
    _scheduleReconnect(reconnectFn) {
        if (this._attempts >= this.maxRetries) {
            console.error('[ReconnectHandler] Max retries exceeded:', this.maxRetries);
            this._state = 'failed';
            this.onGiveUp();
            return;
        }

        this._attempts++;
        const delay = Math.min(
            this.baseDelay * Math.pow(2, this._attempts - 1),
            this.maxDelay
        );

        console.info(`[ReconnectHandler] Attempt ${this._attempts}/${this.maxRetries} in ${delay}ms`);
        this.onReconnect(this._attempts, delay);

        this._timer = setTimeout(() => {
            try {
                reconnectFn();
            } catch (e) {
                console.warn('[ReconnectHandler] Reconnect failed:', e);
                this._scheduleReconnect(reconnectFn);
            }
        }, delay);
    }

    /** Current connection state */
    get state() { return this._state; }

    /** Number of reconnection attempts made */
    get attempts() { return this._attempts; }
}

// Export
if (typeof window !== 'undefined') {
    window.ReconnectHandler = ReconnectHandler;
}
