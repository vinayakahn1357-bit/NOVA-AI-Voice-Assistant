/**
 * Stream Error Handler for NOVA (Phase 14)
 * 
 * Centralized error handling for all streaming scenarios:
 *   - Network errors
 *   - Parse errors (malformed SSE)
 *   - Provider errors (503, 429, etc.)
 *   - Timeout errors
 *   - Abort errors (user cancelled)
 * 
 * Provides user-friendly error messages and recovery actions.
 *
 * Usage:
 *   const errorHandler = new StreamErrorHandler({
 *     onError: (msg, severity) => showToast(msg, severity),
 *     onRetryable: (retryFn) => showRetryButton(retryFn),
 *   });
 *   errorHandler.handle(error, context);
 */

class StreamErrorHandler {
    constructor(options = {}) {
        this.onError = options.onError || ((msg) => console.error(msg));
        this.onRetryable = options.onRetryable || (() => {});
        this._errorLog = [];
        this._maxLog = 50;
    }

    /**
     * Handle a streaming error.
     * @param {Error|Object} error - The error to handle
     * @param {Object} context - Additional context (url, status, etc.)
     * @returns {Object} - { message, severity, retryable, action }
     */
    handle(error, context = {}) {
        const result = this._classify(error, context);

        // Log it
        this._errorLog.push({
            time: Date.now(),
            message: result.message,
            severity: result.severity,
            type: result.type,
        });
        if (this._errorLog.length > this._maxLog) {
            this._errorLog.shift();
        }

        // Notify
        this.onError(result.message, result.severity);

        // Offer retry if applicable
        if (result.retryable && context.retryFn) {
            this.onRetryable(context.retryFn);
        }

        console.warn('[StreamErrorHandler]', result.type, ':', result.message);
        return result;
    }

    /** @private */
    _classify(error, context) {
        const status = context.status || 0;
        const errorMsg = error?.message || String(error) || '';

        // User cancelled
        if (error?.name === 'AbortError' || errorMsg.includes('abort')) {
            return {
                type: 'abort',
                message: 'Request cancelled.',
                severity: 'info',
                retryable: false,
            };
        }

        // Network error
        if (error?.name === 'TypeError' && errorMsg.includes('fetch')
            || errorMsg.includes('NetworkError')
            || errorMsg.includes('Failed to fetch')) {
            return {
                type: 'network',
                message: 'Network connection lost. Check your internet and try again.',
                severity: 'error',
                retryable: true,
            };
        }

        // Rate limited
        if (status === 429) {
            return {
                type: 'rate_limit',
                message: 'Too many requests. Please wait a moment before trying again.',
                severity: 'warning',
                retryable: true,
            };
        }

        // Provider unavailable
        if (status === 503 || errorMsg.includes('unavailable')) {
            return {
                type: 'provider',
                message: 'AI service temporarily unavailable. Retrying...',
                severity: 'warning',
                retryable: true,
            };
        }

        // Server error
        if (status >= 500) {
            return {
                type: 'server',
                message: 'Server error. Please try again in a moment.',
                severity: 'error',
                retryable: true,
            };
        }

        // Auth error
        if (status === 401 || status === 403) {
            return {
                type: 'auth',
                message: 'Session expired. Please log in again.',
                severity: 'error',
                retryable: false,
            };
        }

        // Timeout
        if (errorMsg.includes('timeout') || errorMsg.includes('Timeout')) {
            return {
                type: 'timeout',
                message: 'The request timed out. Try a shorter question.',
                severity: 'warning',
                retryable: true,
            };
        }

        // Parse error (malformed SSE)
        if (errorMsg.includes('JSON') || errorMsg.includes('parse')) {
            return {
                type: 'parse',
                message: 'Received an invalid response. Retrying...',
                severity: 'warning',
                retryable: true,
            };
        }

        // Generic fallback
        return {
            type: 'unknown',
            message: 'Something went wrong. Please try again.',
            severity: 'error',
            retryable: true,
        };
    }

    /**
     * Get recent error log.
     * @param {number} count - Number of recent errors to return
     */
    getLog(count = 10) {
        return this._errorLog.slice(-count);
    }

    /**
     * Clear the error log.
     */
    clearLog() {
        this._errorLog = [];
    }

    /**
     * Get error statistics.
     */
    stats() {
        const last5min = Date.now() - 5 * 60 * 1000;
        const recent = this._errorLog.filter(e => e.time > last5min);
        const byType = {};
        for (const e of recent) {
            byType[e.type] = (byType[e.type] || 0) + 1;
        }
        return {
            total: this._errorLog.length,
            last5min: recent.length,
            byType,
        };
    }
}

// Export
if (typeof window !== 'undefined') {
    window.StreamErrorHandler = StreamErrorHandler;
}
