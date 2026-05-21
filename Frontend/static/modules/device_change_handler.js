/**
 * Device Change Handler for NOVA Voice System
 *
 * Detects microphone device changes (Bluetooth headset plug/unplug,
 * USB mic swap, audio device switching) and automatically reinitializes
 * the audio pipeline to prevent frozen recognition states.
 *
 * Handles:
 *   - Bluetooth reconnect cycles (rapid connect/disconnect/connect)
 *   - USB mic hot-swap
 *   - System audio routing changes
 *   - Browser losing mic permissions after device change
 *
 * Usage:
 *   const handler = new DeviceChangeHandler({
 *     micPreprocessor: _novaMicPreprocessor,
 *     recognition: recognition,
 *     activityThreshold: _novaActivityThreshold,
 *     onDeviceChanged:   (info) => showToast('Mic changed: ' + info.label),
 *     onRecoverySuccess: ()     => console.log('Mic recovered'),
 *     onRecoveryFailed:  (err)  => showToast('Mic recovery failed', 'error'),
 *   });
 *   // ... later:
 *   handler.destroy();
 */

class DeviceChangeHandler {
    /**
     * @param {Object} options
     * @param {Object}   [options.micPreprocessor]        MicPreprocessor instance
     * @param {Object}   [options.recognition]            SpeechRecognition instance
     * @param {Object}   [options.activityThreshold]      SpeechActivityThreshold instance
     * @param {number}   [options.debounceMs=500]         Debounce interval for device changes (ms)
     * @param {number}   [options.maxRetries=3]           Max recovery retries
     * @param {number}   [options.retryDelayMs=1000]      Delay between retries (ms)
     * @param {Function} [options.onDeviceChanged]        Called when device list changes
     * @param {Function} [options.onRecoveryStart]        Called when recovery begins
     * @param {Function} [options.onRecoverySuccess]      Called on successful recovery
     * @param {Function} [options.onRecoveryFailed]       Called when recovery fails
     * @param {Function} [options.onDeviceAdded]          Called when a new audio input device appears
     * @param {Function} [options.onDeviceRemoved]        Called when an audio input device disappears
     */
    constructor(options = {}) {
        this._micPreprocessor   = options.micPreprocessor || null;
        this._recognition       = options.recognition || null;
        this._activityThreshold = options.activityThreshold || null;
        this._debounceMs        = options.debounceMs || 500;
        this._maxRetries        = options.maxRetries || 3;
        this._retryDelayMs      = options.retryDelayMs || 1000;

        this._onDeviceChanged   = options.onDeviceChanged || (() => {});
        this._onRecoveryStart   = options.onRecoveryStart || (() => {});
        this._onRecoverySuccess = options.onRecoverySuccess || (() => {});
        this._onRecoveryFailed  = options.onRecoveryFailed || (() => {});
        this._onDeviceAdded     = options.onDeviceAdded || (() => {});
        this._onDeviceRemoved   = options.onDeviceRemoved || (() => {});

        /** @private */ this._knownDevices = new Map(); // deviceId → {label, groupId}
        /** @private */ this._debounceTimer = null;
        /** @private */ this._recovering = false;
        /** @private */ this._active = false;
        /** @private */ this._boundHandler = null;
        /** @private */ this._wasListening = false; // set by script.js via setListeningState()

        // Stats
        /** @private */ this._changeCount = 0;
        /** @private */ this._recoveryCount = 0;
        /** @private */ this._failureCount = 0;

        // Auto-initialize
        this._init();
    }

    /**
     * Check if the MediaDevices API is available.
     * @returns {boolean}
     */
    static isSupported() {
        return !!(navigator.mediaDevices && typeof navigator.mediaDevices.addEventListener === 'function');
    }

    /**
     * Called by script.js to track whether recognition is currently listening.
     * @param {boolean} listening
     */
    setListeningState(listening) {
        this._wasListening = !!listening;
    }

    /** @private */
    async _init() {
        if (!DeviceChangeHandler.isSupported()) {
            console.warn('[DeviceChangeHandler] MediaDevices API not available — skipping');
            return;
        }

        // Snapshot initial device list
        await this._snapshotDevices();

        // Listen for device changes
        this._boundHandler = () => this._onDeviceChangeEvent();
        navigator.mediaDevices.addEventListener('devicechange', this._boundHandler);
        this._active = true;

        console.info('[DeviceChangeHandler] Monitoring %d audio input device(s)',
            this._knownDevices.size);
    }

    /** @private */
    async _snapshotDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this._knownDevices.clear();
            for (const d of devices) {
                if (d.kind === 'audioinput') {
                    this._knownDevices.set(d.deviceId, {
                        label: d.label || `Microphone (${d.deviceId.slice(0, 8)})`,
                        groupId: d.groupId,
                    });
                }
            }
        } catch (err) {
            console.warn('[DeviceChangeHandler] Could not enumerate devices:', err.message);
        }
    }

    /** @private */
    _onDeviceChangeEvent() {
        // Debounce — Bluetooth devices often fire rapid connect/disconnect/connect
        if (this._debounceTimer) clearTimeout(this._debounceTimer);
        this._debounceTimer = setTimeout(() => this._handleDeviceChange(), this._debounceMs);
    }

    /** @private */
    async _handleDeviceChange() {
        this._changeCount++;
        const previousDevices = new Map(this._knownDevices);

        // Re-snapshot
        await this._snapshotDevices();

        // Diff: find added and removed
        const added = [];
        const removed = [];

        for (const [id, info] of this._knownDevices) {
            if (!previousDevices.has(id)) {
                added.push({ id, ...info });
            }
        }
        for (const [id, info] of previousDevices) {
            if (!this._knownDevices.has(id)) {
                removed.push({ id, ...info });
            }
        }

        const changeInfo = {
            added,
            removed,
            totalDevices: this._knownDevices.size,
            timestamp: Date.now(),
        };

        console.info('[DeviceChangeHandler] Device change: +%d -%d (total: %d)',
            added.length, removed.length, this._knownDevices.size);

        // Notify callbacks
        if (added.length > 0) {
            for (const d of added) this._onDeviceAdded(d);
        }
        if (removed.length > 0) {
            for (const d of removed) this._onDeviceRemoved(d);
        }
        this._onDeviceChanged(changeInfo);

        // If a device was removed, check if the active mic needs recovery
        if (removed.length > 0) {
            await this._attemptRecovery(changeInfo);
        } else if (added.length > 0) {
            // New device added — optionally re-init to prefer it (if mic preprocessor is active)
            // But only if we're not currently in a recovery cycle
            if (!this._recovering && this._micPreprocessor && this._micPreprocessor.isActive) {
                console.info('[DeviceChangeHandler] New device detected — mic is still active, no action needed');
            }
        }
    }

    /** @private */
    async _attemptRecovery(changeInfo) {
        if (this._recovering) {
            console.warn('[DeviceChangeHandler] Recovery already in progress — skipping');
            return;
        }

        this._recovering = true;
        this._onRecoveryStart();

        // Use tracked listening state (set by script.js via setListeningState)
        const wasListening = this._wasListening;

        let retryCount = 0;
        let success = false;

        while (retryCount < this._maxRetries && !success) {
            retryCount++;
            console.info('[DeviceChangeHandler] Recovery attempt %d/%d…', retryCount, this._maxRetries);

            try {
                // Step 1: Stop current recognition
                if (this._recognition) {
                    try { this._recognition.stop(); } catch (e) { /* already stopped */ }
                }

                // Step 2: Reinitialize mic preprocessor
                if (this._micPreprocessor) {
                    this._micPreprocessor.stop();
                    // Brief delay for browser to release the device
                    await this._delay(300);
                    const stream = await this._micPreprocessor.init();

                    if (stream) {
                        // Step 3: Re-bind activity threshold to new stream
                        if (this._activityThreshold) {
                            this._activityThreshold.detach();
                            this._activityThreshold.attachStream(stream);
                            this._activityThreshold.recalibrate();
                        }

                        success = true;
                    } else {
                        throw new Error('MicPreprocessor.init() returned null');
                    }
                } else {
                    // No mic preprocessor — just ensure recognition can restart
                    success = true;
                }

                // Step 4: Restart recognition if it was previously listening
                if (success && wasListening && this._recognition) {
                    await this._delay(200);
                    try {
                        this._recognition.start();
                    } catch (e) {
                        console.warn('[DeviceChangeHandler] Could not restart recognition:', e.message);
                        // Not fatal — user can tap mic manually
                    }
                }

            } catch (err) {
                console.warn('[DeviceChangeHandler] Recovery attempt %d failed:', retryCount, err.message);
                if (retryCount < this._maxRetries) {
                    await this._delay(this._retryDelayMs * retryCount); // exponential backoff
                }
            }
        }

        this._recovering = false;

        if (success) {
            this._recoveryCount++;
            console.info('[DeviceChangeHandler] Recovery successful');
            this._onRecoverySuccess();
        } else {
            this._failureCount++;
            console.error('[DeviceChangeHandler] Recovery failed after %d attempts', this._maxRetries);
            this._onRecoveryFailed(new Error('Mic recovery failed after ' + this._maxRetries + ' attempts'));
        }
    }

    /** @private */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Update references to external objects (if they're created after this handler).
     * @param {Object} refs - { micPreprocessor, recognition, activityThreshold }
     */
    updateRefs(refs) {
        if (refs.micPreprocessor !== undefined) this._micPreprocessor = refs.micPreprocessor;
        if (refs.recognition !== undefined) this._recognition = refs.recognition;
        if (refs.activityThreshold !== undefined) this._activityThreshold = refs.activityThreshold;
    }

    /**
     * Get the current list of known audio input devices.
     * @returns {Array<{id: string, label: string}>}
     */
    getDevices() {
        return Array.from(this._knownDevices.entries()).map(([id, info]) => ({
            id,
            label: info.label,
        }));
    }

    /**
     * Get the currently active mic device label (best effort).
     * @returns {string}
     */
    getActiveDeviceLabel() {
        if (this._micPreprocessor && this._micPreprocessor.stream) {
            const tracks = this._micPreprocessor.stream.getAudioTracks();
            if (tracks.length > 0) {
                return tracks[0].label || 'Unknown Microphone';
            }
        }
        return 'No active device';
    }

    /**
     * Get handler statistics.
     * @returns {Object}
     */
    get stats() {
        return {
            changeCount: this._changeCount,
            recoveryCount: this._recoveryCount,
            failureCount: this._failureCount,
            knownDevices: this._knownDevices.size,
            activeDevice: this.getActiveDeviceLabel(),
            recovering: this._recovering,
        };
    }

    /** @returns {boolean} Whether the handler is actively monitoring */
    get isActive() { return this._active; }

    /**
     * Stop monitoring and release resources.
     */
    destroy() {
        if (this._boundHandler) {
            navigator.mediaDevices.removeEventListener('devicechange', this._boundHandler);
            this._boundHandler = null;
        }
        if (this._debounceTimer) {
            clearTimeout(this._debounceTimer);
            this._debounceTimer = null;
        }
        this._active = false;
        this._knownDevices.clear();
    }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.DeviceChangeHandler = DeviceChangeHandler;
}
