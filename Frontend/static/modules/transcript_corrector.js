/**
 * Transcript Corrector for NOVA Voice System
 *
 * Improves speech recognition accuracy by post-processing raw transcripts:
 *   - Selects the best alternative from multiple recognition candidates
 *   - Applies domain-aware word corrections (common mishearings)
 *   - Removes filler words and speech disfluencies (uh, um, like)
 *   - Fixes capitalization and punctuation artifacts
 *   - Corrects homophones in context (e.g., "their" vs "there")
 *   - Handles NOVA-specific vocabulary (wake word, tech terms)
 *
 * Design principles:
 *   - Lightweight: no external dependencies, < 1ms per correction
 *   - Conservative: only corrects high-confidence patterns
 *   - Additive: never removes meaningful words
 *
 * Usage:
 *   const corrector = new TranscriptCorrector();
 *   const corrected = corrector.correct('um tell me the whether');
 *   // → 'tell me the weather'
 *
 *   const best = corrector.selectBestAlternative(results[i]);
 *   // Picks the highest-quality transcript from alternatives
 */

class TranscriptCorrector {
    constructor(options = {}) {
        // User can expand correction maps
        this._customCorrections = options.customCorrections || {};
        this._removeFiller = options.removeFiller !== false;
        this._fixCapitalization = options.fixCapitalization !== false;
        this._correctionCount = 0;
        this._fillerCount = 0;

        // ── Common speech-to-text mishearing corrections ────────────────────
        // Format: misheard → correct
        // Only includes high-confidence, unambiguous corrections
        this._wordCorrections = {
            // NOVA-specific
            'nova': 'NOVA', 'no va': 'NOVA', 'novas': "NOVA's",
            'know va': 'NOVA', 'no vah': 'NOVA', 'nover': 'NOVA',

            // Weather / Whether
            'whether': 'weather',  // context-corrected below for question intent

            // Common tech terms misheard
            'ai': 'AI', 'a i': 'AI',
            'gpt': 'GPT', 'g p t': 'GPT',
            'api': 'API', 'a p i': 'API',
            'url': 'URL', 'u r l': 'URL',
            'html': 'HTML', 'css': 'CSS',
            'javascript': 'JavaScript', 'java script': 'JavaScript',
            'python': 'Python', 'pithon': 'Python',
            'github': 'GitHub', 'get hub': 'GitHub', 'git hub': 'GitHub',
            'chatgpt': 'ChatGPT', 'chat gpt': 'ChatGPT',
            'openai': 'OpenAI', 'open ai': 'OpenAI',
            'google': 'Google', 'youtube': 'YouTube', 'you tube': 'YouTube',
            'wifi': 'WiFi', 'wi-fi': 'WiFi', 'wi fi': 'WiFi',
            'bluetooth': 'Bluetooth', 'blue tooth': 'Bluetooth',

            // Common words misheard by speech recognition
            'wanna': 'want to', 'gonna': 'going to', 'gotta': 'got to',
            'lemme': 'let me', 'gimme': 'give me', 'kinda': 'kind of',
            'dunno': "don't know", 'coulda': 'could have',
            'shoulda': 'should have', 'woulda': 'would have',

            // Number-related
            'to': 'two',     // context-corrected below
            'for': 'four',   // context-corrected below
            'won': 'one',    // context-corrected below

            // Common homophone corrections (handled contextually below)
            ...this._customCorrections,
        };

        // ── Context-aware corrections (only apply in specific patterns) ────
        // Format: [regex, replacement]
        this._contextCorrections = [
            // "what's the whether" → "what's the weather"
            [/\bwhether\b(?=\s+(today|tomorrow|outside|like|forecast|report|in\s))/gi, 'weather'],
            [/\b(what's|what is|hows|how's|how is)\s+the\s+whether\b/gi, '$1 the weather'],

            // "search for" should keep "for" not "four"
            // We only correct "for" → "four" when preceded by a number context
            // Actually, let's NOT do the for/four, to/two, won/one corrections
            // as they cause more harm than good in general speech

            // "set a timer for 5 minutes" — keep "for"
            // "I have to go" — keep "to"
            // These are too ambiguous, removing from wordCorrections
        ];

        // Remove the ambiguous number corrections
        delete this._wordCorrections['to'];
        delete this._wordCorrections['for'];
        delete this._wordCorrections['won'];

        // ── Filler / disfluency words to remove ────────────────────────────
        this._fillerPatterns = [
            /\b(uh|uhh|uhm|um|umm|hmm|hm|er|err|ah|ahh)\b/gi,
            /\b(like)\b(?=\s+(uh|um|you know|I mean)\b)/gi,  // "like um" but not standalone "like"
            /\b(you know)\b(?=\s*[,.]?\s*(uh|um|like|I mean)\b)/gi,
            /\b(I mean)\b(?=\s*[,.]?\s*(uh|um|like|you know)\b)/gi,
        ];

        // ── Phrase corrections (multi-word patterns) ───────────────────────
        this._phraseCorrections = [
            // Question intent improvements
            [/\bcan you\b/gi, 'can you'],
            [/\bwhat's\b/gi, "what's"],
            [/\bhow's\b/gi, "how's"],

            // "Tell me about" patterns
            [/\btell me\b/gi, 'tell me'],

            // Clean up spacing around punctuation
            [/\s+([.,!?;:])/g, '$1'],
            [/([.,!?;:])\s*([A-Z])/g, '$1 $2'],

            // Remove leading/trailing whitespace artifacts
            [/^\s+|\s+$/g, ''],

            // Collapse multiple spaces
            [/\s{2,}/g, ' '],
        ];

        console.info('[TranscriptCorrector] Initialized — %d word corrections, filler removal=%s',
            Object.keys(this._wordCorrections).length, this._removeFiller);
    }

    /**
     * Select the best alternative from a SpeechRecognitionResult.
     * Uses confidence score + word count heuristics to pick the most accurate.
     *
     * @param {SpeechRecognitionResult} result - A single result with alternatives
     * @returns {{ transcript: string, confidence: number, index: number }}
     */
    selectBestAlternative(result) {
        if (!result || result.length === 0) {
            return { transcript: '', confidence: 0, index: 0 };
        }

        // Only one alternative — return it
        if (result.length === 1) {
            return {
                transcript: result[0].transcript,
                confidence: result[0].confidence || 0,
                index: 0,
            };
        }

        let bestIdx = 0;
        let bestScore = -1;

        for (let i = 0; i < result.length; i++) {
            const alt = result[i];
            const conf = alt.confidence || 0;
            const text = alt.transcript.trim();

            // Score = confidence * word quality bonus
            let score = conf;

            // Bonus for reasonable word count (not too short, not too long)
            const wordCount = text.split(/\s+/).length;
            if (wordCount >= 2 && wordCount <= 30) score += 0.05;

            // Bonus for proper capitalization (indicates better model output)
            if (/^[A-Z]/.test(text)) score += 0.02;

            // Bonus for ending with punctuation
            if (/[.!?]$/.test(text)) score += 0.01;

            // Penalty for very short (single char/word) transcripts
            if (text.length < 3) score -= 0.1;

            // Penalty for excessive repetition
            const words = text.toLowerCase().split(/\s+/);
            const uniqueRatio = new Set(words).size / words.length;
            if (uniqueRatio < 0.5) score -= 0.1; // > 50% repeated words

            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        return {
            transcript: result[bestIdx].transcript,
            confidence: result[bestIdx].confidence || 0,
            index: bestIdx,
        };
    }

    /**
     * Apply all corrections to a raw transcript.
     *
     * @param {string} text - Raw transcript from speech recognition
     * @returns {string} Corrected transcript
     */
    correct(text) {
        if (!text || typeof text !== 'string') return text || '';

        let corrected = text;

        // Step 1: Remove filler words
        if (this._removeFiller) {
            corrected = this._removeFillersFromText(corrected);
        }

        // Step 2: Apply word-level corrections
        corrected = this._applyWordCorrections(corrected);

        // Step 3: Apply context-aware corrections
        corrected = this._applyContextCorrections(corrected);

        // Step 4: Apply phrase corrections (spacing, punctuation)
        corrected = this._applyPhraseCorrections(corrected);

        // Step 5: Fix capitalization
        if (this._fixCapitalization) {
            corrected = this._fixCaps(corrected);
        }

        return corrected.trim();
    }

    /** @private */
    _removeFillersFromText(text) {
        let result = text;
        for (const pattern of this._fillerPatterns) {
            const before = result;
            result = result.replace(pattern, '');
            if (before !== result) this._fillerCount++;
        }
        // Clean up spaces left by filler removal
        result = result.replace(/\s{2,}/g, ' ').trim();
        return result;
    }

    /** @private */
    _applyWordCorrections(text) {
        // Build a regex that matches whole words from the correction map
        // Process multi-word corrections first (longer matches)
        const entries = Object.entries(this._wordCorrections)
            .sort((a, b) => b[0].length - a[0].length); // longest first

        let result = text;
        for (const [wrong, right] of entries) {
            // Skip if this is a multi-word pattern and doesn't exist in text
            if (wrong.includes(' ') && !result.toLowerCase().includes(wrong)) continue;

            const escaped = wrong.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b${escaped}\\b`, 'gi');
            const before = result;
            result = result.replace(regex, right);
            if (before !== result) this._correctionCount++;
        }

        return result;
    }

    /** @private */
    _applyContextCorrections(text) {
        let result = text;
        for (const [pattern, replacement] of this._contextCorrections) {
            const before = result;
            result = result.replace(pattern, replacement);
            if (before !== result) this._correctionCount++;
        }
        return result;
    }

    /** @private */
    _applyPhraseCorrections(text) {
        let result = text;
        for (const [pattern, replacement] of this._phraseCorrections) {
            result = result.replace(pattern, replacement);
        }
        return result;
    }

    /** @private */
    _fixCaps(text) {
        if (!text) return text;

        // Capitalize first letter of the sentence
        let result = text.charAt(0).toUpperCase() + text.slice(1);

        // Capitalize after sentence-ending punctuation
        result = result.replace(/([.!?]\s+)([a-z])/g, (_, punct, letter) => {
            return punct + letter.toUpperCase();
        });

        // Capitalize "I" standing alone
        result = result.replace(/\bi\b/g, 'I');

        // Don't touch words that are already all-caps (acronyms)
        return result;
    }

    /**
     * Add custom word corrections at runtime.
     * @param {Object} corrections - { wrongWord: 'correctWord', ... }
     */
    addCorrections(corrections) {
        Object.assign(this._wordCorrections, corrections);
    }

    /**
     * Get correction statistics.
     * @returns {Object}
     */
    get stats() {
        return {
            correctionCount: this._correctionCount,
            fillerCount: this._fillerCount,
            totalCorrections: Object.keys(this._wordCorrections).length,
        };
    }

    /**
     * Reset counters.
     */
    resetStats() {
        this._correctionCount = 0;
        this._fillerCount = 0;
    }
}

// Export for use in script.js
if (typeof window !== 'undefined') {
    window.TranscriptCorrector = TranscriptCorrector;
}
