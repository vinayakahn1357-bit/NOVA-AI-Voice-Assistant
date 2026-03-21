"""
utils/errors.py — Custom Exceptions & Global Error Handlers for NOVA
"""

from flask import jsonify


# ─── Custom Exceptions ───────────────────────────────────────────────────────

class NovaError(Exception):
    """Base exception for all NOVA errors."""
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str = "An internal error occurred.", code: str = None,
                 status_code: int = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class NovaValidationError(NovaError):
    """Raised when user input fails validation."""
    status_code = 400
    code = "VALIDATION_ERROR"


class NovaAuthError(NovaError):
    """Raised for authentication / authorisation failures."""
    status_code = 401
    code = "AUTH_ERROR"


class NovaProviderError(NovaError):
    """Raised when all AI providers fail."""
    status_code = 503
    code = "PROVIDER_ERROR"


class NovaRateLimitError(NovaError):
    """Raised when a client exceeds the rate limit."""
    status_code = 429
    code = "RATE_LIMIT"


# ─── Flask Error Handler Registration ────────────────────────────────────────

def register_error_handlers(app):
    """Register global JSON error handlers on the Flask app."""

    @app.errorhandler(NovaError)
    def handle_nova_error(exc):
        return jsonify({"error": exc.message, "code": exc.code}), exc.status_code

    @app.errorhandler(400)
    def bad_request(exc):
        return jsonify({"error": str(exc), "code": "BAD_REQUEST"}), 400

    @app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify({"error": "Method not allowed", "code": "METHOD_NOT_ALLOWED"}), 405

    @app.errorhandler(500)
    def internal_error(exc):
        return jsonify({"error": "Internal server error", "code": "INTERNAL_ERROR"}), 500
