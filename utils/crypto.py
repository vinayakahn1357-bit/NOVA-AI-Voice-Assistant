"""
utils/crypto.py — Field-Level Encryption for NOVA (Phase 6)
Provides encrypt_field/decrypt_field using Fernet symmetric encryption.
If ENCRYPTION_KEY is not set, operates in no-op mode (plaintext passthrough).
"""

import os
import base64
import hashlib

from utils.logger import get_logger

log = get_logger("crypto")

_fernet = None
_noop_mode = True


def _init_fernet():
    """Initialize Fernet cipher lazily from ENCRYPTION_KEY."""
    global _fernet, _noop_mode

    key = os.getenv("ENCRYPTION_KEY", "")
    if not key:
        _noop_mode = True
        return

    try:
        from cryptography.fernet import Fernet

        # Derive a valid 32-byte Fernet key from the user-provided key
        derived = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(derived)

        _fernet = Fernet(fernet_key)
        _noop_mode = False
        log.info("Encryption: enabled (Fernet)")
    except ImportError:
        log.warning("Encryption: 'cryptography' package not installed. Using plaintext.")
        _noop_mode = True
    except Exception as e:
        log.warning("Encryption: init failed (%s). Using plaintext.", e)
        _noop_mode = True


def encrypt_field(value: str) -> str:
    """
    Encrypt a string value.
    Returns the encrypted token, or the original value if encryption is disabled.
    """
    global _fernet
    if _fernet is None:
        _init_fernet()

    if _noop_mode or not value:
        return value

    try:
        return _fernet.encrypt(value.encode()).decode()
    except Exception as e:
        log.warning("Encryption failed: %s", e)
        return value


def decrypt_field(encrypted: str) -> str:
    """
    Decrypt an encrypted string.
    Returns the decrypted value, or the original string if decryption fails.
    """
    global _fernet
    if _fernet is None:
        _init_fernet()

    if _noop_mode or not encrypted:
        return encrypted

    try:
        return _fernet.decrypt(encrypted.encode()).decode()
    except Exception as e:
        # May be plaintext from before encryption was enabled
        log.debug("Decryption failed (may be plaintext): %s", e)
        return encrypted
