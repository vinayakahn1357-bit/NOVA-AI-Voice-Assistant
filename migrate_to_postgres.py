"""
migrate_to_postgres.py — Optional SQLite → PostgreSQL Migration Script
Run manually: python migrate_to_postgres.py

Features:
- Reads all data from nova_memory.db and nova_users.json
- Inserts into PostgreSQL tables via SQLAlchemy ORM
- Duplicate handling: skips existing records (idempotent)
- Transaction-safe: rolls back entirely on failure
- Per-table progress logging
- Safe to run multiple times

Requirements:
- DATABASE_URL must point to a PostgreSQL instance
- PostgreSQL tables must already exist (run the app once to auto-create)
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DIR
from database import (
    get_db_session, is_postgres, init_db,
    DBUser, DBMemoryFact, DBUserInterest, DBUserPreference,
    DBDailySummary, DBMeta,
)


def _log(msg): print(f"[MIGRATE] {msg}")


def _migrate_users(session):
    """Migrate users from nova_users.json."""
    users_file = os.path.join(BASE_DIR, "nova_users.json")
    if not os.path.exists(users_file):
        _log("nova_users.json not found — skipping user migration")
        return

    with open(users_file, "r") as f:
        users = json.load(f)

    inserted = 0
    skipped = 0

    for user_data in users:
        user_id = user_data.get("id", "")
        email = user_data.get("email", "")

        if not email:
            skipped += 1
            continue

        exists = session.query(DBUser).filter_by(email=email).first()
        if exists:
            skipped += 1
            continue

        user = DBUser(
            id=user_id or email,
            email=email,
            name=user_data.get("name", ""),
            password_hash=user_data.get("password_hash", ""),
            salt=user_data.get("salt", ""),
            provider=user_data.get("provider", "email"),
            role=user_data.get("role", "user"),
            created_at=datetime.now(timezone.utc),
        )
        session.add(user)
        inserted += 1

    session.flush()
    _log(f"users: {inserted} inserted, {skipped} skipped (duplicate/invalid)")
    return inserted


def _migrate_memory(session, sqlite_conn, user_id="default"):
    """Migrate memory data from SQLite nova_memory.db."""

    # ── Facts ───────────────────────────────────────
    try:
        rows = sqlite_conn.execute("SELECT fact FROM facts").fetchall()
        inserted = 0
        skipped = 0
        for (fact,) in rows:
            exists = session.query(DBMemoryFact).filter_by(
                user_id=user_id, fact=fact
            ).first()
            if exists:
                skipped += 1
                continue
            session.add(DBMemoryFact(user_id=user_id, fact=fact))
            inserted += 1
        session.flush()
        _log(f"facts: {inserted} inserted, {skipped} skipped (duplicate)")
    except sqlite3.OperationalError as e:
        _log(f"facts: table not found ({e})")

    # ── Interests ───────────────────────────────────
    try:
        rows = sqlite_conn.execute("SELECT topic, count FROM interests").fetchall()
        inserted = 0
        skipped = 0
        for topic, count in rows:
            exists = session.query(DBUserInterest).filter_by(
                user_id=user_id, topic=topic
            ).first()
            if exists:
                skipped += 1
                continue
            session.add(DBUserInterest(user_id=user_id, topic=topic, count=count))
            inserted += 1
        session.flush()
        _log(f"interests: {inserted} inserted, {skipped} skipped (duplicate)")
    except sqlite3.OperationalError as e:
        _log(f"interests: table not found ({e})")

    # ── Preferences ─────────────────────────────────
    try:
        rows = sqlite_conn.execute("SELECT key, value FROM preferences").fetchall()
        inserted = 0
        skipped = 0
        for key, value in rows:
            exists = session.query(DBUserPreference).filter_by(
                user_id=user_id, key=key
            ).first()
            if exists:
                skipped += 1
                continue
            session.add(DBUserPreference(user_id=user_id, key=key, value=value))
            inserted += 1
        session.flush()
        _log(f"preferences: {inserted} inserted, {skipped} skipped (duplicate)")
    except sqlite3.OperationalError as e:
        _log(f"preferences: table not found ({e})")

    # ── Daily Summaries ─────────────────────────────
    try:
        rows = sqlite_conn.execute("SELECT day, summary FROM daily_summaries").fetchall()
        inserted = 0
        skipped = 0
        for day, summary in rows:
            exists = session.query(DBDailySummary).filter_by(
                user_id=user_id, day=day
            ).first()
            if exists:
                skipped += 1
                continue
            session.add(DBDailySummary(user_id=user_id, day=day, summary=summary))
            inserted += 1
        session.flush()
        _log(f"daily_summaries: {inserted} inserted, {skipped} skipped (duplicate)")
    except sqlite3.OperationalError as e:
        _log(f"daily_summaries: table not found ({e})")

    # ── Meta ────────────────────────────────────────
    try:
        rows = sqlite_conn.execute("SELECT key, value FROM meta").fetchall()
        inserted = 0
        skipped = 0
        for key, value in rows:
            exists = session.query(DBMeta).filter_by(
                user_id=user_id, key=key
            ).first()
            if exists:
                skipped += 1
                continue
            session.add(DBMeta(user_id=user_id, key=key, value=value))
            inserted += 1
        session.flush()
        _log(f"meta: {inserted} inserted, {skipped} skipped (duplicate)")
    except sqlite3.OperationalError as e:
        _log(f"meta: table not found ({e})")


def main():
    _log("═══════════════════════════════════════════════")
    _log("NOVA SQLite → PostgreSQL Migration")
    _log("═══════════════════════════════════════════════")

    # Verify PostgreSQL is configured
    if not is_postgres():
        _log("ERROR: DATABASE_URL must point to PostgreSQL.")
        _log("Current config uses SQLite — nothing to migrate to.")
        _log("Set DATABASE_URL=postgresql://user:pass@host:port/dbname in .env")
        sys.exit(1)

    # Ensure tables exist
    _log("Ensuring PostgreSQL tables exist...")
    init_db()

    # Open SQLite source
    sqlite_path = os.path.join(BASE_DIR, "nova_memory.db")
    if not os.path.exists(sqlite_path):
        _log(f"WARNING: {sqlite_path} not found — no memory data to migrate")
        sqlite_conn = None
    else:
        sqlite_conn = sqlite3.connect(sqlite_path)
        _log(f"Source SQLite: {sqlite_path}")

    # Begin transaction
    db_session = get_db_session()
    try:
        _log("─── Migrating Users ───────────────────────")
        _migrate_users(db_session)

        if sqlite_conn:
            _log("─── Migrating Memory Data ─────────────────")
            _migrate_memory(db_session, sqlite_conn)

        # Commit all changes in one transaction
        db_session.commit()
        _log("═══════════════════════════════════════════════")
        _log("Migration COMPLETE — all data committed.")
        _log("═══════════════════════════════════════════════")

    except Exception as e:
        db_session.rollback()
        _log(f"MIGRATION FAILED — all changes rolled back: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()
        if sqlite_conn:
            sqlite_conn.close()


if __name__ == "__main__":
    main()
