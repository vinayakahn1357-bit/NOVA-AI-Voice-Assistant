"""
nova_memory.py — NOVA's Persistent Adaptive Memory Engine
Uses SQLite (WAL mode) for fast, thread-safe, structured persistence.
Database: nova_memory.db  (same directory as this file)

Schema
------
facts            : id, fact TEXT UNIQUE, created_at
interests        : topic TEXT PK, count INTEGER
preferences      : key TEXT PK, value TEXT
daily_summaries  : day TEXT PK, summary TEXT
meta             : key TEXT PK, value TEXT
  keys: total_conversations, days_active, first_seen
"""

import json
import os
import sqlite3
import threading
import copy
import requests
from datetime import date

DB_FILE      = os.path.join(os.path.dirname(__file__), "nova_memory.db")
LEGACY_JSON  = os.path.join(os.path.dirname(__file__), "nova_memory.json")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fact       TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT (date('now'))
);

CREATE TABLE IF NOT EXISTS interests (
    topic TEXT PRIMARY KEY NOT NULL,
    count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS preferences (
    key   TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_summaries (
    day     TEXT PRIMARY KEY NOT NULL,
    summary TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL,
    turn       INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    ts         TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, turn)
);
"""


class NovaMemory:
    """Thread-safe, SQLite-backed memory store for NOVA."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._migrate_legacy_json()
        stats = self.get_stats()
        print(f"[NOVA Memory] SQLite ready — "
              f"{stats['facts_count']} facts, "
              f"{stats['interests_count']} interests, "
              f"{stats['total_conversations']} conversations recorded.")

    # ── Schema init ───────────────────────────────────────────────────────────

    def _init_db(self):
        with self._lock:
            self._conn.executescript(_DDL)
            # Seed meta keys if absent
            for key, default in [
                ("total_conversations", "0"),
                ("days_active",         "0"),
                ("first_seen",          str(date.today())),
            ]:
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                    (key, default)
                )
            self._conn.commit()

    # ── Legacy JSON Migration (one-time) ──────────────────────────────────────

    def _migrate_legacy_json(self):
        """Import nova_memory.json into SQLite if it exists and has data."""
        if not os.path.exists(LEGACY_JSON):
            return
        try:
            with open(LEGACY_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        # Only migrate if the DB is still empty (first run after upgrade)
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            if count > 0:
                return  # already migrated

            cur = self._conn.cursor()

            for fact in data.get("facts", []):
                fact = fact.strip()
                if fact:
                    cur.execute(
                        "INSERT OR IGNORE INTO facts (fact) VALUES (?)", (fact,)
                    )

            for topic, cnt in data.get("interests", {}).items():
                topic = topic.strip().title()
                if topic:
                    cur.execute(
                        "INSERT OR REPLACE INTO interests (topic, count) VALUES (?, ?)",
                        (topic, int(cnt))
                    )

            for k, v in data.get("preferences", {}).items():
                if k and v:
                    cur.execute(
                        "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
                        (str(k), str(v))
                    )

            for day, summary in data.get("daily_summaries", {}).items():
                cur.execute(
                    "INSERT OR REPLACE INTO daily_summaries (day, summary) VALUES (?, ?)",
                    (day, summary)
                )

            # Meta scalars
            for key in ("total_conversations", "days_active", "first_seen"):
                val = data.get(key)
                if val is not None:
                    cur.execute(
                        "UPDATE meta SET value = ? WHERE key = ?",
                        (str(val), key)
                    )

            self._conn.commit()
            print("[NOVA Memory] Migrated legacy nova_memory.json → SQLite ✓")
            # Rename the old file so we don't re-import
            os.rename(LEGACY_JSON, LEGACY_JSON + ".migrated")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_meta(self, key: str, default=None):
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def _set_meta(self, key: str, value):
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, str(value))
        )

    # ── Public Getters ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            facts = [
                r[0] for r in
                self._conn.execute("SELECT fact FROM facts ORDER BY id").fetchall()
            ]
            interests = dict(
                self._conn.execute(
                    "SELECT topic, count FROM interests ORDER BY count DESC"
                ).fetchall()
            )
            preferences = dict(
                self._conn.execute(
                    "SELECT key, value FROM preferences"
                ).fetchall()
            )
            top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5]
            daily_count = self._conn.execute(
                "SELECT COUNT(*) FROM daily_summaries"
            ).fetchone()[0]

            return {
                "facts_count":         len(facts),
                "interests_count":     len(interests),
                "top_interests":       [t for t, _ in top_interests],
                "total_conversations": int(self._get_meta("total_conversations", 0)),
                "days_active":         int(self._get_meta("days_active", 0)),
                "first_seen":          self._get_meta("first_seen"),
                "daily_summaries":     daily_count,
                "facts":               facts,
                "interests":           interests,
                "preferences":         preferences,
            }

    def get_memory_context(self) -> str:
        """Returns a prompt snippet summarising what NOVA knows about the user."""
        with self._lock:
            parts = []

            facts = [
                r[0] for r in
                self._conn.execute(
                    "SELECT fact FROM facts ORDER BY id DESC LIMIT 20"
                ).fetchall()
            ][::-1]  # chronological order
            if facts:
                parts.append(
                    "What I know about the user:\n" +
                    "\n".join(f"• {f}" for f in facts)
                )

            top_interests = self._conn.execute(
                "SELECT topic, count FROM interests ORDER BY count DESC LIMIT 6"
            ).fetchall()
            if top_interests:
                parts.append(
                    "User's top interests: " +
                    ", ".join(f"{t} (×{c})" for t, c in top_interests)
                )

            prefs = self._conn.execute(
                "SELECT key, value FROM preferences"
            ).fetchall()
            if prefs:
                parts.append(
                    "User preferences: " +
                    "; ".join(f"{k}: {v}" for k, v in prefs)
                )

            today = str(date.today())
            row = self._conn.execute(
                "SELECT summary FROM daily_summaries WHERE day = ?", (today,)
            ).fetchone()
            if row:
                parts.append("Today's session so far: " + row[0])

        if not parts:
            return ""

        return (
            "\n\n--- NOVA's Memory (use this to personalise your replies) ---\n"
            + "\n\n".join(parts)
            + "\n--- End of Memory ---"
        )

    # ── Learning ──────────────────────────────────────────────────────────────

    def record_conversation(self):
        """Call once per completed conversation turn."""
        with self._lock:
            total = int(self._get_meta("total_conversations", 0)) + 1
            self._set_meta("total_conversations", total)

            today = str(date.today())
            has_today = self._conn.execute(
                "SELECT 1 FROM daily_summaries WHERE day = ?", (today,)
            ).fetchone()
            # Also count today as an active day if we haven't seen it yet
            # We track via a temp key in meta
            seen_today = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'last_active_day'"
            ).fetchone()
            if not seen_today or seen_today[0] != today:
                days = int(self._get_meta("days_active", 0)) + 1
                self._set_meta("days_active", days)
                self._set_meta("last_active_day", today)
                # Ensure first_seen is set
                if not self._get_meta("first_seen"):
                    self._set_meta("first_seen", today)

            self._conn.commit()

    def extract_and_store(self, user_msg: str, nova_reply: str, model: str = "mistral"):
        """
        Run in a background thread after each conversation turn.
        Asks the LLM to extract facts / interests / preferences.
        """
        self._extract_worker(user_msg, nova_reply, model)

    def _extract_worker(self, user_msg: str, nova_reply: str, model: str):
        try:
            extraction_prompt = f"""You are a memory extraction agent for NOVA AI assistant.

Given this conversation exchange, extract useful, specific information about the USER only.
Ignore generic or repetitive information. Focus on:
1. Personal facts (name, job, location, age, etc.)
2. Topics the user is interested in or asked about
3. User preferences (communication style, preferred response length, languages, etc.)
4. Any other memorable facts worth knowing

Conversation:
User: {user_msg[:500]}
Nova: {nova_reply[:300]}

Respond ONLY with a valid JSON object in this exact format (no extra text):
{{
  "facts": ["fact 1", "fact 2"],
  "interests": ["topic1", "topic2"],
  "preferences": {{"key": "value"}}
}}

Rules:
- Only include SPECIFIC, NON-OBVIOUS facts directly mentioned by the user.
- If nothing notable was said, return empty lists/objects.
- Keep each fact concise (under 15 words).
- Do NOT include facts about Nova, only about the user.
"""
            payload = {
                "model": model,
                "prompt": extraction_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            }

            r = requests.post(OLLAMA_URL, json=payload, timeout=30)
            if r.status_code != 200:
                return

            raw = r.json().get("response", "").strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            extracted = json.loads(raw[start:end])
            self._merge_extracted(extracted)

        except Exception as e:
            print(f"[NOVA Memory] Extraction error: {e}")

    def _merge_extracted(self, extracted: dict):
        with self._lock:
            cur = self._conn.cursor()

            # Facts — deduplicate, cap at 100
            existing = {
                r[0].lower() for r in
                cur.execute("SELECT fact FROM facts").fetchall()
            }
            added = 0
            for fact in extracted.get("facts", []):
                fact = fact.strip()
                if fact and fact.lower() not in existing and len(fact) < 200:
                    cur.execute(
                        "INSERT OR IGNORE INTO facts (fact) VALUES (?)", (fact,)
                    )
                    existing.add(fact.lower())
                    added += 1

            # Trim to most recent 100 facts
            cur.execute(
                "DELETE FROM facts WHERE id NOT IN "
                "(SELECT id FROM facts ORDER BY id DESC LIMIT 100)"
            )

            # Interests — increment counters
            for topic in extracted.get("interests", []):
                topic = topic.strip().title()
                if topic:
                    cur.execute(
                        "INSERT INTO interests (topic, count) VALUES (?, 1) "
                        "ON CONFLICT(topic) DO UPDATE SET count = count + 1",
                        (topic,)
                    )

            # Preferences — upsert
            for k, v in extracted.get("preferences", {}).items():
                if k and v:
                    cur.execute(
                        "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
                        (str(k), str(v))
                    )

            self._conn.commit()
            print(f"[NOVA Memory] Stored: {added} new facts, "
                  f"{len(extracted.get('interests', []))} interests")

    def generate_daily_summary(self, conversation_log: list, model: str = "mistral"):
        """Generate and store a short daily summary from the day's conversation log."""
        if not conversation_log:
            return
        self._daily_summary_worker(conversation_log, model)

    def _daily_summary_worker(self, conversation_log: list, model: str):
        try:
            convo_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Nova'}: {m['content'][:200]}"
                for m in conversation_log[-20:]
            )
            prompt = (
                "Summarise this conversation in 1-2 sentences, focusing on what the user "
                "discussed, asked about, or accomplished today. Be specific and concise.\n\n"
                f"Conversation:\n{convo_text}\n\nSummary:"
            )
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 80},
            }
            r = requests.post(OLLAMA_URL, json=payload, timeout=30)
            if r.status_code != 200:
                return
            summary = r.json().get("response", "").strip()
            if summary:
                today = str(date.today())
                with self._lock:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO daily_summaries (day, summary) VALUES (?, ?)",
                        (today, summary)
                    )
                    # Keep only last 30 days
                    self._conn.execute(
                        "DELETE FROM daily_summaries WHERE day NOT IN "
                        "(SELECT day FROM daily_summaries ORDER BY day DESC LIMIT 30)"
                    )
                    self._conn.commit()
                print(f"[NOVA Memory] Daily summary saved for {today}")
        except Exception as e:
            print(f"[NOVA Memory] Daily summary error: {e}")

    def reset(self):
        """Clear all learned memory."""
        with self._lock:
            self._conn.executescript("""
                DELETE FROM facts;
                DELETE FROM interests;
                DELETE FROM preferences;
                DELETE FROM daily_summaries;
                UPDATE meta SET value = '0' WHERE key IN ('total_conversations', 'days_active');
                DELETE FROM meta WHERE key = 'last_active_day';
                UPDATE meta SET value = date('now') WHERE key = 'first_seen';
            """)
            self._conn.commit()
        print("[NOVA Memory] Memory reset.")
