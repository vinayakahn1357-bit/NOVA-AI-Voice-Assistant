"""
services/db_memory_service.py — PostgreSQL-backed Memory Service for NOVA (Phase 6)
Provides multi-user memory storage via SQLAlchemy ORM.
All queries are scoped to user_id for complete data isolation.
"""

import json
import requests
from datetime import date, datetime, timezone

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database import (
    get_db_session, DBMemoryFact, DBUserInterest, DBUserPreference,
    DBDailySummary, DBMeta, is_postgres,
)
from utils.logger import get_logger

log = get_logger("db_memory")

# Groq API endpoint for LLM calls
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class DbMemoryService:
    """
    PostgreSQL-backed memory store for NOVA.
    Drop-in replacement for NovaMemory with multi-user support.
    All methods take user_id to scope data per user.
    """

    def __init__(self):
        self._is_pg = is_postgres()
        log.info("DbMemoryService initialized (backend=%s)",
                 "PostgreSQL" if self._is_pg else "SQLAlchemy-SQLite")

    # ── Public Getters ────────────────────────────────────────────────────

    def get_stats(self, user_id: str = "default") -> dict:
        """Get memory statistics for a specific user."""
        session = get_db_session()
        try:
            facts = [
                r.fact for r in
                session.query(DBMemoryFact)
                .filter_by(user_id=user_id)
                .order_by(DBMemoryFact.id)
                .all()
            ]

            interests_rows = (
                session.query(DBUserInterest.topic, DBUserInterest.count)
                .filter_by(user_id=user_id)
                .order_by(DBUserInterest.count.desc())
                .all()
            )
            interests = {r.topic: r.count for r in interests_rows}
            top_interests = [t for t, _ in sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5]]

            preferences = {
                r.key: r.value for r in
                session.query(DBUserPreference)
                .filter_by(user_id=user_id)
                .all()
            }

            daily_count = (
                session.query(func.count(DBDailySummary.id))
                .filter_by(user_id=user_id)
                .scalar()
            )

            total_conversations = int(self._get_meta(session, user_id, "total_conversations", "0"))
            days_active = int(self._get_meta(session, user_id, "days_active", "0"))
            first_seen = self._get_meta(session, user_id, "first_seen", str(date.today()))

            return {
                "facts_count": len(facts),
                "interests_count": len(interests),
                "top_interests": top_interests,
                "total_conversations": total_conversations,
                "days_active": days_active,
                "first_seen": first_seen,
                "daily_summaries": daily_count,
                "facts": facts,
                "interests": interests,
                "preferences": preferences,
            }
        finally:
            session.close()

    def get_memory_context(self, user_id: str = "default") -> str:
        """Build a prompt snippet summarising what NOVA knows about a user."""
        session = get_db_session()
        try:
            parts = []

            facts = [
                r.fact for r in
                session.query(DBMemoryFact)
                .filter_by(user_id=user_id)
                .order_by(DBMemoryFact.id.desc())
                .limit(20)
                .all()
            ][::-1]  # chronological

            if facts:
                parts.append(
                    "What I know about the user:\n" +
                    "\n".join(f"• {f}" for f in facts)
                )

            top_interests = (
                session.query(DBUserInterest.topic, DBUserInterest.count)
                .filter_by(user_id=user_id)
                .order_by(DBUserInterest.count.desc())
                .limit(6)
                .all()
            )
            if top_interests:
                parts.append(
                    "User's top interests: " +
                    ", ".join(f"{t} (×{c})" for t, c in top_interests)
                )

            prefs = (
                session.query(DBUserPreference.key, DBUserPreference.value)
                .filter_by(user_id=user_id)
                .all()
            )
            if prefs:
                parts.append(
                    "User preferences: " +
                    "; ".join(f"{k}: {v}" for k, v in prefs)
                )

            today = str(date.today())
            row = (
                session.query(DBDailySummary.summary)
                .filter_by(user_id=user_id, day=today)
                .first()
            )
            if row:
                parts.append("Today's session so far: " + row.summary)

        finally:
            session.close()

        if not parts:
            return ""

        return (
            "\n\n--- NOVA's Memory (use this to personalise your replies) ---\n"
            + "\n\n".join(parts)
            + "\n--- End of Memory ---"
        )

    # ── Learning ──────────────────────────────────────────────────────────

    def record_conversation(self, user_id: str = "default"):
        """Record a conversation turn for a user."""
        session = get_db_session()
        try:
            total = int(self._get_meta(session, user_id, "total_conversations", "0")) + 1
            self._set_meta(session, user_id, "total_conversations", str(total))

            today = str(date.today())
            last_active = self._get_meta(session, user_id, "last_active_day", "")
            if last_active != today:
                days = int(self._get_meta(session, user_id, "days_active", "0")) + 1
                self._set_meta(session, user_id, "days_active", str(days))
                self._set_meta(session, user_id, "last_active_day", today)
                if not self._get_meta(session, user_id, "first_seen"):
                    self._set_meta(session, user_id, "first_seen", today)

            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("record_conversation error: %s", e)
        finally:
            session.close()

    def extract_and_store(self, user_msg: str, nova_reply: str,
                          model: str = "mistral", provider_config: dict = None,
                          user_id: str = "default"):
        """Extract facts/interests from a conversation turn and store."""
        self._extract_worker(user_msg, nova_reply, model, provider_config or {}, user_id)

    def _extract_worker(self, user_msg, nova_reply, model, provider_config, user_id):
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
            raw = self._call_llm(extraction_prompt, model, provider_config,
                                 temperature=0.1, max_tokens=300)
            if not raw:
                return

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            extracted = json.loads(raw[start:end])
            self._merge_extracted(extracted, user_id)

        except Exception as e:
            log.warning("Extraction error: %s", e)

    def _merge_extracted(self, extracted: dict, user_id: str):
        session = get_db_session()
        try:
            # Facts — deduplicate, cap at 100
            existing = {
                r.fact.lower() for r in
                session.query(DBMemoryFact).filter_by(user_id=user_id).all()
            }
            added = 0
            for fact in extracted.get("facts", []):
                fact = fact.strip()
                if fact and fact.lower() not in existing and len(fact) < 200:
                    session.add(DBMemoryFact(user_id=user_id, fact=fact))
                    existing.add(fact.lower())
                    added += 1

            session.flush()

            # Trim to most recent 100 facts
            count = session.query(func.count(DBMemoryFact.id)).filter_by(user_id=user_id).scalar()
            if count > 100:
                oldest = (
                    session.query(DBMemoryFact.id)
                    .filter_by(user_id=user_id)
                    .order_by(DBMemoryFact.id.asc())
                    .limit(count - 100)
                    .all()
                )
                if oldest:
                    ids_to_delete = [r.id for r in oldest]
                    session.query(DBMemoryFact).filter(DBMemoryFact.id.in_(ids_to_delete)).delete(synchronize_session=False)

            # Interests — increment counters
            for topic in extracted.get("interests", []):
                topic = topic.strip().title()
                if topic:
                    existing_interest = (
                        session.query(DBUserInterest)
                        .filter_by(user_id=user_id, topic=topic)
                        .first()
                    )
                    if existing_interest:
                        existing_interest.count += 1
                    else:
                        session.add(DBUserInterest(user_id=user_id, topic=topic, count=1))

            # Preferences — upsert
            for k, v in extracted.get("preferences", {}).items():
                if k and v:
                    existing_pref = (
                        session.query(DBUserPreference)
                        .filter_by(user_id=user_id, key=str(k))
                        .first()
                    )
                    if existing_pref:
                        existing_pref.value = str(v)
                    else:
                        session.add(DBUserPreference(user_id=user_id, key=str(k), value=str(v)))

            session.commit()
            log.info("Stored: %d new facts, %d interests (user=%s)",
                     added, len(extracted.get("interests", [])), user_id)

        except Exception as e:
            session.rollback()
            log.warning("merge_extracted error: %s", e)
        finally:
            session.close()

    def generate_daily_summary(self, conversation_log: list, model: str = "mistral",
                                provider_config: dict = None, user_id: str = "default"):
        """Generate and store a daily summary."""
        if not conversation_log:
            return
        self._daily_summary_worker(conversation_log, model, provider_config or {}, user_id)

    def _daily_summary_worker(self, conversation_log, model, provider_config, user_id):
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
            summary = self._call_llm(prompt, model, provider_config,
                                     temperature=0.3, max_tokens=80)
            if not summary:
                return

            today = str(date.today())
            db_session = get_db_session()
            try:
                existing = (
                    db_session.query(DBDailySummary)
                    .filter_by(user_id=user_id, day=today)
                    .first()
                )
                if existing:
                    existing.summary = summary
                else:
                    db_session.add(DBDailySummary(user_id=user_id, day=today, summary=summary))

                # Keep only last 30 days
                count = db_session.query(func.count(DBDailySummary.id)).filter_by(user_id=user_id).scalar()
                if count > 30:
                    oldest = (
                        db_session.query(DBDailySummary.id)
                        .filter_by(user_id=user_id)
                        .order_by(DBDailySummary.day.asc())
                        .limit(count - 30)
                        .all()
                    )
                    if oldest:
                        ids_to_delete = [r.id for r in oldest]
                        db_session.query(DBDailySummary).filter(DBDailySummary.id.in_(ids_to_delete)).delete(synchronize_session=False)

                db_session.commit()
                log.info("Daily summary saved for %s (user=%s)", today, user_id)

            except Exception as e:
                db_session.rollback()
                log.warning("daily_summary save error: %s", e)
            finally:
                db_session.close()

        except Exception as e:
            log.warning("Daily summary error: %s", e)

    def reset(self, user_id: str = "default"):
        """Clear all learned memory for a user."""
        session = get_db_session()
        try:
            session.query(DBMemoryFact).filter_by(user_id=user_id).delete()
            session.query(DBUserInterest).filter_by(user_id=user_id).delete()
            session.query(DBUserPreference).filter_by(user_id=user_id).delete()
            session.query(DBDailySummary).filter_by(user_id=user_id).delete()
            # Reset meta counters
            for key in ("total_conversations", "days_active"):
                self._set_meta(session, user_id, key, "0")
            meta_row = session.query(DBMeta).filter_by(user_id=user_id, key="last_active_day").first()
            if meta_row:
                session.delete(meta_row)
            self._set_meta(session, user_id, "first_seen", str(date.today()))
            session.commit()
            log.info("Memory reset for user=%s", user_id)
        except Exception as e:
            session.rollback()
            log.warning("Memory reset error: %s", e)
        finally:
            session.close()

    # ── LLM Routing ───────────────────────────────────────────────────────

    def _call_llm(self, prompt, model, provider_config, temperature=0.1, max_tokens=300):
        """Call LLM via best available provider. Same logic as NovaMemory._call_llm."""
        provider = provider_config.get("provider", "ollama")
        groq_key = provider_config.get("groq_api_key", "")

        if groq_key and provider in ("groq", "hybrid"):
            groq_model = provider_config.get("groq_model", "llama-3.3-70b-versatile")
            try:
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                r = requests.post(GROQ_API_URL, json=payload,
                                  headers=headers, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    return (data.get("choices", [{}])[0]
                            .get("message", {}).get("content", "")).strip()
            except Exception as e:
                log.warning("Groq extraction failed: %s", e)

        cloud_key = provider_config.get("ollama_api_key", "")
        cloud_url = provider_config.get("ollama_cloud_url", "")
        if cloud_key and cloud_url and provider in ("ollama_cloud", "hybrid"):
            try:
                headers = {"Authorization": f"Bearer {cloud_key}"}
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                }
                r = requests.post(cloud_url, json=payload,
                                  headers=headers, timeout=30)
                if r.status_code == 200:
                    return r.json().get("response", "").strip()
            except Exception as e:
                log.warning("Ollama Cloud extraction failed: %s", e)

        return ""

    # ── Meta Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_meta(session, user_id, key, default=None):
        row = session.query(DBMeta).filter_by(user_id=user_id, key=key).first()
        return row.value if row else default

    @staticmethod
    def _set_meta(session, user_id, key, value):
        row = session.query(DBMeta).filter_by(user_id=user_id, key=key).first()
        if row:
            row.value = str(value)
        else:
            session.add(DBMeta(user_id=user_id, key=key, value=str(value)))
