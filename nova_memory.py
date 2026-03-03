"""
nova_memory.py — NOVA's Persistent Adaptive Memory Engine
Extracts facts, interests, and preferences from every conversation
and stores them to nova_memory.json so NOVA grows smarter over time.
"""

import json
import os
import threading
import requests
from datetime import date

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "nova_memory.json")
OLLAMA_URL  = "http://localhost:11434/api/generate"

_EMPTY_STATE = {
    "facts":              [],        # ["User's name is Vinayaka", ...]
    "interests":          {},        # {"Python": 5, "AI": 3}
    "preferences":        {},        # {"tone": "concise", ...}
    "daily_summaries":    {},        # {"2026-03-01": "..."}
    "total_conversations": 0,
    "days_active":         0,
    "first_seen":         None,
}


class NovaMemory:
    """Thread-safe, file-backed memory store for NOVA."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Back-fill any missing keys from the empty template
                for k, v in _EMPTY_STATE.items():
                    data.setdefault(k, v)
                return data
            except Exception:
                pass
        state = dict(_EMPTY_STATE)
        state["first_seen"] = str(date.today())
        return state

    def _save(self):
        """Write current state to disk. Must be called under lock."""
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[NOVA Memory] Save failed: {e}")

    # ── Public Getters ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            top_interests = sorted(
                self._state["interests"].items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            return {
                "facts_count":         len(self._state["facts"]),
                "interests_count":     len(self._state["interests"]),
                "top_interests":       [t for t, _ in top_interests],
                "total_conversations": self._state["total_conversations"],
                "days_active":         self._state["days_active"],
                "first_seen":          self._state["first_seen"],
                "daily_summaries":     len(self._state["daily_summaries"]),
                "facts":               list(self._state["facts"]),
                "interests":           dict(self._state["interests"]),
                "preferences":         dict(self._state["preferences"]),
            }

    def get_memory_context(self) -> str:
        """Returns a prompt snippet summarising what NOVA knows about the user."""
        with self._lock:
            parts = []

            if self._state["facts"]:
                # Keep only the most recent / relevant 20 facts to avoid prompt bloat
                recent_facts = self._state["facts"][-20:]
                parts.append("What I know about the user:\n" +
                             "\n".join(f"• {f}" for f in recent_facts))

            if self._state["interests"]:
                top = sorted(self._state["interests"].items(),
                             key=lambda x: x[1], reverse=True)[:6]
                parts.append("User's top interests: " +
                             ", ".join(f"{t} (×{c})" for t, c in top))

            if self._state["preferences"]:
                prefs = "; ".join(f"{k}: {v}"
                                  for k, v in self._state["preferences"].items())
                parts.append("User preferences: " + prefs)

            today = str(date.today())
            if today in self._state["daily_summaries"]:
                parts.append("Today's session so far: " +
                             self._state["daily_summaries"][today])

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
            self._state["total_conversations"] += 1
            today = str(date.today())
            if today not in self._state["daily_summaries"]:
                self._state["days_active"] += 1
                if not self._state["first_seen"]:
                    self._state["first_seen"] = today
            self._save()

    def extract_and_store(self, user_msg: str, nova_reply: str, model: str = "mistral"):
        """
        Run in a background thread after each conversation turn.
        Asks the LLM to extract facts / interests / preferences
        from the exchange, then persists them.
        """
        t = threading.Thread(
            target=self._extract_worker,
            args=(user_msg, nova_reply, model),
            daemon=True
        )
        t.start()

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

            # Extract JSON from response (model may add markdown fences)
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
            # Merge facts (deduplicate by lowercased comparison)
            existing_lower = {f.lower() for f in self._state["facts"]}
            for fact in extracted.get("facts", []):
                fact = fact.strip()
                if fact and fact.lower() not in existing_lower and len(fact) < 200:
                    self._state["facts"].append(fact)
                    existing_lower.add(fact.lower())
            # Cap at 100 facts (keep most recent)
            if len(self._state["facts"]) > 100:
                self._state["facts"] = self._state["facts"][-100:]

            # Merge interests (increment counters)
            for topic in extracted.get("interests", []):
                topic = topic.strip().title()
                if topic:
                    self._state["interests"][topic] = \
                        self._state["interests"].get(topic, 0) + 1

            # Merge preferences (overwrite)
            for k, v in extracted.get("preferences", {}).items():
                if k and v:
                    self._state["preferences"][str(k)] = str(v)

            self._save()
            print(f"[NOVA Memory] Stored: {len(extracted.get('facts', []))} facts, "
                  f"{len(extracted.get('interests', []))} interests")

    def generate_daily_summary(self, conversation_log: list, model: str = "mistral"):
        """Generate and store a short daily summary from the day's conversation log."""
        if not conversation_log:
            return
        t = threading.Thread(
            target=self._daily_summary_worker,
            args=(conversation_log, model),
            daemon=True
        )
        t.start()

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
                with self._lock:
                    today = str(date.today())
                    self._state["daily_summaries"][today] = summary
                    # Keep only last 30 days
                    if len(self._state["daily_summaries"]) > 30:
                        oldest = sorted(self._state["daily_summaries"].keys())[0]
                        del self._state["daily_summaries"][oldest]
                    self._save()
                print(f"[NOVA Memory] Daily summary saved for {today}")
        except Exception as e:
            print(f"[NOVA Memory] Daily summary error: {e}")

    def reset(self):
        """Clear all learned memory."""
        with self._lock:
            self._state = dict(_EMPTY_STATE)
            self._state["first_seen"] = str(date.today())
            self._save()
        print("[NOVA Memory] Memory reset.")
