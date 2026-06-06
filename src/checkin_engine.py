"""
Proactive Check-in Engine
Analyses usage patterns and triggers smart check-in nudges.
No external dependencies — pure Python.
"""

import json
import os
import datetime
from collections import Counter

CHECKIN_FILE = "checkin_state.json"


class CheckinEngine:
    def __init__(self):
        self.state = self._load()

    # ── Record a session ─────────────────────────────────────────────────────────
    def record_session(self, emotion: str):
        now = datetime.datetime.now()
        self.state["sessions"].append({
            "emotion":   emotion,
            "hour":      now.hour,
            "weekday":   now.weekday(),       # 0=Mon … 6=Sun
            "date":      now.strftime("%Y-%m-%d"),
            "timestamp": now.isoformat(),
        })
        # Keep last 60 sessions only
        self.state["sessions"] = self.state["sessions"][-60:]
        self.state["last_seen"] = now.isoformat()
        self._save()

    # ── Should we nudge the user? ────────────────────────────────────────────────
    def should_checkin(self) -> tuple:
        """
        Returns (should_show: bool, message: str)
        Rules checked in priority order.
        """
        sessions  = self.state.get("sessions", [])
        last_seen = self.state.get("last_seen")

        # Rule 1: First-ever session
        if not sessions:
            return False, ""

        # Rule 2: Long absence (>18 hours since last session)
        if last_seen:
            try:
                last_dt   = datetime.datetime.fromisoformat(last_seen)
                hours_gap = (datetime.datetime.now() - last_dt).total_seconds() / 3600
                if hours_gap > 18:
                    return True, self._absence_message(int(hours_gap))
            except Exception:
                pass

        # Rule 3: Recurring negative emotion at same time of day
        msg = self._pattern_message(sessions)
        if msg:
            return True, msg

        # Rule 4: Streak about to break
        streak_msg = self._streak_warning(sessions)
        if streak_msg:
            return True, streak_msg

        return False, ""

    # ── Pattern detection ────────────────────────────────────────────────────────
    def _pattern_message(self, sessions: list) -> str:
        if len(sessions) < 5:
            return ""

        now_hour    = datetime.datetime.now().hour
        time_bucket = self._hour_to_bucket(now_hour)

        same_time = [
            s for s in sessions[-20:]
            if self._hour_to_bucket(s["hour"]) == time_bucket
        ]

        if len(same_time) < 3:
            return ""

        emotions = [s["emotion"] for s in same_time]
        most_common, count = Counter(emotions).most_common(1)[0]

        negative = ["sad", "anxious", "stressed", "angry"]
        if most_common in negative and count >= 3:
            bucket_label = {
                "morning":   "in the mornings",
                "afternoon": "in the afternoons",
                "evening":   "in the evenings",
                "night":     "at night",
            }.get(time_bucket, "around this time")

            msg_map = {
                "stressed": f"You've felt stressed {bucket_label} lately. Want to do a quick reset?",
                "anxious":  f"Anxiety tends to peak for you {bucket_label}. Let's check in. 💙",
                "sad":      f"You've been feeling low {bucket_label} recently. I'm here for you.",
                "angry":    f"You've felt frustrated {bucket_label} a few times. Want to talk?",
            }
            return msg_map.get(most_common, "")

        return ""

    def _absence_message(self, hours: int) -> str:
        if hours < 24:
            return "Hey, you haven't checked in for a while. How are you feeling right now? 💙"
        days = int(hours // 24)
        return f"It's been {days} day{'s' if days > 1 else ''} since your last check-in. How are you doing?"

    def _streak_warning(self, sessions: list) -> str:
        if len(sessions) < 2:
            return ""
        today     = datetime.date.today().isoformat()
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        dates     = [s["date"] for s in sessions]
        if yesterday in dates and today not in dates:
            return "🔥 Don't break your streak! Check in today to keep it going."
        return ""

    def _hour_to_bucket(self, hour: int) -> str:
        if   5  <= hour < 12: return "morning"
        elif 12 <= hour < 17: return "afternoon"
        elif 17 <= hour < 21: return "evening"
        else:                  return "night"

    # ── Persistence ──────────────────────────────────────────────────────────────
    def _save(self):
        try:
            with open(CHECKIN_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def _load(self) -> dict:
        try:
            if os.path.exists(CHECKIN_FILE):
                with open(CHECKIN_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"sessions": [], "last_seen": None}

    def get_stats(self) -> dict:
        sessions = self.state.get("sessions", [])
        if not sessions:
            return {"total": 0, "peak_hour": None, "top_emotion": None}

        hours    = [s["hour"] for s in sessions]
        peak     = Counter(hours).most_common(1)[0][0] if hours else None
        emotions = [s["emotion"] for s in sessions]
        top_em   = Counter(emotions).most_common(1)[0][0] if emotions else None

        return {
            "total":       len(sessions),
            "peak_hour":   f"{peak:02d}:00" if peak is not None else None,
            "top_emotion": top_em,
        }