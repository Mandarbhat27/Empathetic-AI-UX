"""
Predictive Proactive Empathy Engine — Feature 4
Analyses tomorrow's calendar + historical emotion patterns
→ predicts stress risk tonight → sends proactive prep suggestion

No new packages needed — pure Python
"""

import json
import os
import datetime
from collections import Counter

CALENDAR_FILE  = "mock_calendar.json"
PREDICTION_LOG = "prediction_log.json"


# ════════════════════════════════════════════════════════════════════════════════
# MOCK CALENDAR — auto-created with sample data if not present
# ════════════════════════════════════════════════════════════════════════════════
DEFAULT_CALENDAR = {
    "description": "Edit this file to simulate your real schedule",
    "meetings": [
        {
            "title":      "Team standup",
            "date":       "tomorrow",
            "start_time": "09:00",
            "end_time":   "09:30",
            "type":       "meeting"
        },
        {
            "title":      "Product review",
            "date":       "tomorrow",
            "start_time": "09:45",
            "end_time":   "11:00",
            "type":       "meeting"
        },
        {
            "title":      "1:1 with manager",
            "date":       "tomorrow",
            "start_time": "11:15",
            "end_time":   "12:00",
            "type":       "meeting"
        },
        {
            "title":      "Lunch break",
            "date":       "tomorrow",
            "start_time": "12:00",
            "end_time":   "13:00",
            "type":       "break"
        },
        {
            "title":      "Sprint planning",
            "date":       "tomorrow",
            "start_time": "13:00",
            "end_time":   "15:00",
            "type":       "meeting"
        },
        {
            "title":      "Design review",
            "date":       "tomorrow",
            "start_time": "15:15",
            "end_time":   "16:00",
            "type":       "meeting"
        },
    ]
}


def ensure_calendar_exists():
    """Create mock_calendar.json if it doesn't exist"""
    if not os.path.exists(CALENDAR_FILE):
        with open(CALENDAR_FILE, "w") as f:
            json.dump(DEFAULT_CALENDAR, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# PREDICTIVE ENGINE
# ════════════════════════════════════════════════════════════════════════════════
class PredictiveEmpathyEngine:
    """
    Predicts tomorrow's stress risk based on:
    1. Calendar density (number + back-to-back meetings)
    2. Historical emotion patterns on similar-density days
    3. Time-of-day pattern (when user usually feels stressed)
    """

    def __init__(self, checkin_engine=None):
        self.checkin   = checkin_engine
        self.name      = "PredictiveEmpathyEngine"
        ensure_calendar_exists()

    # ── Main run ─────────────────────────────────────────────────────────────────
    def predict(self) -> dict:
        """
        Returns full prediction dict including:
        risk_level, score, schedule_analysis, history_pattern,
        should_warn, warning_message, prep_kit
        """
        schedule = self._analyse_schedule()
        history  = self._analyse_history()
        score    = self._calculate_risk(schedule, history)
        level    = self._score_to_level(score)

        return {
            "risk_level":        level,
            "risk_score":        score,
            "schedule_analysis": schedule,
            "history_pattern":   history,
            "should_warn":       score >= 55,
            "warning_message":   self._build_warning(level, schedule),
            "prep_kit":          self._build_prep_kit(level, schedule),
            "prediction_time":   datetime.datetime.now().strftime("%I:%M %p"),
        }

    # ── Schedule analysis ────────────────────────────────────────────────────────
    def _analyse_schedule(self) -> dict:
        try:
            with open(CALENDAR_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            return {"total_meetings": 0, "back_to_back": 0, "free_blocks": 0, "density": "light"}

        meetings = [m for m in data.get("meetings", []) if m.get("type") != "break"]
        total    = len(meetings)

        # Detect back-to-back (gap < 20 mins)
        b2b_count = 0
        sorted_m  = sorted(meetings, key=lambda x: x.get("start_time", "00:00"))
        for i in range(len(sorted_m) - 1):
            try:
                end_h, end_m   = map(int, sorted_m[i]["end_time"].split(":"))
                start_h, start_m = map(int, sorted_m[i+1]["start_time"].split(":"))
                gap_mins = (start_h * 60 + start_m) - (end_h * 60 + end_m)
                if 0 <= gap_mins <= 20:
                    b2b_count += 1
            except Exception:
                pass

        # Count free blocks (gaps > 45 mins)
        free_blocks = 0
        for i in range(len(sorted_m) - 1):
            try:
                end_h, end_m    = map(int, sorted_m[i]["end_time"].split(":"))
                start_h, start_m= map(int, sorted_m[i+1]["start_time"].split(":"))
                gap_mins = (start_h * 60 + start_m) - (end_h * 60 + end_m)
                if gap_mins > 45:
                    free_blocks += 1
            except Exception:
                pass

        density = (
            "intense"  if total >= 5 and b2b_count >= 3 else
            "heavy"    if total >= 4 or b2b_count >= 2 else
            "moderate" if total >= 2 else
            "light"
        )

        return {
            "total_meetings": total,
            "back_to_back":   b2b_count,
            "free_blocks":    free_blocks,
            "density":        density,
            "meetings":       sorted_m,
        }

    # ── Historical pattern analysis ───────────────────────────────────────────
    def _analyse_history(self) -> dict:
        if not self.checkin:
            return {"pattern_found": False, "typical_emotion": None, "confidence": 0}

        sessions = self.checkin.state.get("sessions", [])
        if len(sessions) < 5:
            return {"pattern_found": False, "typical_emotion": None, "confidence": 0}

        tomorrow_weekday = (datetime.date.today() + datetime.timedelta(days=1)).weekday()

        # Find sessions on same weekday
        same_weekday = [s for s in sessions if s.get("weekday") == tomorrow_weekday]

        if len(same_weekday) < 2:
            return {"pattern_found": False, "typical_emotion": None, "confidence": 0}

        emotions = [s["emotion"] for s in same_weekday]
        top_em, count = Counter(emotions).most_common(1)[0]
        confidence = round(count / len(same_weekday) * 100)

        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_name  = day_names[tomorrow_weekday]

        return {
            "pattern_found":   True,
            "typical_emotion": top_em,
            "confidence":      confidence,
            "day_name":        day_name,
            "sample_size":     len(same_weekday),
        }

    # ── Risk score ───────────────────────────────────────────────────────────────
    def _calculate_risk(self, schedule: dict, history: dict) -> int:
        score = 0

        # Schedule component (0–60 points)
        density_scores = {"intense": 60, "heavy": 45, "moderate": 20, "light": 5}
        score += density_scores.get(schedule.get("density", "light"), 0)

        # Back-to-back bonus
        score += min(schedule.get("back_to_back", 0) * 5, 20)

        # History component (0–40 points)
        if history.get("pattern_found"):
            negative = ["stressed", "anxious", "sad", "angry"]
            if history.get("typical_emotion") in negative:
                score += int(history.get("confidence", 0) * 0.4)

        return min(score, 100)

    def _score_to_level(self, score: int) -> str:
        if score >= 75: return "high"
        if score >= 55: return "moderate"
        if score >= 30: return "low"
        return "minimal"

    # ── Warning message ──────────────────────────────────────────────────────────
    def _build_warning(self, level: str, schedule: dict) -> str:
        density  = schedule.get("density", "light")
        meetings = schedule.get("total_meetings", 0)
        b2b      = schedule.get("back_to_back", 0)

        if level == "high":
            return (
                f"Tomorrow looks intense — {meetings} meetings "
                f"with {b2b} back-to-back blocks. "
                f"Based on your patterns, this kind of day can be draining. "
                f"Here's your prep kit for tonight 🎯"
            )
        elif level == "moderate":
            return (
                f"Tomorrow has a {density} schedule ({meetings} meetings). "
                f"A little prep tonight can make a real difference. "
                f"Here are some things that might help 💙"
            )
        else:
            return (
                f"Tomorrow looks manageable ({meetings} meetings). "
                f"You're in good shape! A quick wind-down tonight keeps the streak going 🌙"
            )

    # ── Prep kit ─────────────────────────────────────────────────────────────────
    def _build_prep_kit(self, level: str, schedule: dict) -> list:
        b2b   = schedule.get("back_to_back", 0)
        free  = schedule.get("free_blocks", 0)

        base = [
            {
                "icon":  "🌙",
                "title": "Wind down by 10 PM",
                "desc":  "Good sleep is the most impactful thing you can do before a busy day"
            },
            {
                "icon":  "📋",
                "title": "Write your top 3 priorities",
                "desc":  "Decide tonight what MUST happen tomorrow so you don't decide under pressure"
            },
        ]

        if level in ["high", "moderate"]:
            base.append({
                "icon":  "⏱️",
                "title": "Schedule micro-breaks",
                "desc":  f"You have {b2b} back-to-back blocks — protect at least one 10-min gap"
                         if b2b > 0 else "Block 2 x 10-min breaks between meetings — protect them"
            })
            base.append({
                "icon":  "💧",
                "title": "Prep your environment",
                "desc":  "Water bottle filled, headphones charged, desk cleared before you sleep"
            })

        if level == "high":
            base.append({
                "icon":  "🧘",
                "title": "5-min morning grounding",
                "desc":  "Before the first meeting, do 5 box breaths (4s in, 4s hold, 4s out, 4s hold)"
            })
            base.append({
                "icon":  "🎵",
                "title": "Focus playlist ready",
                "desc":  "Queue up an instrumental playlist for the heavy blocks — it reduces cognitive load"
            })

        if free == 0 and level == "high":
            base.append({
                "icon":  "🚪",
                "title": "Build in one hard stop",
                "desc":  "Pick one meeting you will leave exactly on time — no overruns. Protect yourself."
            })

        return base

    def get_calendar_summary(self) -> dict:
        """For displaying in the UI without running full prediction"""
        return self._analyse_schedule()