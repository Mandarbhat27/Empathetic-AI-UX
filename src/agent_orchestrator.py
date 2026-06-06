"""
Multi-Agent Orchestration System — Feature 2
4 agents working together for richer empathetic responses:
  1. MemoryAgent    → retrieves relevant past context
  2. PersonaAgent   → tracks user style, evolves AI personality
  3. GuardianAgent  → safety checker before every response
  4. SynthesisAgent → orchestrates all 3, calls Groq LLM

No new packages needed — pure Python + existing Groq API
"""

import os
import re
import json
import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama3-8b-8192"

# ════════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Memory Agent
# Retrieves relevant past context from MemoryEngine
# ════════════════════════════════════════════════════════════════════════════════
class MemoryAgent:
    """
    Manages short-term + long-term memory context.
    Builds a structured memory brief for the Synthesis Agent.
    """
    def __init__(self, memory_engine):
        self.memory = memory_engine
        self.name   = "MemoryAgent"

    def run(self, user_text: str, emotion: str) -> dict:
        recent_context  = self.memory.get_context()
        similar_memories= self.memory.search_similar(user_text, n_results=3)
        dominant        = self.memory.get_dominant_emotion()
        trend           = self.memory.get_emotion_trend(5)

        # Build insight string
        trend_str = " → ".join(trend) if trend else "no history yet"
        recurring = self._detect_recurring(trend)

        return {
            "agent":            self.name,
            "recent_context":   recent_context,
            "similar_memories": similar_memories,
            "dominant_emotion": dominant,
            "emotion_trend":    trend_str,
            "recurring_pattern":recurring,
            "memory_count":     len(self.memory.long_term),
        }

    def _detect_recurring(self, trend: list) -> str:
        if len(trend) < 3:
            return None
        negative = ["sad", "anxious", "stressed", "angry"]
        last3 = trend[-3:]
        if all(e in negative for e in last3):
            return f"User has been {last3[-1]} for 3+ consecutive sessions"
        if trend.count(trend[-1]) >= 3:
            return f"Recurring emotion: {trend[-1]}"
        return None


# ════════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Persona Agent
# Tracks user communication style and evolves AI personality over time
# ════════════════════════════════════════════════════════════════════════════════
class PersonaAgent:
    """
    Analyses how the user communicates and adapts the AI's personality.
    Tracks: formality level, emoji usage, message length, vocabulary richness.
    """
    def __init__(self):
        self.name          = "PersonaAgent"
        self.history       = []   # list of user messages analysed
        self.persona_file  = "persona_state.json"
        self.state         = self._load()

    def run(self, user_text: str, chat_history: list) -> dict:
        # Analyse current message
        style = self._analyse_style(user_text)
        self._update_state(style)

        # Build persona profile
        profile = self._build_profile()
        tone_instruction = self._tone_instruction(profile)

        return {
            "agent":             self.name,
            "current_style":     style,
            "accumulated_profile": profile,
            "tone_instruction":  tone_instruction,
            "sessions_tracked":  self.state.get("session_count", 0),
        }

    def _analyse_style(self, text: str) -> dict:
        words      = text.split()
        has_emoji  = bool(re.search(r'[^\w\s,\.!?]', text))
        has_slang  = any(w.lower() in ["lol","lmao","omg","tbh","ngl","idk","bruh","fr","nah","ya"] for w in words)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        formal     = avg_word_len > 5 and not has_slang and not has_emoji

        return {
            "word_count":    len(words),
            "has_emoji":     has_emoji,
            "has_slang":     has_slang,
            "formal":        formal,
            "avg_word_len":  round(avg_word_len, 1),
            "uses_questions": text.strip().endswith("?"),
        }

    def _update_state(self, style: dict):
        s = self.state
        s["session_count"]    = s.get("session_count", 0) + 1
        s["emoji_count"]      = s.get("emoji_count", 0) + (1 if style["has_emoji"] else 0)
        s["slang_count"]      = s.get("slang_count", 0) + (1 if style["has_slang"] else 0)
        s["formal_count"]     = s.get("formal_count", 0) + (1 if style["formal"] else 0)
        s["total_words"]      = s.get("total_words", 0) + style["word_count"]
        self._save()

    def _build_profile(self) -> dict:
        s   = self.state
        n   = max(s.get("session_count", 1), 1)
        return {
            "prefers_emoji":   s.get("emoji_count", 0) / n > 0.4,
            "uses_slang":      s.get("slang_count", 0) / n > 0.3,
            "is_formal":       s.get("formal_count", 0) / n > 0.5,
            "avg_msg_length":  round(s.get("total_words", 0) / n, 1),
            "sessions":        n,
        }

    def _tone_instruction(self, profile: dict) -> str:
        parts = []
        if profile["is_formal"]:
            parts.append("Use formal, professional language")
        elif profile["uses_slang"]:
            parts.append("Use casual, conversational language")
        else:
            parts.append("Use warm, friendly language")

        if profile["prefers_emoji"]:
            parts.append("include 1-2 relevant emojis naturally")
        else:
            parts.append("avoid emojis unless truly fitting")

        if profile["avg_msg_length"] < 8:
            parts.append("keep your response concise (user prefers short messages)")
        elif profile["avg_msg_length"] > 20:
            parts.append("you can be a bit more detailed (user writes longer messages)")

        return ". ".join(parts) + "."

    def _save(self):
        try:
            with open(self.persona_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def _load(self) -> dict:
        try:
            if os.path.exists(self.persona_file):
                with open(self.persona_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}


# ════════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Guardian Agent
# Safety checker — runs on EVERY response before it's shown to the user
# ════════════════════════════════════════════════════════════════════════════════
class GuardianAgent:
    """
    Screens user input for crisis signals.
    Screens LLM output for harmful/inappropriate content.
    Never blocks — only flags and redirects.
    """

    CRISIS_KEYWORDS = [
        "kill myself", "end my life", "suicide", "want to die",
        "hurt myself", "self harm", "can't go on", "no point living",
        "not worth living", "giving up on life"
    ]

    HARMFUL_OUTPUT_PATTERNS = [
        r"you should (hurt|harm|kill)",
        r"(medication|drug) (overdose|too much)",
        r"it('s| is) better (to die|if you died)",
    ]

    CRISIS_RESOURCE = (
        "I want to make sure you're okay. "
        "If you're having thoughts of harming yourself, please reach out to a crisis helpline — "
        "iCall India: 9152987821 · Vandrevala Foundation: 1860-2662-345 (24/7). "
        "You don't have to face this alone. 💙"
    )

    def __init__(self):
        self.name = "GuardianAgent"

    def check_input(self, user_text: str) -> dict:
        text_lower = user_text.lower()
        crisis     = any(kw in text_lower for kw in self.CRISIS_KEYWORDS)
        return {
            "agent":          self.name,
            "crisis_detected": crisis,
            "crisis_response": self.CRISIS_RESOURCE if crisis else None,
            "input_safe":      not crisis,
        }

    def check_output(self, response_text: str) -> dict:
        harmful = any(
            re.search(p, response_text.lower())
            for p in self.HARMFUL_OUTPUT_PATTERNS
        )
        return {
            "agent":         self.name,
            "output_safe":   not harmful,
            "was_filtered":  harmful,
        }

    def safe_fallback(self, emotion: str) -> str:
        return (
            f"I can hear that you're going through something really difficult. "
            f"Your feelings are completely valid. "
            f"Would you like to talk about what's been happening? "
            f"I'm here and I'm listening. 💙"
        )


# ════════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Synthesis Agent (Orchestrator)
# Combines all 3 agent outputs → builds final LLM prompt → returns response
# ════════════════════════════════════════════════════════════════════════════════
class SynthesisAgent:
    """
    The orchestrator. Receives outputs from all other agents,
    builds an enriched prompt, calls Groq LLM, returns final response.
    """

    BASE_SYSTEM = """You are EmpathyOS — a warm, emotionally intelligent AI companion.
Your responses should be:
- Genuinely empathetic and non-judgmental
- Concise (3-5 sentences max)
- End with ONE thoughtful follow-up question
- Never give medical advice
- Never use bullet points in your response"""

    def __init__(self):
        self.name = "SynthesisAgent"

    def run(
        self,
        user_text:    str,
        emotion:      str,
        memory_brief: dict,
        persona_brief:dict,
        guardian_brief:dict,
        chat_history: list,
    ) -> dict:

        # Guardian intercept — crisis detected
        if guardian_brief.get("crisis_detected"):
            return {
                "agent":          self.name,
                "response":       guardian_brief["crisis_response"],
                "guardian_flag":  True,
                "prompt_used":    None,
                "agents_used":    ["GuardianAgent (crisis intercept)"],
            }

        # Build enriched system prompt
        system_prompt = self._build_system(memory_brief, persona_brief, emotion)

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent context
        if memory_brief.get("recent_context"):
            messages.append({
                "role":    "system",
                "content": f"Recent session context:\n{memory_brief['recent_context']}"
            })

        # Add similar past memories
        if memory_brief.get("similar_memories"):
            mems = "\n".join(
                f"- [{m['emotion']}] {m['text']}"
                for m in memory_brief["similar_memories"][:2]
            )
            messages.append({
                "role":    "system",
                "content": f"Relevant past memories (use subtly if helpful):\n{mems}"
            })

        # Last 6 chat turns
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Current user message with emotion
        messages.append({
            "role":    "user",
            "content": f"[Detected emotion: {emotion}]\n{user_text}"
        })

        # Call Groq
        response_text = self._call_groq(messages)

        # Guardian output check
        output_check = GuardianAgent().check_output(response_text)
        if output_check["was_filtered"]:
            response_text = GuardianAgent().safe_fallback(emotion)

        return {
            "agent":         self.name,
            "response":      response_text,
            "guardian_flag": output_check["was_filtered"],
            "agents_used":   ["MemoryAgent", "PersonaAgent", "GuardianAgent", "SynthesisAgent"],
        }

    def _build_system(self, memory_brief: dict, persona_brief: dict, emotion: str) -> str:
        parts = [self.BASE_SYSTEM]

        # Persona instruction
        tone = persona_brief.get("tone_instruction", "")
        if tone:
            parts.append(f"\nTone instruction from Persona Agent: {tone}")

        # Emotion-specific instruction
        emotion_guides = {
            "anxious":  "Be calm, slow, grounding. Do not rush. Validate the anxiety first.",
            "sad":      "Be warm, gentle. Validate feelings before offering any perspective.",
            "angry":    "Be non-reactive, grounding. Acknowledge the frustration without amplifying it.",
            "stressed": "Be practical and focusing. Help narrow focus to one thing at a time.",
            "happy":    "Match the energy. Be celebratory and curious.",
            "excited":  "Be enthusiastic. Channel the energy productively.",
            "neutral":  "Be warm and gently curious. Invite the user to share more.",
        }
        guide = emotion_guides.get(emotion, "")
        if guide:
            parts.append(f"\nEmotion guide: {guide}")

        # Recurring pattern flag
        pattern = memory_brief.get("recurring_pattern")
        if pattern:
            parts.append(f"\nMemory Agent flag: {pattern}. Acknowledge this gently if appropriate.")

        return "\n".join(parts)

    def _call_groq(self, messages: list) -> str:
        if not GROQ_API_KEY:
            return self._fallback()
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json"
                },
                json={
                    "model":       MODEL,
                    "messages":    messages,
                    "max_tokens":  220,
                    "temperature": 0.8,
                },
                timeout=12
            )
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback()

    def _fallback(self) -> str:
        return (
            "I hear you, and I'm really glad you shared that with me. "
            "It takes courage to put feelings into words. "
            "What's been weighing on you most today?"
        )


# ════════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Main entry point
# ════════════════════════════════════════════════════════════════════════════════
class AgentOrchestrator:
    """
    Top-level orchestrator. Call run() from app.py.
    Returns final response + agent debug info.
    """

    def __init__(self, memory_engine):
        self.memory_agent   = MemoryAgent(memory_engine)
        self.persona_agent  = PersonaAgent()
        self.guardian_agent = GuardianAgent()
        self.synthesis_agent= SynthesisAgent()

    def run(
        self,
        user_text:    str,
        emotion:      str,
        chat_history: list,
    ) -> dict:
        """
        Full pipeline:
        1. Guardian checks user input for crisis signals
        2. Memory Agent retrieves context
        3. Persona Agent analyses style
        4. Synthesis Agent builds prompt + calls LLM
        5. Guardian checks output
        Returns: { response, agents_used, guardian_flag, debug }
        """

        # Step 1 — Guardian input check
        guardian_in = self.guardian_agent.check_input(user_text)

        # Step 2 — Memory Agent
        memory_brief = self.memory_agent.run(user_text, emotion)

        # Step 3 — Persona Agent
        persona_brief = self.persona_agent.run(user_text, chat_history)

        # Step 4+5 — Synthesis + Guardian output check
        result = self.synthesis_agent.run(
            user_text    = user_text,
            emotion      = emotion,
            memory_brief = memory_brief,
            persona_brief= persona_brief,
            guardian_brief=guardian_in,
            chat_history = chat_history,
        )

        return {
            "response":      result["response"],
            "agents_used":   result["agents_used"],
            "guardian_flag": result["guardian_flag"],
            "debug": {
                "memory":   memory_brief,
                "persona":  persona_brief,
                "guardian": guardian_in,
            }
        }

    def get_persona_profile(self) -> dict:
        return self.persona_agent._build_profile()

    def get_persona_tone(self) -> str:
        profile = self.persona_agent._build_profile()
        return self.persona_agent._tone_instruction(profile)