"""
LLM Engine
Uses Groq API (free tier) — runs fast on any laptop, no GPU needed
Get free API key at: https://console.groq.com
"""

import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set in .env file
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"  # Free on Groq


EMPATHY_SYSTEM_PROMPT = """You are EmpathyOS — a warm, emotionally intelligent AI companion.
Your role is to:
1. Acknowledge the user's current emotional state with genuine empathy
2. Respond in a supportive, non-judgmental way
3. Ask ONE thoughtful follow-up question to understand better
4. Keep responses concise (3-5 sentences max)
5. Never give medical advice — suggest professional help only when appropriate
6. Adapt your tone: calm for anxiety, warm for sadness, grounding for anger, celebratory for happiness

You have memory of past conversations — use this context naturally without being creepy about it.
Do NOT list suggestions here — those are shown separately in the UI.
"""


def get_empathetic_response(
    user_text: str,
    emotion: str,
    memory_context: str,
    chat_history: list
) -> str:
    """
    Call Groq API for an empathetic response.
    Falls back to rule-based response if API key not set.
    """
    if not GROQ_API_KEY:
        return _fallback_response(emotion, user_text)

    # Build messages
    messages = [{"role": "system", "content": EMPATHY_SYSTEM_PROMPT}]

    # Add memory context if available
    if memory_context:
        messages.append({
            "role": "system",
            "content": f"Recent conversation context:\n{memory_context}"
        })

    # Add last 6 chat messages for continuity
    for msg in chat_history[-6:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add current emotion as context
    messages.append({
        "role": "user",
        "content": f"[Detected emotion: {emotion}]\n{user_text}"
    })

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.8
            },
            timeout=10
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return _fallback_response(emotion, user_text)


def _fallback_response(emotion: str, text: str) -> str:
    """Rule-based fallback when no API key is set"""
    responses = {
        "happy": [
            "That's wonderful to hear! 😊 Your positive energy is contagious. What's been the highlight of your day so far?",
            "It's great that you're feeling this way! Happiness like this is worth savouring. What's bringing you this joy?",
        ],
        "sad": [
            "I hear you, and I'm really glad you shared that with me. 💙 Feeling sad is completely valid. Would you like to talk about what's weighing on you?",
            "It sounds like you're going through something tough right now. You don't have to face it alone. What's on your mind?",
        ],
        "angry": [
            "That frustration sounds completely valid. 💪 It's okay to feel angry — what matters is what we do with it. What triggered this for you?",
            "I can feel the intensity in what you're sharing. Let's slow down together. What's the core of what's upsetting you?",
        ],
        "anxious": [
            "Anxiety can feel overwhelming, but you're doing the right thing by acknowledging it. 🌿 Take a slow breath with me. What's your biggest worry right now?",
            "That feeling of uncertainty is so hard to sit with. You're not alone in this. What's your mind circling around most?",
        ],
        "stressed": [
            "Sounds like you have a lot on your plate right now. 🌱 Let's take this one step at a time. What feels most urgent to you?",
            "That's a heavy load you're carrying. It's okay to pause and breathe. What would make the biggest difference for you today?",
        ],
        "excited": [
            "Your excitement is electric! 🎉 Love this energy! What's got you so pumped up?",
            "This is amazing — I can feel your enthusiasm! Tell me everything, what's happening?",
        ],
        "neutral": [
            "Thanks for checking in. 😊 Sometimes neutral is actually a great place to be — clear and grounded. How's your day going overall?",
            "I'm here with you. A calm moment can be a good moment. What's on your mind today?",
        ]
    }
    import random
    options = responses.get(emotion, responses["neutral"])
    return random.choice(options)
