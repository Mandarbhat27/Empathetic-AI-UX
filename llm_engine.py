"""
LLM Engine — v2
Uses Groq API (free tier, Llama3-8B) for empathetic responses
Now uses semantic memory context from ChromaDB
Get free API key: https://console.groq.com
"""

import os
import random
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama3-8b-8192"

EMPATHY_SYSTEM_PROMPT = """You are EmpathyOS — a warm, emotionally intelligent AI companion built for everyday life.

Your role:
1. Acknowledge the user's current emotional state with genuine empathy
2. Respond in a supportive, non-judgmental, conversational way
3. Ask ONE thoughtful follow-up question to understand better
4. Keep responses concise — 3 to 5 sentences max
5. Never give medical advice — suggest professional help only when truly appropriate
6. Adapt your tone:
   - Anxious → calm, slow, grounding
   - Sad → warm, gentle, validating
   - Angry → grounding, non-reactive
   - Happy/Excited → celebratory, energetic
   - Stressed → practical, focusing

Use the memory context naturally — like a friend who remembers, not a database.
Do NOT list suggestions — those appear separately in the UI.
Do NOT use bullet points in your response."""


def get_empathetic_response(
    user_text: str,
    emotion: str,
    memory_context: str,
    chat_history: list,
    similar_memories: list = None
) -> str:
    if not GROQ_API_KEY:
        return _fallback_response(emotion)

    messages = [{"role": "system", "content": EMPATHY_SYSTEM_PROMPT}]

    # Add recent session context
    if memory_context:
        messages.append({
            "role": "system",
            "content": f"Recent conversation context:\n{memory_context}"
        })

    # Add semantically similar past memories if available
    if similar_memories:
        mem_lines = "\n".join(
            f"- [{m['emotion']}] {m['text']}" for m in similar_memories[:3]
        )
        messages.append({
            "role": "system",
            "content": f"Relevant past memories (use subtly if helpful):\n{mem_lines}"
        })

    # Last 6 turns for continuity
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Current message with detected emotion
    messages.append({
        "role": "user",
        "content": f"[Detected emotion: {emotion}]\n{user_text}"
    })

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json"
            },
            json={
                "model":       MODEL,
                "messages":    messages,
                "max_tokens":  220,
                "temperature": 0.8
            },
            timeout=12
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return _fallback_response(emotion)


def _fallback_response(emotion: str) -> str:
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
            "That frustration sounds completely valid. 💪 It's okay to feel angry. What triggered this for you?",
            "I can feel the intensity in what you're sharing. Let's slow down together. What's the core of what's upsetting you?",
        ],
        "anxious": [
            "Anxiety can feel overwhelming, but you're doing the right thing by acknowledging it. 🌿 Take a slow breath. What's your biggest worry right now?",
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
            "Thanks for checking in. 😊 A calm moment can be a good moment. What's on your mind today?",
            "I'm here with you. How's your day going overall?",
        ]
    }
    return random.choice(responses.get(emotion, responses["neutral"]))
