"""
Emotion Detection Engine
Uses rule-based keyword matching + optional transformer model
Runs 100% locally on CPU — no GPU needed
"""

import re

# ─── Keyword-based emotion lexicon ─────────────────────────────────────────────
EMOTION_KEYWORDS = {
    "happy": [
        "happy", "great", "wonderful", "amazing", "fantastic", "joyful", "excited",
        "blessed", "grateful", "love", "awesome", "good", "glad", "cheerful",
        "thrilled", "elated", "pleased", "delighted", "smile", "laugh", "fun",
        "celebrate", "winning", "success", "proud", "confident", "enjoy"
    ],
    "sad": [
        "sad", "unhappy", "depressed", "miserable", "crying", "tears", "grief",
        "heartbroken", "lonely", "alone", "hopeless", "helpless", "empty",
        "worthless", "hurt", "pain", "lost", "miss", "gone", "failure", "awful",
        "terrible", "horrible", "bad", "down", "low", "blue", "sorrow"
    ],
    "angry": [
        "angry", "furious", "mad", "rage", "hate", "annoyed", "frustrated",
        "irritated", "livid", "outraged", "fed up", "sick of", "tired of",
        "pissed", "upset", "bitter", "resentful", "betrayed", "unfair", "wrong"
    ],
    "anxious": [
        "anxious", "anxiety", "worried", "nervous", "scared", "fear", "panic",
        "overthinking", "dread", "uneasy", "tense", "overwhelmed", "stress",
        "what if", "cant sleep", "can't sleep", "racing thoughts", "unsure",
        "uncertain", "doubt", "afraid", "terrified", "paranoid"
    ],
    "stressed": [
        "stressed", "stress", "pressure", "deadline", "too much", "overwhelmed",
        "exhausted", "burnout", "burn out", "tired", "no time", "busy",
        "overloaded", "swamped", "can't cope", "cant cope", "falling behind",
        "behind schedule", "workload", "hectic", "chaos", "mess"
    ],
    "excited": [
        "excited", "thrilled", "pumped", "hyped", "cant wait", "can't wait",
        "looking forward", "amazing news", "great news", "so good", "wow",
        "incredible", "unbelievable", "new job", "promotion", "trip", "vacation",
        "adventure", "opportunity", "dream", "finally"
    ]
}

NEGATION_WORDS = ["not", "no", "never", "don't", "dont", "didn't", "didnt", "isn't", "isnt", "wasn't", "wasnt"]

def detect_emotion(text: str) -> str:
    """
    Detect emotion from text using weighted keyword matching.
    Returns: emotion string
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    # Check for negations
    has_negation = any(neg in words for neg in NEGATION_WORDS)

    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Multi-word keyword
                if " " in keyword:
                    scores[emotion] += 2
                else:
                    scores[emotion] += 1

    # Negation flips happy ↔ sad
    if has_negation:
        if scores["happy"] > 0:
            scores["sad"] += scores["happy"]
            scores["happy"] = 0

    # Get highest scoring emotion
    max_score = max(scores.values())
    if max_score == 0:
        return "neutral"

    # If tie, prioritize by severity
    priority = ["angry", "anxious", "stressed", "sad", "excited", "happy"]
    for p in priority:
        if scores[p] == max_score:
            return p

    return max(scores, key=scores.get)


def get_emotion_color(emotion: str) -> str:
    colors = {
        "happy":   "#F5A623",
        "sad":     "#6B7FD7",
        "angry":   "#E05555",
        "anxious": "#48BB78",
        "stressed":"#ED8936",
        "excited": "#D69E2E",
        "neutral": "#667EEA",
    }
    return colors.get(emotion, "#667EEA")


def get_emotion_suggestions(emotion: str) -> list:
    suggestions = {
        "happy": [
            "Share your positive energy — call a friend or write about what's going well",
            "This is a great time to tackle that task you've been putting off",
            "Express gratitude — write 3 things you're thankful for today",
            "Do something creative — your energy is flowing!",
            "Spread the joy — a kind message to someone can make their day too",
        ],
        "sad": [
            "Be gentle with yourself — it's okay to feel this way",
            "Talk to someone you trust — connection heals",
            "Step outside for even 10 minutes — fresh air helps",
            "Watch or read something comforting, not numbing",
            "Write your feelings — journaling releases emotional weight",
            "Drink water and eat something nourishing",
        ],
        "angry": [
            "Take a 10-minute walk before responding to anything",
            "Write what's making you angry — then don't send it",
            "Do 10 jumping jacks or push-ups — move the energy out",
            "Wait 20 minutes before any major decisions",
            "Identify what boundary was crossed — that's the real source",
        ],
        "anxious": [
            "Try box breathing: 4s in, 4s hold, 4s out, 4s hold",
            "Write down your worry — then write the realistic outcome",
            "Limit social media and news for the next 2 hours",
            "Do a 5-minute body scan meditation",
            "Focus only on what you can control right now",
            "Call or text someone grounding to you",
        ],
        "stressed": [
            "Write down everything on your mind — brain dump it all",
            "Pick ONE priority for today and ignore the rest",
            "Take a 5-minute break — step away completely",
            "Break your biggest task into 3 tiny steps",
            "Say no to one non-essential thing today",
            "Drink water — dehydration amplifies stress",
        ],
        "excited": [
            "Channel this energy into your most important goal today",
            "Share your excitement — enthusiasm is contagious!",
            "Write down your ideas before the energy fades",
            "Start that project you've been dreaming about",
            "Use this momentum to build a new habit",
        ],
        "neutral": [
            "A calm mind is a creative mind — try journaling",
            "This is a good time to plan or organize something",
            "Learn something new — your mind is clear and ready",
            "Check in with someone you haven't spoken to in a while",
            "Set one intention for the rest of the day",
        ]
    }
    return suggestions.get(emotion, suggestions["neutral"])
