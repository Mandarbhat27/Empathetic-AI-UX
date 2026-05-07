import streamlit as st
import datetime
import json
import os
import random
from emotion_engine import detect_emotion, get_emotion_color, get_emotion_suggestions
from memory_engine import MemoryEngine
from llm_engine import get_empathetic_response
from dotenv import load_dotenv

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmpathyOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State Init ─────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = MemoryEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

# ─── Dynamic Theming ────────────────────────────────────────────────────────────
emotion = st.session_state.current_emotion
color_map = {
    "happy":    {"bg": "#FFF9E6", "accent": "#F5A623", "emoji": "😊"},
    "sad":      {"bg": "#EEF2FF", "accent": "#6B7FD7", "emoji": "😢"},
    "angry":    {"bg": "#FFF0F0", "accent": "#E05555", "emoji": "😠"},
    "anxious":  {"bg": "#F0FFF4", "accent": "#48BB78", "emoji": "😰"},
    "stressed": {"bg": "#FFF5F0", "accent": "#ED8936", "emoji": "😤"},
    "excited":  {"bg": "#FFFBEB", "accent": "#D69E2E", "emoji": "🤩"},
    "neutral":  {"bg": "#F8F9FA", "accent": "#667EEA", "emoji": "😐"},
}
theme = color_map.get(emotion, color_map["neutral"])

st.markdown(f"""
<style>
    .stApp {{ background-color: {theme['bg']}; }}
    .main-title {{
        font-size: 2.2rem; font-weight: 800;
        color: {theme['accent']}; margin-bottom: 0.2rem;
    }}
    .emotion-badge {{
        display: inline-block;
        background: {theme['accent']}22;
        color: {theme['accent']};
        padding: 4px 16px; border-radius: 20px;
        font-weight: 600; font-size: 0.95rem;
        border: 1.5px solid {theme['accent']};
        margin-bottom: 1rem;
    }}
    .chat-bubble-user {{
        background: {theme['accent']}22;
        border-left: 4px solid {theme['accent']};
        padding: 10px 16px; border-radius: 8px;
        margin: 6px 0; font-size: 0.95rem;
    }}
    .chat-bubble-ai {{
        background: white;
        border-left: 4px solid #ccc;
        padding: 10px 16px; border-radius: 8px;
        margin: 6px 0; font-size: 0.95rem;
    }}
    .suggestion-card {{
        background: white;
        border: 1.5px solid {theme['accent']}44;
        border-radius: 12px; padding: 12px;
        margin: 6px 0; font-size: 0.9rem;
    }}
    .stTextInput > div > div > input {{
        border: 2px solid {theme['accent']}66 !important;
        border-radius: 10px !important;
    }}
    .stButton > button {{
        background: {theme['accent']} !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
    }}
    div[data-testid="metric-container"] {{
        background: white;
        border-radius: 12px;
        padding: 12px;
        border: 1.5px solid {theme['accent']}33;
    }}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div class='main-title'>🧠 EmpathyOS</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='emotion-badge'>{theme['emoji']} Feeling: {emotion.capitalize()}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Your Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔥 Streak", f"{st.session_state.streak} days")
    with col2:
        st.metric("💬 Sessions", len(st.session_state.chat_history))

    st.markdown("---")
    st.markdown("### 🗓️ Mood History")
    if st.session_state.mood_log:
        for entry in st.session_state.mood_log[-5:][::-1]:
            e_theme = color_map.get(entry['emotion'], color_map['neutral'])
            st.markdown(
                f"<div style='font-size:0.82rem; color:#555;'>"
                f"{e_theme['emoji']} <b>{entry['emotion'].capitalize()}</b> — {entry['time']}"
                f"</div>", unsafe_allow_html=True
            )
    else:
        st.caption("No mood history yet. Start chatting!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown(f"<div class='main-title'>🧠 EmpathyOS</div>", unsafe_allow_html=True)
st.markdown(f"<div class='emotion-badge'>{theme['emoji']} Current Mood: {emotion.capitalize()}</div>", unsafe_allow_html=True)
st.markdown("*Your AI companion that truly understands how you feel*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["💬 Talk to EmpathyOS", "🌿 Suggestions", "📈 Mood Analytics"])

# ─── Tab 1: Chat ────────────────────────────────────────────────────────────────
with tab1:
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>🧑 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>EmpathyOS:</b> {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "How are you feeling right now?",
            placeholder="Type anything... I'm here to listen 💙",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([5, 1])
        with col2:
            submitted = st.form_submit_button("Send 💬")

    if submitted and user_input.strip():
        # Detect emotion
        detected = detect_emotion(user_input)
        st.session_state.current_emotion = detected

        # Log mood
        st.session_state.mood_log.append({
            "emotion": detected,
            "time": datetime.datetime.now().strftime("%I:%M %p"),
            "date": datetime.datetime.now().strftime("%b %d"),
            "text": user_input
        })

        # Save to memory
        st.session_state.memory.add(user_input, detected)

        # Get context from memory
        context = st.session_state.memory.get_context()

        # Get AI response
        response = get_empathetic_response(user_input, detected, context, st.session_state.chat_history)

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Update streak
        st.session_state.streak += 1

        st.rerun()

# ─── Tab 2: Suggestions ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🌿 Personalized Suggestions")
    st.markdown(f"Based on your current mood: **{emotion.capitalize()}** {theme['emoji']}")
    st.markdown("")

    suggestions = get_emotion_suggestions(emotion)
    icons = ["🎯", "🌱", "💡", "🎵", "🏃", "📖", "🧘", "💧"]

    for i, suggestion in enumerate(suggestions):
        icon = icons[i % len(icons)]
        st.markdown(
            f"<div class='suggestion-card'>{icon} {suggestion}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🆘 Quick Relief Techniques")
    technique_map = {
        "anxious":  "**Box Breathing:** Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4 times.",
        "stressed": "**5-4-3-2-1 Grounding:** Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "sad":      "**Gratitude Pause:** Write down 3 small things that went okay today, even tiny ones.",
        "angry":    "**Cooldown Walk:** Step away for 5 minutes. Physical movement resets emotional state.",
        "happy":    "**Savour It:** Write down what made you happy today — it reinforces positive neural pathways.",
        "excited":  "**Channel It:** Direct that energy into something creative — sketch, write, or build something.",
        "neutral":  "**Mindful Check-in:** Take 3 deep breaths and ask yourself what you truly need right now.",
    }
    st.info(technique_map.get(emotion, technique_map["neutral"]))

# ─── Tab 3: Analytics ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 Your Mood Journey")

    if len(st.session_state.mood_log) < 2:
        st.info("💡 Chat more to unlock your mood analytics! Needs at least 2 entries.")
    else:
        # Emotion frequency chart
        emotion_counts = {}
        for entry in st.session_state.mood_log:
            e = entry["emotion"]
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        st.markdown("#### Emotion Frequency")
        for em, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            em_theme = color_map.get(em, color_map["neutral"])
            bar_width = int((count / len(st.session_state.mood_log)) * 100)
            st.markdown(
                f"<div style='margin:4px 0;'>"
                f"{em_theme['emoji']} <b>{em.capitalize()}</b>: "
                f"<span style='display:inline-block; width:{bar_width}%; "
                f"background:{em_theme['accent']}; height:14px; border-radius:4px; "
                f"vertical-align:middle;'></span> {count}x"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### Recent Mood Timeline")
        for entry in st.session_state.mood_log[-8:][::-1]:
            e_theme = color_map.get(entry['emotion'], color_map['neutral'])
            st.markdown(
                f"<div style='padding:6px; border-left:3px solid {e_theme['accent']}; margin:4px 0;'>"
                f"{e_theme['emoji']} <b>{entry['emotion'].capitalize()}</b> at {entry['time']} — "
                f"<i style='color:#777;'>'{entry['text'][:60]}{'...' if len(entry['text'])>60 else ''}'</i>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Dominant mood insight
        dominant = max(emotion_counts, key=emotion_counts.get)
        st.markdown("---")
        st.markdown("#### 🔍 Insight")
        insight_map = {
            "happy":    "You've been in a great headspace! Keep doing what's working for you.",
            "sad":      "You've been carrying some heaviness. That's okay — reaching out is the first step.",
            "anxious":  "Anxiety has been present. Try grounding techniques and limit news/social media.",
            "stressed": "High stress detected. Consider breaking tasks into smaller steps today.",
            "angry":    "Frustration has been high. Physical activity can help release stored tension.",
            "excited":  "You're riding high energy! Channel it into something meaningful.",
            "neutral":  "You've been balanced. A good time to set intentions for the week.",
        }
        st.success(f"**Your dominant mood:** {color_map[dominant]['emoji']} {dominant.capitalize()} — {insight_map.get(dominant, '')}")
