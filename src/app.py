"""
EmpathyOS v4 — Features 2 & 4 integrated
  Feature 2: Multi-Agent Orchestration (Memory + Persona + Guardian + Synthesis)
  Feature 4: Predictive Proactive Empathy (calendar + circadian pattern)
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import datetime
from emotion_engine     import detect_emotion, get_emotion_suggestions, get_detection_method
from memory_engine      import MemoryEngine
from pdf_engine         import generate_mood_report
from voice_engine       import transcribe_audio, analyse_vocal_tone, check_dependencies
from face_engine        import detect_emotion_from_image, annotate_image, check_face_dependencies
from checkin_engine     import CheckinEngine
from agent_orchestrator import AgentOrchestrator
from predictive_engine  import PredictiveEmpathyEngine

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmpathyOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ────────────────────────────────────────────────────────────────
DEFAULTS = {
    "memory":            None,
    "checkin":           None,
    "orchestrator":      None,
    "predictor":         None,
    "chat_history":      [],
    "current_emotion":   "neutral",
    "streak":            0,
    "mood_log":          [],
    "input_mode":        "text",
    "dismissed_nudge":   False,
    "dismissed_predict": False,
    "agent_debug":       None,
    "prediction_cache":  None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Init engines
if st.session_state.memory is None:
    st.session_state.memory = MemoryEngine()
if st.session_state.checkin is None:
    st.session_state.checkin = CheckinEngine()
if st.session_state.orchestrator is None:
    st.session_state.orchestrator = AgentOrchestrator(st.session_state.memory)
if st.session_state.predictor is None:
    st.session_state.predictor = PredictiveEmpathyEngine(st.session_state.checkin)

# ── Theme ────────────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "happy":   {"bg":"#FFFBEB","accent":"#F5A623","emoji":"😊"},
    "sad":     {"bg":"#EEF2FF","accent":"#6B7FD7","emoji":"😢"},
    "angry":   {"bg":"#FFF0F0","accent":"#E05555","emoji":"😠"},
    "anxious": {"bg":"#F0FFF4","accent":"#48BB78","emoji":"😰"},
    "stressed":{"bg":"#FFF5F0","accent":"#ED8936","emoji":"😤"},
    "excited": {"bg":"#FEFCE8","accent":"#D69E2E","emoji":"🤩"},
    "neutral": {"bg":"#F8F9FA","accent":"#667EEA","emoji":"😐"},
}
emotion = st.session_state.current_emotion
theme   = COLOR_MAP.get(emotion, COLOR_MAP["neutral"])

st.markdown(f"""
<style>
  .stApp {{ background-color:{theme['bg']}; }}
  .hero {{ background:linear-gradient(135deg,{theme['accent']}22,{theme['bg']});
    border:1.5px solid {theme['accent']}44; border-radius:16px;
    padding:18px 24px; margin-bottom:14px; }}
  .hero-title {{ font-size:1.9rem; font-weight:900; color:#1A1A2E; margin:0; }}
  .hero-sub   {{ font-size:0.9rem; color:#555; margin:4px 0 0; }}
  .pill {{ display:inline-block; background:{theme['accent']}22; color:{theme['accent']};
    padding:3px 14px; border-radius:20px; font-weight:700; font-size:0.82rem;
    border:1.5px solid {theme['accent']}66; }}
  .chat-user {{ background:{theme['accent']}18; border-left:4px solid {theme['accent']};
    padding:10px 14px; border-radius:0 10px 10px 0; margin:5px 0; }}
  .chat-ai {{ background:white; border-left:4px solid #CBD5E1;
    padding:10px 14px; border-radius:0 10px 10px 0; margin:5px 0; }}
  .card {{ background:white; border:1.5px solid {theme['accent']}33;
    border-radius:12px; padding:14px; margin:6px 0; font-size:0.9rem; }}
  .nudge {{ background:#FFF9C4; border:1.5px solid #F59E0B;
    border-radius:12px; padding:12px 16px; margin-bottom:10px; font-size:0.9rem; }}
  .predict-banner {{ background:#EEF2FF; border:1.5px solid #6B7FD7;
    border-radius:12px; padding:14px 18px; margin-bottom:10px; }}
  .agent-badge {{ display:inline-block; background:{theme['accent']}15;
    border:1px solid {theme['accent']}44; border-radius:20px;
    padding:2px 10px; font-size:0.75rem; color:{theme['accent']}; margin:2px; }}
  .stButton>button {{ background:{theme['accent']} !important; color:white !important;
    border:none !important; border-radius:10px !important; font-weight:600 !important; }}
  div[data-testid="metric-container"] {{ background:white; border-radius:12px;
    border:1.5px solid {theme['accent']}33; padding:10px; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PREDICTIVE BANNER — shows if tomorrow looks stressful
# ════════════════════════════════════════════════════════════════════════════════
if not st.session_state.dismissed_predict:
    # Cache prediction so it doesn't recalculate every rerun
    if st.session_state.prediction_cache is None:
        st.session_state.prediction_cache = st.session_state.predictor.predict()

    pred = st.session_state.prediction_cache
    if pred["should_warn"]:
        col1, col2 = st.columns([11, 1])
        with col1:
            level_emoji = {"high":"🔴","moderate":"🟡","low":"🟢","minimal":"⚪"}.get(pred["risk_level"],"🟡")
            st.markdown(f"""
            <div class='predict-banner'>
              <b>{level_emoji} Tomorrow Forecast:</b> {pred['warning_message']}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✕", key="dismiss_predict"):
                st.session_state.dismissed_predict = True
                st.rerun()


# ── Proactive nudge banner ───────────────────────────────────────────────────────
if not st.session_state.dismissed_nudge:
    should, msg = st.session_state.checkin.should_checkin()
    if should and msg:
        col1, col2 = st.columns([11, 1])
        with col1:
            st.markdown(f"<div class='nudge'>💡 {msg}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("✕", key="dismiss_nudge"):
                st.session_state.dismissed_nudge = True
                st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0;">
      <div style="font-size:2rem;">🧠</div>
      <div style="font-size:1.3rem;font-weight:800;color:#1A1A2E;">EmpathyOS</div>
      <div style="font-size:0.75rem;color:#778899;">Multi-Agent Emotional AI</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:center;margin:6px 0;'><span class='pill'>{theme['emoji']} {emotion.capitalize()}</span></div>", unsafe_allow_html=True)
    st.markdown("---")

    mem = st.session_state.memory.get_storage_info()
    c1, c2 = st.columns(2)
    c1.metric("🔥 Streak",   f"{st.session_state.streak}d")
    c2.metric("💬 Chats",    len(st.session_state.chat_history) // 2)
    c3, c4 = st.columns(2)
    c3.metric("🧠 Memories", mem["total_entries"])
    c4.metric("🎭 Persona",  f"{st.session_state.orchestrator.get_persona_profile()['sessions']}s")

    st.markdown("---")

    # Agent status
    st.markdown("**🤖 Active Agents**")
    agents = ["🧠 Memory Agent","🎭 Persona Agent","🛡️ Guardian Agent","🔀 Synthesis Agent"]
    for a in agents:
        st.markdown(f"<div style='font-size:0.8rem;color:#10B981;padding:2px 0;'>● {a}</div>", unsafe_allow_html=True)

    # Persona profile
    st.markdown("---")
    st.markdown("**🎭 Persona Profile**")
    profile = st.session_state.orchestrator.get_persona_profile()
    tone    = st.session_state.orchestrator.get_persona_tone()
    st.markdown(f"""
    <div class='card' style='font-size:0.8rem;padding:10px;'>
      Sessions: {profile['sessions']}<br>
      Style: {'Formal' if profile['is_formal'] else 'Casual'}<br>
      Emojis: {'Yes' if profile['prefers_emoji'] else 'No'}<br>
      Slang: {'Yes' if profile['uses_slang'] else 'No'}<br>
      Avg msg: {profile['avg_msg_length']} words
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Mood history
    st.markdown("**🗓️ Recent Moods**")
    if st.session_state.mood_log:
        for e in st.session_state.mood_log[-5:][::-1]:
            ec = COLOR_MAP.get(e["emotion"], COLOR_MAP["neutral"])
            st.markdown(f"<div style='font-size:0.8rem;color:#555;padding:2px 0;'>{ec['emoji']} <b>{e['emotion'].capitalize()}</b> · {e['time']}</div>", unsafe_allow_html=True)
    else:
        st.caption("Start chatting!")

    st.markdown("---")
    st.session_state.input_mode = st.radio(
        "Mode", ["text","voice","face"],
        format_func=lambda x: {"text":"⌨️ Text","voice":"🎙️ Voice","face":"📷 Face"}[x],
        label_visibility="collapsed"
    )
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🧠 EmpathyOS {theme['emoji']}</div>
  <div class="hero-sub">Multi-Agent AI · Current mood: <b>{emotion}</b> · 4 agents active</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat", "🌿 Suggestions", "🔮 Tomorrow", "📈 Analytics", "📄 Report", "⚙️ System"
])


# ════════════════════════════════════════════════════════════════════════════════
# SHARED INPUT PROCESSOR — now using AgentOrchestrator
# ════════════════════════════════════════════════════════════════════════════════
def process_input(text: str, source: str = "text", extra_emotion: str = None):
    text_em  = detect_emotion(text)
    final_em = _fuse(text_em, extra_emotion)

    st.session_state.current_emotion = final_em
    st.session_state.mood_log.append({
        "emotion": final_em, "source": source,
        "time":    datetime.datetime.now().strftime("%I:%M %p"),
        "date":    datetime.datetime.now().strftime("%b %d"),
        "text":    text,
    })
    st.session_state.streak += 1
    st.session_state.checkin.record_session(final_em)
    st.session_state.memory.add(text, final_em)

    # ── MULTI-AGENT PIPELINE ──────────────────────────────────────────────────
    result = st.session_state.orchestrator.run(
        user_text    = text,
        emotion      = final_em,
        chat_history = st.session_state.chat_history,
    )

    st.session_state.chat_history.append({"role": "user",     "content": text})
    st.session_state.chat_history.append({"role": "assistant","content": result["response"]})
    st.session_state.agent_debug = result
    st.rerun()


def _fuse(text_em: str, other_em: str) -> str:
    if other_em is None or other_em == text_em:
        return text_em
    for p in ["angry","anxious","sad","stressed","excited","happy","neutral"]:
        if p in [text_em, other_em]:
            return p
    return text_em


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    # Show agent debug info if available
    if st.session_state.agent_debug:
        d = st.session_state.agent_debug
        agents_str = " · ".join(d.get("agents_used", []))
        flag_str   = " 🛡️ Guardian filtered" if d.get("guardian_flag") else ""
        st.markdown(
            f"<div style='font-size:0.75rem;color:#888;margin-bottom:6px;'>"
            f"🤖 Last response via: {agents_str}{flag_str}</div>",
            unsafe_allow_html=True
        )

    # Chat history
    for msg in st.session_state.chat_history:
        cls  = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='{cls}'>{icon} {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TEXT MODE
    if st.session_state.input_mode == "text":
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", placeholder="How are you feeling? 💙", label_visibility="collapsed")
            _, c2 = st.columns([5, 1])
            with c2:
                sent = st.form_submit_button("Send 💬")
        if sent and user_input.strip():
            process_input(user_input, source="text")

    # VOICE MODE
    elif st.session_state.input_mode == "voice":
        st.markdown("### 🎙️ Voice Input")
        audio_file = st.file_uploader("Upload audio", type=["wav","mp3","ogg","m4a"], label_visibility="collapsed")
        if audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            if st.button("🔊 Transcribe & Send"):
                with st.spinner("Transcribing..."):
                    result = transcribe_audio(audio_bytes)
                if result["error"]:
                    st.error(f"❌ {result['error']}")
                else:
                    tone_info = analyse_vocal_tone(audio_bytes)
                    voice_em  = None
                    if not tone_info.get("error"):
                        tone_map = {"energetic":"excited","assertive":"stressed","calm":"neutral","quiet":"sad"}
                        voice_em = tone_map.get(tone_info.get("tone_hint",""), None)
                        st.caption(f"🎵 Tone: **{tone_info['tone_hint']}** · {tone_info['duration_sec']}s")
                    st.success(f"📝 *{result['text']}*")
                    if result["text"]:
                        process_input(result["text"], source="voice", extra_emotion=voice_em)

    # FACE MODE
    else:
        st.markdown("### 📷 Face Emotion")
        face_deps = check_face_dependencies()
        if not face_deps["deepface"]:
            st.warning("Run: `pip install deepface tf-keras opencv-python-headless`")
        else:
            src = st.radio("Source", ["📁 Upload","📷 Webcam"], label_visibility="collapsed")
            image_bytes = None
            if src == "📁 Upload":
                up = st.file_uploader("Photo", type=["jpg","jpeg","png"], label_visibility="collapsed")
                if up: image_bytes = up.read()
            else:
                cam = st.camera_input("Snap", label_visibility="collapsed")
                if cam: image_bytes = cam.read()

            if image_bytes:
                with st.spinner("Analysing face..."):
                    fr = detect_emotion_from_image(image_bytes)
                if fr["error"] and not fr["face_detected"]:
                    st.error(fr["error"])
                else:
                    ann = annotate_image(image_bytes, fr["emotion"], fr["confidence"])
                    st.image(ann, caption=f"{fr['emotion'].capitalize()} ({fr['confidence']:.0f}%)", use_column_width=True)
                    with st.form("face_form", clear_on_submit=True):
                        face_text = st.text_input("Add context", placeholder=f"Tell me about feeling {fr['emotion']}...", label_visibility="collapsed")
                        _, fc2 = st.columns([5,1])
                        with fc2: fs = st.form_submit_button("Send 💬")
                    if fs:
                        process_input(face_text or f"I'm feeling {fr['emotion']}", source="face", extra_emotion=fr["emotion"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUGGESTIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### 🌿 Suggestions for: {theme['emoji']} {emotion.capitalize()}")
    icons = ["🎯","🌱","💡","🎵","🏃","📖","🧘","💧"]
    for i, s in enumerate(get_emotion_suggestions(emotion)):
        st.markdown(f"<div class='card'>{icons[i%len(icons)]} {s}</div>", unsafe_allow_html=True)

    st.markdown("---")
    technique_map = {
        "anxious":  ("Box Breathing",        "Inhale 4s → Hold 4s → Exhale 4s → Hold 4s"),
        "stressed": ("5-4-3-2-1 Grounding",  "5 see · 4 touch · 3 hear · 2 smell · 1 taste"),
        "sad":      ("Gratitude Pause",       "Write 3 small things that went okay today"),
        "angry":    ("Cooldown Walk",         "Step away 5 min — physical movement resets stress hormones"),
        "happy":    ("Savour It",             "Write what made you happy — reinforces neural pathways"),
        "excited":  ("Channel It",            "Direct the energy into something creative right now"),
        "neutral":  ("Mindful Check-in",      "3 deep breaths — what do you truly need right now?"),
    }
    title, desc = technique_map.get(emotion, technique_map["neutral"])
    st.info(f"**{title}**\n\n{desc}")

    # Persona-aware tip
    profile = st.session_state.orchestrator.get_persona_profile()
    if profile["sessions"] >= 3:
        st.markdown("---")
        st.markdown("**🎭 Personalised for your style**")
        tone = st.session_state.orchestrator.get_persona_tone()
        st.markdown(f"<div class='card' style='font-size:0.85rem;'>Based on {profile['sessions']} sessions, EmpathyOS knows you prefer: {tone}</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOMORROW (PREDICTIVE EMPATHY)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔮 Tomorrow's Forecast")
    st.markdown("*Based on your calendar + emotional patterns*")

    if st.button("🔄 Refresh Prediction"):
        st.session_state.prediction_cache = None
        st.session_state.dismissed_predict = False
        st.rerun()

    if st.session_state.prediction_cache is None:
        with st.spinner("Analysing tomorrow..."):
            st.session_state.prediction_cache = st.session_state.predictor.predict()

    pred = st.session_state.prediction_cache

    # Risk level display
    level_map = {
        "high":    ("🔴","High stress risk","#EF4444"),
        "moderate":("🟡","Moderate stress risk","#F59E0B"),
        "low":     ("🟢","Low stress risk","#10B981"),
        "minimal": ("⚪","Minimal stress risk","#94A3B8"),
    }
    emoji_l, label_l, color_l = level_map.get(pred["risk_level"], level_map["low"])
    risk_pct = pred["risk_score"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level",    f"{emoji_l} {label_l}")
    col2.metric("Risk Score",    f"{risk_pct}/100")
    col3.metric("Meetings",      pred["schedule_analysis"].get("total_meetings", 0))

    # Schedule breakdown
    sched = pred["schedule_analysis"]
    st.markdown("---")
    st.markdown("#### 📅 Tomorrow's Schedule")
    st.markdown(f"""
    <div class='card'>
      <b>Density:</b> {sched.get('density','?').capitalize()}<br>
      <b>Total meetings:</b> {sched.get('total_meetings',0)}<br>
      <b>Back-to-back blocks:</b> {sched.get('back_to_back',0)}<br>
      <b>Free blocks (&gt;45 min):</b> {sched.get('free_blocks',0)}
    </div>""", unsafe_allow_html=True)

    # Meeting list
    meetings = sched.get("meetings", [])
    if meetings:
        st.markdown("**Scheduled meetings:**")
        for m in meetings:
            st.markdown(
                f"<div style='font-size:0.85rem;padding:3px 0;color:#555;'>"
                f"🕐 {m.get('start_time','')}–{m.get('end_time','')} · <b>{m.get('title','')}</b></div>",
                unsafe_allow_html=True
            )

    # History pattern
    hist = pred["history_pattern"]
    if hist.get("pattern_found"):
        st.markdown("---")
        st.markdown("#### 📊 Historical Pattern")
        st.markdown(f"""
        <div class='card'>
          On <b>{hist.get('day_name','')}</b> you typically feel
          <b style='color:{COLOR_MAP.get(hist["typical_emotion"],COLOR_MAP["neutral"])["accent"]};'>
          {hist.get('typical_emotion','').capitalize()}</b>
          ({hist.get('confidence',0)}% of the time, based on {hist.get('sample_size',0)} sessions)
        </div>""", unsafe_allow_html=True)

    # Prep kit
    st.markdown("---")
    st.markdown("#### 🎯 Tonight's Prep Kit")
    for item in pred["prep_kit"]:
        st.markdown(f"""
        <div class='card'>
          <b>{item['icon']} {item['title']}</b><br>
          <span style='font-size:0.85rem;color:#555;'>{item['desc']}</span>
        </div>""", unsafe_allow_html=True)

    # Edit calendar tip
    st.markdown("---")
    st.markdown("""
    <div class='card' style='font-size:0.82rem;border-color:#6B7FD7;'>
    📝 <b>Customise your schedule:</b> Edit <code>mock_calendar.json</code> in your project folder
    to add your real meetings. Refresh the prediction after saving.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Mood Journey")
    log = st.session_state.mood_log

    if len(log) < 2:
        st.info("💡 Need at least 2 check-ins to show analytics.")
    else:
        counts  = {}
        for e in log:
            counts[e["emotion"]] = counts.get(e["emotion"],0) + 1
        total    = len(log)
        dominant = max(counts, key=counts.get)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Check-ins",    total)
        c2.metric("Dominant",     f"{COLOR_MAP[dominant]['emoji']} {dominant.capitalize()}")
        c3.metric("Unique moods", len(counts))
        c4.metric("🔥 Streak",    f"{st.session_state.streak}d")

        st.markdown("---")
        st.markdown("#### Emotion Frequency")
        for em, count in sorted(counts.items(), key=lambda x:-x[1]):
            ec  = COLOR_MAP.get(em, COLOR_MAP["neutral"])
            pct = int(count/total*100)
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin:5px 0;'>
              <span style='width:110px;font-size:0.9rem;'>{ec['emoji']} {em.capitalize()}</span>
              <div style='flex:1;background:#eee;border-radius:6px;height:16px;'>
                <div style='width:{pct}%;background:{ec['accent']};height:16px;border-radius:6px;'></div>
              </div>
              <span style='width:60px;font-size:0.82rem;color:#555;'>{count}x · {pct}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Recent Timeline")
        for entry in log[-8:][::-1]:
            ec   = COLOR_MAP.get(entry["emotion"],COLOR_MAP["neutral"])
            prev = entry["text"][:60]+("..." if len(entry["text"])>60 else "")
            st.markdown(f"""
            <div style='padding:7px 12px;border-left:3px solid {ec['accent']};
                        margin:4px 0;background:white;border-radius:0 8px 8px 0;'>
              <b style='color:{ec['accent']};'>{ec['emoji']} {entry['emotion'].capitalize()}</b>
              <span style='color:#999;font-size:0.78rem;margin-left:8px;'>{entry['time']}</span>
              <div style='color:#555;font-size:0.85rem;margin-top:2px;'><i>{prev}</i></div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — PDF REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📄 Export Mood Report")
    if not st.session_state.mood_log:
        st.info("💡 Start chatting to generate a report!")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Check-ins", len(st.session_state.mood_log))
        c2.metric("Streak",    f"{st.session_state.streak}d 🔥")
        c3.metric("Sessions",  len(st.session_state.chat_history)//2)
        if st.button("📥 Generate & Download PDF"):
            with st.spinner("Building report..."):
                try:
                    pdf = generate_mood_report(
                        st.session_state.mood_log,
                        st.session_state.streak,
                        len(st.session_state.chat_history)//2
                    )
                    fname = f"EmpathyOS_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button("⬇️ Download PDF", pdf, fname, "application/pdf")
                    st.success("✅ Ready!")
                except Exception as e:
                    st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — SYSTEM
# ════════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### ⚙️ System Status")

    face_deps  = check_face_dependencies()
    voice_deps = check_dependencies()
    mem_info   = st.session_state.memory.get_storage_info()

    # Agent statuses
    st.markdown("**🤖 Multi-Agent System (Feature 2)**")
    agent_info = [
        ("🧠 Memory Agent",    True,  "Retrieves context + similar past memories"),
        ("🎭 Persona Agent",   True,  "Tracks communication style · evolves AI personality"),
        ("🛡️ Guardian Agent",  True,  "Safety checker on every input + output"),
        ("🔀 Synthesis Agent", True,  "Orchestrator · builds enriched prompt · calls Groq"),
    ]
    for name, status, desc in agent_info:
        color = "#059669" if status else "#DC2626"
        badge = "✅ Active" if status else "❌ Inactive"
        st.markdown(f"""
        <div class='card' style='padding:10px;'>
          <div style='display:flex;justify-content:space-between;'>
            <span style='font-weight:600;'>{name}</span>
            <span style='color:{color};font-size:0.82rem;font-weight:700;'>{badge}</span>
          </div>
          <div style='font-size:0.8rem;color:#777;margin-top:3px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Predictive status
    st.markdown("---")
    st.markdown("**🔮 Predictive Empathy (Feature 4)**")
    pred = st.session_state.prediction_cache or {}
    pred_active = pred.get("risk_level") is not None
    st.markdown(f"""
    <div class='card' style='padding:10px;'>
      <div style='display:flex;justify-content:space-between;'>
        <span style='font-weight:600;'>📅 Calendar Engine</span>
        <span style='color:#059669;font-size:0.82rem;font-weight:700;'>✅ Active</span>
      </div>
      <div style='font-size:0.8rem;color:#777;margin-top:3px;'>
        Reads mock_calendar.json · detects density · predicts stress risk
      </div>
    </div>""", unsafe_allow_html=True)

    # Other components
    st.markdown("---")
    st.markdown("**🔬 Detection Engines**")
    components = [
        ("📝 Text Emotion (DistilBERT)",  True,                            "Already active"),
        ("📷 Face Emotion (DeepFace)",     face_deps["deepface"],            "pip install deepface tf-keras opencv-python-headless"),
        ("🎙️ Voice (SpeechRecognition)",  voice_deps["speech_recognition"], "pip install SpeechRecognition"),
        ("🧠 ChromaDB Memory",             "ChromaDB" in mem_info["backend"],"pip install chromadb sentence-transformers"),
        ("📄 PDF Export",                  True,                            "Already active"),
    ]
    for name, status, cmd in components:
        color = "#059669" if status else "#DC2626"
        badge = "✅ Active" if status else "❌ Not installed"
        st.markdown(f"""
        <div class='card' style='padding:10px;'>
          <div style='display:flex;justify-content:space-between;'>
            <span style='font-weight:600;'>{name}</span>
            <span style='color:{color};font-size:0.82rem;'>{badge}</span>
          </div>
          {'<div style="font-size:0.78rem;color:#999;margin-top:3px;">Install: <code>' + cmd + '</code></div>' if not status else ''}
        </div>""", unsafe_allow_html=True)

    # Persona debug
    st.markdown("---")
    st.markdown("**🎭 Persona Agent State**")
    profile = st.session_state.orchestrator.get_persona_profile()
    st.json(profile)