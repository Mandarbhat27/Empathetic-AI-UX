"""
PDF Report Engine
Generates a beautiful mood report PDF from session data
Install: pip install reportlab --break-system-packages
"""

import io
import datetime

def generate_mood_report(mood_log: list, streak: int, chat_count: int) -> bytes:
    """
    Generate a PDF mood report and return as bytes.
    Works fully offline — no cloud needed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm,   bottomMargin=2*cm
        )

        # ── Color palette ──────────────────────────────────────────────────────
        PURPLE  = colors.HexColor("#6C63FF")
        DARK    = colors.HexColor("#1A1A2E")
        TEAL    = colors.HexColor("#0F9B8E")
        ORANGE  = colors.HexColor("#F59E0B")
        LIGHT   = colors.HexColor("#F0F0FA")
        WHITE   = colors.white

        EMOTION_COLORS = {
            "happy":   colors.HexColor("#F5A623"),
            "sad":     colors.HexColor("#6B7FD7"),
            "angry":   colors.HexColor("#E05555"),
            "anxious": colors.HexColor("#48BB78"),
            "stressed":colors.HexColor("#ED8936"),
            "excited": colors.HexColor("#D69E2E"),
            "neutral": colors.HexColor("#667EEA"),
        }

        EMOTION_EMOJI = {
            "happy": "😊", "sad": "😢", "angry": "😠",
            "anxious": "😰", "stressed": "😤", "excited": "🤩", "neutral": "😐"
        }

        # ── Styles ─────────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "Title", fontSize=26, textColor=PURPLE,
            alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold"
        )
        subtitle_style = ParagraphStyle(
            "Subtitle", fontSize=11, textColor=colors.HexColor("#667788"),
            alignment=TA_CENTER, spaceAfter=2, fontName="Helvetica"
        )
        section_style = ParagraphStyle(
            "Section", fontSize=14, textColor=DARK,
            spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold"
        )
        body_style = ParagraphStyle(
            "Body", fontSize=10, textColor=colors.HexColor("#333344"),
            spaceAfter=4, fontName="Helvetica", leading=15
        )
        small_style = ParagraphStyle(
            "Small", fontSize=9, textColor=colors.HexColor("#778899"),
            spaceAfter=2, fontName="Helvetica", leading=13
        )

        story = []

        # ── Header ─────────────────────────────────────────────────────────────
        story.append(Paragraph("🧠 EmpathyOS", title_style))
        story.append(Paragraph("Your Personal Mood Report", subtitle_style))
        now = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated: {now}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=16))

        # ── Summary stats ──────────────────────────────────────────────────────
        story.append(Paragraph("📊 Summary", section_style))

        total = len(mood_log)
        emotion_counts = {}
        for entry in mood_log:
            e = entry.get("emotion", "neutral")
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        dominant_pct = round((emotion_counts.get(dominant, 0) / total * 100)) if total else 0

        stats_data = [
            ["Metric", "Value"],
            ["Total Check-ins", str(total)],
            ["Conversation Sessions", str(chat_count)],
            ["Daily Streak", f"{streak} days 🔥"],
            ["Dominant Mood", f"{EMOTION_EMOJI.get(dominant, '')} {dominant.capitalize()} ({dominant_pct}%)"],
        ]

        stats_table = Table(stats_data, colWidths=[9*cm, 8*cm])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  PURPLE),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  11),
            ("BACKGROUND",  (0, 1), (-1, -1), LIGHT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCDD")),
            ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
            ("TOPPADDING",  (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 14))

        # ── Emotion breakdown ──────────────────────────────────────────────────
        story.append(Paragraph("🎭 Emotion Breakdown", section_style))

        if emotion_counts:
            emotion_rows = [["Emotion", "Count", "Percentage", "Bar"]]
            for em, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
                pct = round(count / total * 100)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                emotion_rows.append([
                    f"{EMOTION_EMOJI.get(em, '')} {em.capitalize()}",
                    str(count),
                    f"{pct}%",
                    bar
                ])

            em_table = Table(emotion_rows, colWidths=[5*cm, 2.5*cm, 3*cm, 6.5*cm])
            em_table.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  TEAL),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0),  10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 1), (-1, -1), 9),
                ("FONTNAME",    (0, 1), (3, -1),  "Courier"),
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCDD")),
                ("ALIGN",       (1, 0), (2, -1),  "CENTER"),
                ("TOPPADDING",  (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(em_table)

        story.append(Spacer(1, 14))

        # ── Mood Timeline ──────────────────────────────────────────────────────
        if mood_log:
            story.append(Paragraph("🕐 Mood Timeline (Last 15 Entries)", section_style))

            timeline_rows = [["Time", "Emotion", "What you said"]]
            for entry in mood_log[-15:][::-1]:
                em   = entry.get("emotion", "neutral")
                text = entry.get("text", "")
                text = text[:60] + "..." if len(text) > 60 else text
                timeline_rows.append([
                    entry.get("time", ""),
                    f"{EMOTION_EMOJI.get(em, '')} {em.capitalize()}",
                    text
                ])

            tl_table = Table(timeline_rows, colWidths=[2.5*cm, 4*cm, 10.5*cm])
            tl_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  DARK),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0),  10),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
                ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",     (0, 1), (-1, -1), 9),
                ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCDD")),
                ("TOPPADDING",   (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
                ("LEFTPADDING",  (0, 0), (-1, -1), 7),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(tl_table)
            story.append(Spacer(1, 14))

        # ── Insight ────────────────────────────────────────────────────────────
        story.append(Paragraph("💡 Insight", section_style))
        insight_map = {
            "happy":   "You've been in a great headspace! Keep doing what's working for you. Your positivity is a strength.",
            "sad":     "You've been carrying some heaviness. That takes courage to acknowledge. Consider talking to someone you trust.",
            "anxious": "Anxiety has been present in your sessions. Grounding techniques and reducing information overload can help significantly.",
            "stressed":"High stress levels detected. Breaking tasks into smaller steps and saying no to non-essentials can create breathing room.",
            "angry":   "Frustration has been recurring. Physical activity and identifying root causes can help channel this energy productively.",
            "excited": "You're riding high energy waves! Channel that momentum into meaningful goals before it fades.",
            "neutral": "You've been in a balanced state. A great time to plan, reflect, and set intentions.",
        }
        story.append(Paragraph(
            f"<b>Dominant mood: {EMOTION_EMOJI.get(dominant,'')} {dominant.capitalize()}</b><br/><br/>"
            + insight_map.get(dominant, "Keep checking in — consistency is the key to self-awareness."),
            body_style
        ))

        # ── Footer ─────────────────────────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCDD")))
        story.append(Paragraph(
            "Generated by EmpathyOS — On-Device Emotional Intelligence | All data stored locally, never uploaded",
            small_style
        ))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        return _fallback_pdf_text(mood_log, streak, chat_count)


def _fallback_pdf_text(mood_log, streak, chat_count) -> bytes:
    """Plain text fallback if reportlab not installed"""
    lines = [
        "EmpathyOS Mood Report",
        "=" * 40,
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}",
        f"Total check-ins: {len(mood_log)}",
        f"Streak: {streak} days",
        "",
        "Mood Timeline:",
        "-" * 40,
    ]
    for entry in mood_log[-20:][::-1]:
        lines.append(f"[{entry.get('time','')}] {entry.get('emotion','').upper()} — {entry.get('text','')[:60]}")
    return "\n".join(lines).encode("utf-8")
