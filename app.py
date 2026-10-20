import os
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from video_utils import (
    fetch_fast_transcript,
    download_highest_quality_video,
    export_ultra_quality_short,
    add_karaoke_captions_to_video
)

st.set_page_config(page_title="AI Shorts Generator", layout="centered", page_icon="⚡")

st.title("⚡ صانع الشورتس الذكي بالذكاء الاصطناعي")
st.write("ضع رابط أي مقطع يوتيوب، وسيقوم النظام فوراً بتحليل المحتوى واختيار أفضل لقطة وإنتاج شورت عمودي مع كابشنز كاريوكي.")

class ViralMoment(BaseModel):
    start_time: float = Field(description="توقيت بداية اللحظة بالثواني")
    end_time: float = Field(description="توقيت نهاية اللحظة بالثواني")
    virality_score: int = Field(description="نسبة انتشار المقطع من 0 إلى 100")
    reason: str = Field(description="تحليل سبب اختيار اللحظة وقوة الخطاف")
    suggested_title: str = Field(description="عنوان شورتس جاذب مع الهاشتاقات")


def analyze_transcript(segments):
    client = genai.Client()
    formatted = "\n".join([f"[{s['start']:.2f}s -> {s['end']:.2f}s]: {s['text']}" for s in segments])
    prompt = f"""
    أنت خبير محتوى وتحليل بيانات متخصص في مقاطع YouTube Shorts الفيروسية (Viral).
    تفقد هذا النص المفرغ مع التوقيتات:
    \"\"\"{formatted}\"\"\"
    المطلوب:
    """
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ViralMoment,
            temperature=0.2,
        ),
    )
    return ViralMoment.model_validate_json(response.text)


video_url = st.text_input("🔗 ضع رابط يوتيوب هنا:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 توليد مقطع الشورتس الآن", type="primary"):
    if not video_url:
        st.error("يرجى إدخال رابط يوتيوب أولاً.")
    else:
        status = st.status("بدء المعالجة...", expanded=True)
        try:
            status.write("⚡ 1. فحص الحوار واستخراج النص وتوقيتاته...")
            transcript_segments = fetch_fast_transcript(video_url)

            status.write("🧠 2. الذكاء الاصطناعي يحلل اللحظة الأكثر انتشاراً...")
            viral_data = analyze_transcript(transcript_segments)

            status.write("📥 3. جاري تحميل الفيديو الأصلي...")
            raw_video = download_highest_quality_video(video_url, "raw_video.mp4")

            status.write(f"✂️ 4. قص المقطع من {viral_data.start_time:.1f}s إلى {viral_data.end_time:.1f}s وضبط الأبعاد 9:16...")
            cropped_file = export_ultra_quality_short(
                raw_video,
                viral_data.start_time,
                viral_data.end_time,
                "temp_short.mp4"
            )

            status.write("🎨 5. دمج كابشنز الكاريوكي المتزامنة تلقائياً...")
            final_file = add_karaoke_captions_to_video(
                video_path=cropped_file,
                output_path="final_karaoke_short.mp4",
                max_words_per_line=4,
                active_color="#FFE500",
                normal_color="#FFFFFF"
            )

            status.update(label=" اكتمل إنتاج الشورتس بنجاح!", state="complete", expanded=False)

            st.success("تم تجهيز المقطع النهائي!")
            st.subheader(viral_data.suggested_title)
            col1, col2 = st.columns(2)
            col1.metric("نسبة الانتشار المقدرة", f"{viral_data.virality_score}%")
            col2.write(f"**سبب الاختيار:** {viral_data.reason}")

            st.video(final_file)

            with open(final_file, "rb") as f:
                st.download_button(
                    label="📥 تحميل الفيديو النهائي (MP4)",
                    data=f,
                    file_name="viral_short.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            status.update(label="حدث خطأ", state="error")
            st.error(f"تفاصيل الخطأ: {str(e)}")