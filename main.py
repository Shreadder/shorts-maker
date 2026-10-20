import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from video_utils import (
    download_highest_quality_video,
    transcribe_video,
    export_ultra_quality_short,
    extract_frame_thumbnail,
    add_karaoke_captions_to_video
)


class ViralMoment(BaseModel):
    start_time: float = Field(description="توقيت بداية اللحظة بالثواني")
    end_time: float = Field(description="توقيت نهاية اللحظة بالثواني")
    virality_score: int = Field(description="نسبة انتشار المقطع من 0 إلى 100")
    reason: str = Field(description="تحليل سبب اختيار اللحظة وقوة الخطاف")
    suggested_title: str = Field(description="عنوان شورتس جاذب مع الهاشتاقات")


def analyze_transcript_for_viral_moment(segments: list) -> ViralMoment:
    print("\nجاري تحليل الحوار بالذكاء الاصطناعي لاكتشاف اللحظة الأكثر انتشاراً...")
    client = genai.Client()

    formatted_transcript = "\n".join(
        [f"[{seg['start']:.2f}s -> {seg['end']:.2f}s]: {seg['text']}" for seg in segments]
    )

    prompt = f"""
    أنت خبير محتوى وتحليل بيانات متخصص في مقاطع YouTube Shorts الفيروسية (Viral).
    تفقد هذا النص المفرغ مع التوقيتات:
    \"\"\"
    {formatted_transcript}
    \"\"\"
    المطلوب:
    1. استخرج أفضل مقطع مدته بين 30 إلى 60 ثانية يبدأ بـ Hook قوي يجذب الانتباه.
    2. تأكد أن الفكرة كاملة المعنى وليست مبتورة في النهاية.
    3. احسب نسبة الانتشار المتوقعة (Virality Score من 0 إلى 100).
    4. اكتب عنواناً مقترحاً جذاباً.
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


if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # 1. التحميل والتفريغ
    raw_video = download_highest_quality_video(VIDEO_URL, "raw_video.mp4")
    transcription = transcribe_video(raw_video, model_size="base")

    # 2. التحليل الذكي
    viral_data = analyze_transcript_for_viral_moment(transcription["segments"])

    print("\n" + "=" * 50)
    print(" تقرير تحليل الانتشار (Virality Report):")
    print(f"- التوقيت المختار: من {viral_data.start_time:.1f}s إلى {viral_data.end_time:.1f}s")
    print(f"- نسبة الانتشار المتوقعة: {viral_data.virality_score}%")
    print(f"- العنوان المقترح: {viral_data.suggested_title}")
    print(f"- سبب الاختيار: {viral_data.reason}")
    print("=" * 50 + "\n")

    # 3. قص الشورتس
    short_file = export_ultra_quality_short(
        raw_video,
        viral_data.start_time,
        viral_data.end_time,
        "ai_viral_short.mp4"
    )
    extract_frame_thumbnail(short_file, timestamp_sec=1.5, output_thumb="viral_thumb.png")

    # 4. إضافة نصوص الكاريوكي المتزامنة تلقائياً
    karaoke_file = add_karaoke_captions_to_video(
        video_path=short_file,
        output_path="final_karaoke_short.mp4",
        max_words_per_line=4,
        active_color="#FFE500",  # أصفر فسفوري للكلمة الحالية
        normal_color="#FFFFFF",  # أبيض للكلمات الأخرى
        font_size=60
    )

    print("\n اكتمل مسار العمل بالكامل! تم إنتاج الفيديو النهائي مع الكاريوكي: final_karaoke_short.mp4")