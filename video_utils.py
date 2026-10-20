import io
import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faster_whisper import WhisperModel
import yt_dlp
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from youtube_transcript_api import YouTubeTranscriptApi


def extract_youtube_id(url: str) -> str:
    """استخراج المعرف الخاص بالفيديو من رابط يوتيوب."""
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else url


def fetch_fast_transcript(url: str, fallback_video_path: str = None) -> list:
    """جلب تفريغ يوتيوب الفوري، وفي حال عدم توفره يتم استخدام faster-whisper السريع."""
    video_id = extract_youtube_id(url)

    # 1. محاولة جلب الترجمة الجاهزة من يوتيوب
    try:
        print(f"جاري فحص ترجمة يوتيوب المباشرة ({video_id})...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "ar"])
        segments = []
        for item in transcript:
            segments.append({
                "start": item["start"],
                "end": item["start"] + item["duration"],
                "text": item["text"]
            })
        print(f"تم العثور على ترجمة يوتيوب مباشرة ({len(segments)} سطر).")
        return segments
    except Exception:
        print("لا تتوفر ترجمة مباشرة من يوتيوب، جاري تفريغ الصوت عبر faster-whisper...")

    # 2. الحل التلقائي: تحميل الصوت وتفريغه بسرعة عبر faster-whisper
    audio_path = "temp_fast_audio.mp3"
    if not os.path.exists(audio_path):
        print("تحميل مسار الصوت السريع...")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "temp_fast_audio.%(ext)s",
            "overwrites": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    print("جاري تفريغ الصوت بنموذج faster-whisper (حسابات int8 السريعة)...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments_generator, _ = model.transcribe(audio_path, beam_size=5)

    segments = []
    for s in segments_generator:
        segments.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip()
        })

    if not segments:
        raise RuntimeError("الفيديو لا يحتوي على كلام أو صوت منطوق واضح.")

    return segments


def download_highest_quality_video(url: str, output_path: str = "raw_video.mp4") -> str:
    """تحميل الفيديو الأصلي بأعلى جودة ممكنة."""
    print(f"جاري تحميل الفيديو بجودة كاملة: {url}...")
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_path,
        "overwrites": True,
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoRemuxer",
                "preferedformat": "mp4",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


def crop_to_vertical_9_16(clip: VideoFileClip) -> VideoFileClip:
    """تحويل أبعاد الفيديو إلى 9:16 عمودياً للشورتس وتيك توك."""
    orig_w, orig_h = clip.size
    target_aspect_ratio = 9 / 16
    current_aspect_ratio = orig_w / orig_h

    if current_aspect_ratio > target_aspect_ratio:
        new_w = int(orig_h * target_aspect_ratio)
        x_center = orig_w / 2
        cropped = clip.cropped(
            x1=x_center - new_w / 2,
            y1=0,
            x2=x_center + new_w / 2,
            y2=orig_h
        )
    else:
        new_h = int(orig_w / target_aspect_ratio)
        y_center = orig_h / 2
        cropped = clip.cropped(
            x1=0,
            y1=y_center - new_h / 2,
            x2=orig_w,
            y2=y_center + new_h / 2
        )

    return cropped.resized(new_size=(1080, 1920))


def export_ultra_quality_short(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str = "temp_short.mp4"
) -> str:
    """قص اللحظة المحددة وضبط مقاييس الفيديو بسرعة وكفاءة."""
    print(f"جاري قص المقطع من {start_time:.1f}s إلى {end_time:.1f}s...")
    with VideoFileClip(video_path) as full_video:
        sub_clip = full_video.subclipped(start_time, end_time)
        processed_clip = crop_to_vertical_9_16(sub_clip)

        ffmpeg_params = [
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.2",
            "-b:a", "192k",
            "-ar", "48000"
        ]

        # استخدام 30 إطاراً في الثانية لتسريع الريندر للنصف
        processed_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="veryfast",
            ffmpeg_params=ffmpeg_params,
            threads=4
        )
    return output_path


def generate_karaoke_frame(
    words_list: list,
    active_idx: int,
    frame_width: int,
    font_path: str = "arialbd.ttf",
    font_size: int = 55,
    normal_color: str = "#FFFFFF",
    active_color: str = "#FFE600",
    stroke_color: str = "#000000",
    stroke_width: int = 5
) -> np.ndarray:
    """توليد صورة سطر الكلمات مع تلوين الكلمة النشطة."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    space_width = font.getbbox(" ")[2] - font.getbbox(" ")[0]
    total_line_width = 0
    word_boxes = []

    for w in words_list:
        bbox = font.getbbox(w)
        w_width = bbox[2] - bbox[0]
        w_height = bbox[3] - bbox[1]
        word_boxes.append((w_width, w_height))
        total_line_width += w_width + space_width

    total_line_width -= space_width
    max_height = max(b[1] for b in word_boxes) + (stroke_width * 4)

    img = Image.new("RGBA", (frame_width, max_height + 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    current_x = (frame_width - total_line_width) / 2
    current_y = 10

    for i, word in enumerate(words_list):
        fill_color = active_color if i == active_idx else normal_color
        draw.text(
            (current_x, current_y),
            word,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color
        )
        current_x += word_boxes[i][0] + space_width

    return np.array(img)


def add_karaoke_captions_to_video(
    video_path: str,
    output_path: str = "final_karaoke_short.mp4",
    max_words_per_line: int = 4,
    active_color: str = "#FFE500",
    normal_color: str = "#FFFFFF",
    font_size: int = 55
) -> str:
    """تفريغ المقطع القصير وتركيب كابشنز الكاريوكي المتزامنة عبر faster-whisper."""
    print("تفريغ المقطع واستخراج توقيت الكلمات بدقة عبر faster-whisper...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments_generator, _ = model.transcribe(video_path, word_timestamps=True)

    all_words = []
    for segment in segments_generator:
        if segment.words:
            for w in segment.words:
                clean_word = w.word.strip().upper()
                if clean_word:
                    all_words.append({
                        "word": clean_word,
                        "start": w.start,
                        "end": w.end
                    })

    if not all_words:
        return video_path

    groups = [
        all_words[i:i + max_words_per_line]
        for i in range(0, len(all_words), max_words_per_line)
    ]

    subtitle_clips = []

    with VideoFileClip(video_path) as video:
        video_w, video_h = video.size
        y_position = int(video_h * 0.72)

        for group in groups:
            group_words_text = [item["word"] for item in group]

            for idx, active_item in enumerate(group):
                start_t = active_item["start"]
                if idx + 1 < len(group):
                    end_t = group[idx + 1]["start"]
                else:
                    end_t = active_item["end"]

                duration = max(end_t - start_t, 0.15)

                frame_img = generate_karaoke_frame(
                    words_list=group_words_text,
                    active_idx=idx,
                    frame_width=video_w,
                    font_size=font_size,
                    normal_color=normal_color,
                    active_color=active_color,
                    stroke_color="#000000",
                    stroke_width=5
                )

                clip = (
                    ImageClip(frame_img)
                    .with_start(start_t)
                    .with_duration(duration)
                    .with_position(("center", y_position))
                )
                subtitle_clips.append(clip)

        print(f"جاري تصدير الفيديو النهائي ({len(subtitle_clips)} كلمة)...")
        final_video = CompositeVideoClip([video, *subtitle_clips])

        ffmpeg_params = [
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.2",
            "-b:a", "192k",
            "-ar", "48000"
        ]

        # تصدير بـ 30 إطاراً ونمط veryfast لتسريع التوليد بشكل فوري
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="veryfast",
            ffmpeg_params=ffmpeg_params,
            threads=4
        )

    return output_path