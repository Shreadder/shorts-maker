import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faster_whisper import WhisperModel
import yt_dlp
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from youtube_transcript_api import YouTubeTranscriptApi


def extract_youtube_id(url: str) -> str:
    """استخراج معرّف الفيديو من رابط يوتيوب."""
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else url


def fetch_fast_transcript(url: str) -> list:
    """جلب الترجمة المباشرة أو تفريغ الصوت سحابياً عبر faster-whisper مع تخطي حظر 403."""
    video_id = extract_youtube_id(url)

    # 1. فحص توفر تفريغ يوتيوب التلقائي المباشر
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "ar"])
        segments = []
        for item in transcript:
            segments.append({
                "start": item["start"],
                "end": item["start"] + item["duration"],
                "text": item["text"]
            })
        return segments
    except Exception:
        pass

    # 2. تحميل الصوت فقط في حال عدم توفر ترجمة مع تخطي حظر السيرفرات السحابية
    audio_path = "temp_fast_audio.mp3"
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass

    ydl_opts = {
        "format": "ba/ba*",
        "outtmpl": "temp_fast_audio.%(ext)s",
        "overwrites": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

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
        raise RuntimeError("لم يتم العثور على حوار منطوق لتحليله.")

    return segments


def download_clip_range_only(
    url: str,
    start_time: float,
    end_time: float,
    output_path: str = "temp_short_raw.mp4"
) -> str:
    """تحميل الثواني المحددة فقط من يوتيوب مباشرة مع تجاوز حظر 403 السحابي."""
    print(f"تحميل اللحظة المحددة فقط ({start_time:.1f}s إلى {end_time:.1f}s)...")
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "overwrites": True,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        "force_keyframes_at_cuts": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


def crop_to_vertical_9_16(clip: VideoFileClip) -> VideoFileClip:
    """قص وتوسيط الفيديو ليناسب شاشات الجوال 9:16."""
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


def process_short_video(raw_clip_path: str, output_path: str = "temp_cropped.mp4") -> str:
    """تحويل أبعاد المقطع المقتطع إلى 9:16 بنمط معالجة فائق السرعة."""
    with VideoFileClip(raw_clip_path) as clip:
        processed_clip = crop_to_vertical_9_16(clip)
        processed_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="ultrafast",
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
    """توليد الإطار الرسومي لك
