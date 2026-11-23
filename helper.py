"""
Helper functions for the exercise notebooks.

All API calls in this file go to the OpenAI API (model gpt-6-luna and
Whisper). Other providers, e.g. Google Gemini or Anthropic Claude, have
their own APIs with different code.

OpenAI API (needs OPENAI_API_KEY):
    describe(prompt, path)              answer about one image as text
    ask(prompt, answer_format, images)  answer in a fixed format as a dict
    transcribe(video_path)              speech with start and end times

No API:
    image_to_input(path)                image file -> OpenAI API input
    extract_frames(video_path)          one frame per second as JPG files
    show_images(paths)                  images in a grid with file names
    find_images(folder, names)          uploaded .jpg or .png files
    join_clips(clips)                   joins video clips with ffmpeg

Every API answer is written to api_log.jsonl, with its token counts.
"""

# Standard Python libraries.
import base64
import json
import math
import mimetypes
import subprocess
from pathlib import Path

# Libraries installed from requirements.txt.
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from PIL import Image

# Folder with the course images and videos (next to this file).
DATA = Path(__file__).parent / "data"

# The OpenAI model for text and images.
MODEL = "gpt-6-luna"


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def connect():
    """Connect to the OpenAI API. The key is read from OPENAI_API_KEY."""
    return OpenAI(max_retries=5, timeout=60)  # retry busy requests


def image_to_input(path):
    """Turn an image file into an OpenAI API input."""
    image_type = mimetypes.guess_type(path)[0]  # e.g. "image/jpeg"

    # The request is text, so the image bytes are converted to text.
    image_bytes = Path(path).read_bytes()
    image_text = base64.b64encode(image_bytes).decode()

    return {
        "type": "input_image",
        "image_url": f"data:{image_type};base64,{image_text}",
        "detail": "low",  # low detail = fewer tokens = cheaper
    }


def write_log(prompt, model, answer, usage=None):
    """Add one line with prompt, model, answer and tokens to the log."""
    entry = {"prompt": prompt, "model": model, "answer": answer}
    if usage is not None:
        entry["input_tokens"] = usage.input_tokens
        entry["output_tokens"] = usage.output_tokens

    with open("api_log.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")


def describe(prompt, path):
    """Send a prompt and one image, get the answer as text."""
    # 1. Put the text and the image into one message.
    content = [
        {"type": "input_text", "text": prompt},
        image_to_input(path),
    ]

    # 2. Send the message to the OpenAI Responses API.
    response = connect().responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "none"},  # no extra thinking: faster, cheaper
        store=False,  # OpenAI does not keep the answer
    )

    # 3. Log the answer and return it.
    write_log(prompt, MODEL, response.output_text, response.usage)
    return response.output_text


def ask(prompt, answer_format, images):
    """Send a prompt and a list of images, get the answer as a dict."""
    # 1. Put the text and the images into one message.
    content = [{"type": "input_text", "text": prompt}]
    for path in images:
        content.append(image_to_input(path))

    # 2. Send the message to the OpenAI Responses API.
    #    The answer must follow answer_format.
    response = connect().responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        text_format=answer_format,
        reasoning={"effort": "none"},
        store=False,
    )

    # 3. Log the raw answer.
    write_log(prompt, MODEL, response.output_text, response.usage)

    # 4. Stop with a clear message if there is no valid answer.
    if response.output_parsed is None:
        message = f"No valid answer for {images}. Look at api_log.jsonl."
        raise ValueError(message)

    # 5. Return the answer as a Python dict.
    return response.output_parsed.model_dump()


def transcribe(video_path):
    """Turn the speech in a video into a table with start, end and text."""
    # 1. Send the video to Whisper. It reads the sound from the mp4.
    with open(video_path, "rb") as audio:
        response = connect().audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # 2. Log the answer.
    write_log("transcribe", "whisper-1", response.text)

    # 3. One row per speech segment.
    segments = response.model_dump()["segments"]
    return pd.DataFrame(segments)[["start", "end", "text"]]


# ---------------------------------------------------------------------------
# Images and videos (no API)
# ---------------------------------------------------------------------------

def extract_frames(video_path, folder="frames", seconds=8):
    """Save one frame per second as JPG files. Return the file paths."""
    # Create the folder and open the video.
    Path(folder).mkdir(exist_ok=True)
    video = cv2.VideoCapture(str(video_path))

    # Jump to each second and save the frame there.
    paths = []
    for t in range(seconds):
        video.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = video.read()
        if not ok:  # the video ended earlier
            break
        path = Path(folder) / f"frame_{t}s.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)

    # Close the video file.
    video.release()
    return paths


def show_images(paths, columns=4):
    """Show images in a grid, with the file name above each image."""
    # One row for every `columns` images.
    columns = min(columns, len(paths))
    rows = math.ceil(len(paths) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3 * columns, 2.2 * rows), squeeze=False
    )

    # Hide all axes, then draw one image per cell.
    for ax in axes.flat:
        ax.axis("off")
    for i in range(len(paths)):
        axes.flat[i].imshow(Image.open(paths[i]))
        axes.flat[i].set_title(Path(paths[i]).stem)
    plt.show()


def find_images(folder, names):
    """Find exactly one .jpg or .png file per name. Return the paths."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    paths = []
    for name in names:
        found = list(folder.glob(name + ".jpg"))
        found += list(folder.glob(name + ".png"))
        message = f"{folder}/ needs exactly one {name}.jpg or {name}.png"
        assert len(found) == 1, message
        paths.append(found[0])
    return paths


def join_clips(clips, output="film.mp4"):
    """Join video clips into one film with ffmpeg, without re-encoding."""
    # ffmpeg reads the list of clips from a text file.
    with open("scenes.txt", "w", encoding="utf-8") as file:
        for clip in clips:
            file.write(f"file '{Path(clip).as_posix()}'\n")

    # -c copy: the clips need the same format, resolution and frame rate.
    # -safe 0: allows paths like ../data/...
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "scenes.txt", "-c", "copy", output,
    ]
    subprocess.run(command, check=True)
    return Path(output)
