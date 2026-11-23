"""
Helper functions for the exercise notebooks.

All API calls in this file go to the OpenAI API (models gpt-6-luna,
gpt-image-2 and Whisper). Other providers, e.g. Google Gemini or
Anthropic Claude, have their own APIs with different code.

OpenAI API (needs OPENAI_API_KEY):
    describe(prompt, path)              answer about one image as text
    ask(prompt, answer_format, images)  answer in a fixed format as a dict
    transcribe(video_path)              speech with start and end times
    generate(prompt, path)              create one picture and save it
    create_pictures(names, prompts)     create a set of pictures as PNG files

No API:
    image_to_input(path)                image file -> OpenAI API input
    extract_frames(video_path)          one frame per second as JPG files
    show_images(paths)                  images in a grid with file names

Every API answer is written to api_log.jsonl.
"""

# Standard Python libraries.
import base64
import json
import math
import mimetypes
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

# The OpenAI model that creates pictures (about US$0.006 per picture).
IMAGE_MODEL = "gpt-image-2"

# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def connect():
    """Connect to the OpenAI API. The key is read from OPENAI_API_KEY."""
    # Retry with backoff: after a rate limit (429) or a timeout, the
    # OpenAI library waits a little longer after each failed try and
    # gives up after 5 retries. A call waits at most 60 seconds.
    return OpenAI(max_retries=5, timeout=60)


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


def write_log(prompt, model, answer):
    """Add one line with prompt, model and answer to api_log.jsonl."""
    entry = {"prompt": prompt, "model": model, "answer": answer}
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
    write_log(prompt, MODEL, response.output_text)
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
    write_log(prompt, MODEL, response.output_text)

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


def generate(prompt, path):
    """Create one picture from a prompt and save it as a PNG file."""
    # 1. Ask the OpenAI Images API for one square picture.
    #    Low quality is the cheapest setting and enough for this exercise.
    response = connect().images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality="low",
        n=1,
    )

    # 2. The picture comes back as text (base64). Turn it into a file.
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_bytes(base64.b64decode(response.data[0].b64_json))

    # 3. Log the prompt and where the picture was saved.
    write_log(prompt, IMAGE_MODEL, str(path))
    return Path(path)


def create_pictures(names, prompts, folder="variants"):
    """Create one picture per name and prompt. Return the file paths.

    The file pictures.json in the folder remembers the prompt and model
    of every picture. An unchanged picture is kept, so you do not pay
    twice. A picture whose prompt or model changed is created anew.
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    record_file = folder / "pictures.json"
    records = {}
    if record_file.exists():
        records = json.loads(record_file.read_text(encoding="utf-8"))

    paths = []
    for name, prompt in zip(names, prompts):
        path = folder / f"{name}.png"
        wanted = {"prompt": prompt, "model": IMAGE_MODEL}

        # Create the picture only if it is missing or its prompt changed.
        if path.exists() and records.get(name) == wanted:
            print("Unchanged", name)
        else:
            generate(prompt, path)
            records[name] = wanted
            record_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("Created", name)
        paths.append(path)
    return paths


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
