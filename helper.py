"""
Helper functions for the exercise notebooks.

All API calls in this file go to the OpenAI API
(Responses API, model gpt-6-luna). Other providers, e.g. Google Gemini
or Anthropic Claude, have their own APIs with different code.

The notebooks import what they need from this file:

image_to_input(path)
    Turns an image file into an OpenAI API input.
ask(prompt, answer_format, images)
    Asks the OpenAI model, returns the answer as a Python dict.
describe(prompt, path)
    Asks the OpenAI model about one image, returns the answer as text.
extract_frames(video_path)
    Saves one frame per second as JPG files (no API).

Every answer from ask() and describe() is written to api_log.jsonl.
"""

# Standard Python libraries.
import base64
import json
import mimetypes
from pathlib import Path

# Libraries installed from requirements.txt.
import cv2
from openai import OpenAI

# Folder with the course images and videos (next to this file).
DATA = Path(__file__).parent / "data"

# Connect to the OpenAI API. The key is read from OPENAI_API_KEY.
# Busy requests are retried up to 5 times.
client = OpenAI(max_retries=5, timeout=60)
MODEL = "gpt-6-luna"


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


def write_log(prompt, answer):
    """Add one line with prompt, model and raw answer to api_log.jsonl."""
    entry = {"prompt": prompt, "model": MODEL, "answer": answer}
    with open("api_log.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ask(prompt, answer_format, images):
    """Send a prompt and a list of images, get the answer as a dict."""
    # 1. Put the text and the images into one message.
    content = [{"type": "input_text", "text": prompt}]
    for path in images:
        content.append(image_to_input(path))

    # 2. Send the message to the OpenAI Responses API.
    #    The answer must follow answer_format.
    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        text_format=answer_format,
        reasoning={"effort": "none"},  # no extra thinking: faster, cheaper
        store=False,  # OpenAI does not keep the answer
    )

    # 3. Log the raw answer.
    write_log(prompt, response.output_text)

    # 4. Stop with a clear message if there is no valid answer.
    if response.output_parsed is None:
        message = f"No valid answer for {images}. Look at api_log.jsonl."
        raise ValueError(message)

    # 5. Return the answer as a Python dict.
    return response.output_parsed.model_dump()


def describe(prompt, path):
    """Send a prompt and one image, get the answer as text."""
    # 1. Put the text and the image into one message.
    content = [
        {"type": "input_text", "text": prompt},
        image_to_input(path),
    ]

    # 2. Send the message to the OpenAI Responses API.
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "none"},
        store=False,
    )

    # 3. Log the raw answer and return it.
    write_log(prompt, response.output_text)
    return response.output_text


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
