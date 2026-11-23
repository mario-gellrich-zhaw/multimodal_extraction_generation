"""
Helper functions for the exercise notebooks.

The notebooks import what they need from this file:

    image_to_input(path)                  turns an image file into an API input
    ask(prompt, answer_format, images)    asks the model, returns a Python dict
    describe(prompt, path)                asks the model about one image, returns text
    extract_frames(video_path)            saves one frame per second as JPG files

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

# Connect to OpenAI. The key is read from OPENAI_API_KEY; busy requests are retried.
client = OpenAI(max_retries=5, timeout=60)
MODEL = "gpt-6-luna"


# Turn an image file into an API input.
def image_to_input(path):
    image_type = mimetypes.guess_type(path)[0]  # e.g. "image/jpeg"
    image_text = base64.b64encode(Path(path).read_bytes()).decode()  # bytes -> text
    return {
        "type": "input_image",
        "image_url": f"data:{image_type};base64,{image_text}",
        "detail": "low",  # low detail = fewer tokens = cheaper
    }


# Add one line with prompt, model and raw answer to api_log.jsonl.
def write_log(prompt, answer):
    entry = {"prompt": prompt, "model": MODEL, "answer": answer}
    with open("api_log.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Send a prompt and a list of images, get the answer as a dict.
def ask(prompt, answer_format, images):
    # 1. Put the text and the images into one message.
    content = [{"type": "input_text", "text": prompt}]
    for path in images:
        content.append(image_to_input(path))

    # 2. Send the message. The answer must follow answer_format.
    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        text_format=answer_format,
        reasoning={"effort": "none"},  # no extra thinking: faster and cheaper
        store=False,  # OpenAI does not keep the answer
    )

    # 3. Log the raw answer.
    write_log(prompt, response.output_text)

    # 4. Stop with a clear message if there is no valid answer.
    if response.output_parsed is None:
        raise ValueError(f"No valid answer for {images}. Look at api_log.jsonl.")

    # 5. Return the answer as a Python dict.
    return response.output_parsed.model_dump()


# Send a prompt and one image, get the answer as text.
def describe(prompt, path):
    # 1. Put the text and the image into one message.
    content = [{"type": "input_text", "text": prompt}, image_to_input(path)]

    # 2. Send the message.
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "none"},
        store=False,
    )

    # 3. Log the raw answer and return it.
    write_log(prompt, response.output_text)
    return response.output_text


# Save one frame per second of a video as JPG files. Returns the file paths.
def extract_frames(video_path, folder="frames", seconds=8):
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
