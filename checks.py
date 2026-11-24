"""
Checks for the steps in the task notebooks.

Each check prints a short message:
    ✓  your result has the expected form – go on
    ✗  something is missing, with a tip what to look at

The checks only test the form of a result (columns, types, sizes),
never the exact answer of the model, because that changes from run
to run.
"""

import pandas as pd


def _ok(message):
    print("✓", message)


def _fail(message, tip):
    print("✗", message)
    print("  Tip:", tip)


# ---------------------------------------------------------------------------
# Exercise 1
# ---------------------------------------------------------------------------

FIELDS_1 = {"scene", "people_count", "text_in_image"}


def check_record(answer):
    """Step 1: one answer with the fields scene, people_count, text_in_image."""
    if not isinstance(answer, dict):
        _fail("answer is not a dict.",
              "Store the result of ask(PROMPT, ImageRecord, [...]) in answer.")
        return
    missing = FIELDS_1 - set(answer)
    if missing:
        _fail(f"missing fields: {', '.join(sorted(missing))}.",
              "Add one line per field to the class ImageRecord.")
        return
    if not isinstance(answer["people_count"], int):
        _fail("people_count is not a whole number.", "Write people_count: int.")
        return
    if answer["text_in_image"] is not None and not isinstance(answer["text_in_image"], str):
        _fail("text_in_image is neither text nor None.", "Write text_in_image: str | None.")
        return
    _ok("all three fields with the right types.")


def check_extracted(extracted, files):
    """Step 2: a table with one row per image and a column file."""
    if not isinstance(extracted, pd.DataFrame):
        _fail("extracted is not a DataFrame.", "Build it with pd.DataFrame(rows).")
        return
    if len(extracted) != len(files):
        _fail(f"{len(extracted)} rows, expected {len(files)} – one per image.",
              "Append one row per image inside the loop.")
        return
    missing = (FIELDS_1 | {"file"}) - set(extracted.columns)
    if missing:
        _fail(f"missing columns: {', '.join(sorted(missing))}.",
              'Use ImageRecord from step 1 and add row["file"] = name.')
        return
    if sorted(extracted["file"]) != sorted(files):
        _fail("the file names do not match the images.",
              "Use the filename only, e.g. \"sheep.jpg\", not the full path.")
        return
    _ok(f"{len(extracted)} rows, one per image, with all columns.")


def check_scores(people, cer):
    """Step 3: an accuracy between 0 and 1 and a character error rate >= 0."""
    if not (isinstance(people, (int, float)) and 0 <= people <= 1):
        _fail(f"people should be a share between 0 and 1, got {people!r}.",
              "Use accuracy_score(label, model).")
        return
    if not (isinstance(cer, (int, float)) and cer >= 0):
        _fail(f"cer should be a number of 0 or more, got {cer!r}.",
              "Use jiwer.cer(label_texts, model_texts).")
        return
    _ok(f"people accuracy = {people:.2f}, CER = {cer:.2f}.")


# ---------------------------------------------------------------------------
# Exercises 2 and 4: frames of a video
# ---------------------------------------------------------------------------

def check_frame_paths(paths, expected=None):
    """Step: one saved JPG file per second of the video."""
    from pathlib import Path
    if not isinstance(paths, list) or not paths:
        _fail("paths is not a list of files.", "Store the result of extract_frames(...) in paths.")
        return
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        _fail(f"{len(missing)} files do not exist.", "Run extract_frames(...) again.")
        return
    if expected is not None and len(paths) != expected:
        _fail(f"{len(paths)} frames, expected {expected} – one per second.",
              "Use extract_frames(video_path, folder=...) with the right video.")
        return
    _ok(f"{len(paths)} frames saved, one per second.")


# ---------------------------------------------------------------------------
# Exercise 2
# ---------------------------------------------------------------------------

def check_frames(frames, paths):
    """Step 2: one row per frame with t, sky and oars_in_water."""
    if not isinstance(frames, pd.DataFrame):
        _fail("frames is not a DataFrame.", "Build it with pd.DataFrame(rows).")
        return
    if len(frames) != len(paths):
        _fail(f"{len(frames)} rows, expected {len(paths)} – one per frame.",
              "Append one row per frame inside the loop.")
        return
    missing = {"t", "sky", "oars_in_water"} - set(frames.columns)
    if missing:
        _fail(f"missing columns: {', '.join(sorted(missing))}.",
              'Use your FrameRecord and add row["t"] = t.')
        return
    if list(frames["t"]) != list(range(len(paths))):
        _fail("t should be 0, 1, 2, ... – the second of each frame.", "Loop with for t in range(len(paths)).")
        return
    _ok(f"{len(frames)} rows, one per second, with t, sky and oars_in_water.")


def check_changes(changes, frames):
    """Step 3: the seconds where the sky changes."""
    if not isinstance(changes, pd.DataFrame):
        _fail("changes is not a DataFrame.", 'Use frames.loc[changed, ["t", "sky"]].')
        return
    if not {"t", "sky"} <= set(changes.columns):
        _fail("changes needs the columns t and sky.", 'Select them with ["t", "sky"].')
        return
    if len(changes) == 0 or changes["t"].iloc[0] != 0:
        _fail("the first row should be t = 0, where the clip starts.",
              'frames["sky"] != frames["sky"].shift() is also True for t = 0.')
        return
    if len(changes) > len(frames):
        _fail("more rows than frames.", "Keep only the rows where the sky changes.")
        return
    _ok(f"{len(changes)} moments: the start and {len(changes) - 1} change(s) of the sky.")


# ---------------------------------------------------------------------------
# Exercise 3
# ---------------------------------------------------------------------------

def check_prompts(prompt_a, prompt_b, subject, setting, composition):
    """Step 1: two prompts that differ only in the light block."""
    if not (isinstance(prompt_a, str) and isinstance(prompt_b, str)):
        _fail("prompt_a and prompt_b must be texts.", 'Build them with f"{subject}, {setting}, ...".')
        return
    if prompt_a == prompt_b:
        _fail("both prompts are the same.", "Change the light block for prompt B.")
        return
    for name, block in [("subject", subject), ("setting", setting), ("composition", composition)]:
        if block not in prompt_a or block not in prompt_b:
            _fail(f"the {name} block is not the same in both prompts.",
                  "Change only the light block, so you know what caused the difference.")
            return
    _ok("only the light block differs between A and B.")


def check_pictures(paths, expected=4):
    """Step 2: the created picture files."""
    from pathlib import Path
    if not isinstance(paths, list) or len(paths) != expected:
        _fail(f"expected a list of {expected} pictures.", "Store the result of create_pictures(...) in paths.")
    elif not all(Path(p).exists() for p in paths):
        _fail("some picture files do not exist.", "Run create_pictures(...) again.")
    else:
        _ok(f"{expected} pictures created.")


def check_ratings(ratings, expected=4):
    """Step 3: your rating and the API rating per picture."""
    if not isinstance(ratings, pd.DataFrame):
        _fail("ratings is not a DataFrame.", "Build it with pd.DataFrame().")
        return
    missing = {"name", "version", "me", "api"} - set(ratings.columns)
    if missing:
        _fail(f"missing columns: {', '.join(sorted(missing))}.", "Add one column per item.")
        return
    if len(ratings) != expected:
        _fail(f"{len(ratings)} rows, expected {expected} – one per picture.", "One row per picture.")
        return
    if not all(isinstance(v, bool) for v in list(ratings["me"]) + list(ratings["api"])):
        _fail("me and api must be True or False.", 'Use answer["matches"] for api.')
        return
    _ok("4 pictures, rated by you and by the API.")


# ---------------------------------------------------------------------------
# Exercise 4
# ---------------------------------------------------------------------------

def check_scene(plan, flow_prompt, character, dog):
    """Step 1: a scene prompt with five parts and the fixed descriptions in front."""
    parts = {"shot", "camera", "action", "light", "sound"}
    if not isinstance(plan, dict) or parts - set(plan):
        _fail("plan needs the five parts shot, camera, action, light and sound.",
              "Use the class ScenePrompt from the example.")
        return
    if not isinstance(flow_prompt, str) or not flow_prompt.startswith(f"{character}. {dog}."):
        _fail("flow_prompt must start with the fixed descriptions.",
              'Write f"{CHARACTER}. {DOG}. " + " ".join(plan.values()).')
        return
    _ok("five parts, fixed descriptions in front – ready for Google Flow.")


def check_consistency(ratings, paths):
    """Step 3: one row per frame with beanie, jacket and dog."""
    allowed = {"match", "differs", "not visible"}
    if not isinstance(ratings, pd.DataFrame):
        _fail("ratings is not a DataFrame.", "Build it with pd.DataFrame(rows).")
        return
    missing = {"beanie", "jacket", "dog", "frame"} - set(ratings.columns)
    if missing:
        _fail(f"missing columns: {', '.join(sorted(missing))}.",
              'Add the fields to your class and row["frame"] = path.name.')
        return
    if len(ratings) != len(paths):
        _fail(f"{len(ratings)} rows, expected {len(paths)} – one per frame.", "One row per frame.")
        return
    values = set(ratings[["beanie", "jacket", "dog"]].values.ravel())
    if not values <= allowed:
        _fail(f"unexpected ratings: {values - allowed}.", 'Use Literal["match", "differs", "not visible"].')
        return
    _ok(f"{len(ratings)} frames rated for beanie, jacket and dog.")
