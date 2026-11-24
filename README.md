# Multimodal AI: generating and extracting pictures & motion pictures

MSc Data Science · ZHAW · Dr. Mario Gellrich, HS2026

Four Python 3.11 notebooks accompany the lecture. Allow about 20 minutes per exercise. The **OpenAI API** is used for extraction, planning and scoring; other providers, e.g. Google Gemini or Anthropic Claude, have their own APIs with different code. You generate video manually in **Google Flow**, a web app; in exercise 3 the OpenAI API creates the pictures, and you can try Google Flow for your own pictures.

## Start in GitHub Codespaces

1. Open this repository on GitHub. Select **Code → Codespaces → Create codespace** on the course branch. Wait for the Python 3.11 setup to finish. It installs ffmpeg, ffprobe and the Python packages.
2. Enter your own course API key when Codespaces prompts for `OPENAI_API_KEY`. If you already have a codespace, open **GitHub → Settings → Codespaces → New secret**, name it `OPENAI_API_KEY`, paste your key into the secret value, and grant this repository access. Stop and restart the codespace to load the new secret. Keep your key out of notebooks, files and screenshots.
3. Open the first notebook from the Explorer and choose the **Python 3.11** kernel. Run the cells from top to bottom. Pause at the manual Google Flow step in exercise 4.

See [creating a codespace](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository) and [managing personal Codespaces secrets](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces).

## Set up Google Flow

1. If you need a Google account, go to **Create account → For my personal use** and follow the prompts. You need to be 18 or older with verified age and in a supported region. [Create a Google Account](https://support.google.com/accounts/answer/27441) · [Google Flow eligibility](https://support.google.com/flow/answer/16353333?hl=en)
2. Open [Google Flow](https://labs.google/flow) in Chrome or Edge, sign in and create a project. Images are free: try prompts A and B from exercise 3 to create your own pictures. Save your video credits for exercise 4. Download the clip and upload it through the Codespaces Explorer.
3. Google Flow lists 50 daily credits for free accounts. At 10 credits per generation, Veo 3.1 Lite allows up to five clips and supports Ingredients to Video at 8 seconds. Model and feature prices vary. Check the displayed credit cost and whether the model supports Ingredients to Video before generating. Exercise 4 requires one clip and includes a fallback. [Google Flow credits](https://support.google.com/flow/answer/16526234?hl=en) [Google Flow video model features](https://support.google.com/flow/answer/16352836?hl=en) · [Google Flow images](https://support.google.com/flow/answer/16729550?hl=en)

## Exercises

| Notebook | What you will do |
| --- | --- |
| [1. Image → JSON → DataFrame](01_Image_Extraction/image_extraction_example.ipynb) | **Example:** one image to the OpenAI API with its cost, a fixed answer format, a table for three lake images, scored against labels. **Task:** people and text in three new images (parking lot, alpine pasture, city street), scored with accuracy and character error rate. |
| [2. Video → frames → JSON timeline](02_Video_Extraction/video_extraction_example.ipynb) | **Example:** cut Ueli's clip into one frame per second, describe the frames, transcribe the speech and merge both on time. **Task:** fields per frame for a sunrise clip and the moments when the sky changes. |
| [3. Prompt experiment – Google Flow and OpenAI API](03_Image_Generation/image_generation_example.ipynb) | **Example:** a prompt from building blocks, one block changed, one picture per version, rated by the OpenAI API. **Task:** your own experiment with the light block – two pictures per version, rated by eye and by the OpenAI API. |
| [4. Storyboard → clips → measure consistency](04_Video_Generation/video_generation_example.ipynb) | **Example:** the prompt for one scene, the character sheet, and one feature checked frame by frame. **Task:** a new scene, the clip made in Google Flow (optionally with your own character sheet), and beanie, jacket and dog checked against the sheet. |

Each exercise has two notebooks. The **example** (`…_example.ipynb`, linked in the table) is short and complete – run it and read the code. The **tasks** notebook (`…_tasks.ipynb`) gives you a new goal in three steps; you build the code yourself by copying code from the example and adapting it. Each step has a **check** cell (✓ go on, ✗ shows what is missing; `checks.py` in the main folder), a hint and a collapsed solution.

The notebooks keep the exercise steps visible and import the technical parts from [helper.py](helper.py): the OpenAI API calls (`describe`, `ask`, `transcribe`, `generate`, `create_pictures`), image encoding, logging, video information, frame extraction and image display. Open it to see how they work.

The supplied synthetic images, reference sheet and clips are under `data/`. All rows in `data/images/labels.csv` are draft labels for the lecturer to verify. Use `null` when no text is visible and `[unreadable]` when visible text cannot be read. Use only the supplied assets or synthetic results from Google Flow or the OpenAI API. Every OpenAI API call in `helper.py`, including Whisper, appends the prompt, model ID, raw answer and, for `gpt-6-luna`, the input and output tokens to a local, Git-ignored `api_log.jsonl` in the exercise folder. Exercise 3 saves its pictures in the Git-ignored folder `variants/`, exercise 4 an optional own character sheet in `reference/`. Clear notebook outputs before sharing notebook edits.

## API models and cost

The notebooks use `gpt-6-luna` for text and vision through the OpenAI Responses API. Fixed answer formats are written as Pydantic classes and sent with `client.responses.parse()` (Structured Outputs). GPT-6 Luna requires an API project with a supported paid usage tier. Standard rates used here are US$0.10 per million input tokens and US$0.50 per million output tokens. Estimates use the reported token counts, charging all input at the uncached rate. [GPT-6 Luna model and pricing](https://developers.openai.com/api/docs/models/gpt-6-luna) · [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Exercise 2 uses `whisper-1` at US$0.006 per audio minute for the slide’s segment timestamps. OpenAI marks this model deprecated and lists February 26, 2027 as the shutdown date. [Whisper pricing](https://developers.openai.com/api/docs/models/whisper-1) · [Speech-to-text timestamps](https://developers.openai.com/api/docs/guides/speech-to-text) · [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)

Exercise 3 creates six pictures – two in the example, four in the task – with `gpt-image-2` at low quality (1024×1024). Image output is billed at US$30 per million tokens; one picture used 196 output tokens, so about US$0.006 per picture and US$0.036 for all six. To change the model, edit `IMAGE_MODEL` in `helper.py`. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)

One pass through all four required exercises is designed to stay below **CHF 1 per student**. With about 50 short GPT-6 Luna calls (examples and tasks together) and about eight seconds of Whisper audio, the OpenAI usage estimate is below US$0.01 at the listed rates, before exchange rates or taxes. Creating the six exercise 3 pictures adds about US$0.036, an optional own character sheet in exercise 4 about US$0.006. The exercise 1 example shows how to compute the exact cost of a call from the tokens the API reports and scale it to a whole folder. Run each paid block once for the required work, since reruns also incur charges. Google Flow credits, Codespaces usage and any initial API billing top-up are separate from per-run API usage.
