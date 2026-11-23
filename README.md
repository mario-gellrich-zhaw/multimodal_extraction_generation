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
| [1. Image → JSON → DataFrame](01_Image_Extraction/image_extraction.ipynb) | Describe one image, ask for a fixed JSON format, extract three images (parking lot, alpine pasture, city street) into a table and score each field against draft labels. |
| [2. Video → frames → JSON timeline](02_Video_Extraction/video_extraction.ipynb) | Cut a clip into one frame per second, describe the frames, transcribe the speech and merge both on one timeline. |
| [3. Prompt experiment – Google Flow and OpenAI API](03_Image_Generation/image_generation.ipynb) | Write a prompt from building blocks, vary one block, let the OpenAI API create two pictures per version and rate one feature by eye and with the OpenAI API. |
| [4. Storyboard → clips → measure consistency](04_Video_Generation/video_generation.ipynb) | Let the OpenAI API draft a storyboard, generate one clip and check whether the character’s look matches the sheet. |

Each exercise folder has two notebooks. In the notebook from the table, some code is missing: where you see `# TODO`, replace `...` with your own code. The finished version is next to it with the ending `_solution.ipynb`, for example `image_extraction_solution.ipynb`. Try the exercise first, then compare.

The notebooks keep the exercise steps visible and import the technical parts from [helper.py](helper.py): the OpenAI API calls (`describe`, `ask`, `transcribe`), image encoding, logging, frame extraction, image display and the upload check. Open it to see how they work.

The supplied synthetic images, reference sheet and clips are under `data/`. All rows in `data/images/labels.csv` are draft labels for the lecturer to verify. Use `null` when no text is visible and `[unreadable]` when visible text cannot be read. Use only the supplied assets or synthetic Google Flow results. Every OpenAI API call in `helper.py`, including Whisper, appends the prompt, model ID and raw answer to a local, Git-ignored `api_log.jsonl` in the exercise folder. Clear notebook outputs before sharing notebook edits.

## API models and cost

The notebooks use `gpt-6-luna` for text and vision through the OpenAI Responses API. Fixed answer formats are written as Pydantic classes and sent with `client.responses.parse()` (Structured Outputs). GPT-6 Luna requires an API project with a supported paid usage tier. Standard rates used here are US$0.10 per million input tokens and US$0.50 per million output tokens. Estimates use the reported token counts, charging all input at the uncached rate. [GPT-6 Luna model and pricing](https://developers.openai.com/api/docs/models/gpt-6-luna) · [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Exercise 2 uses `whisper-1` at US$0.006 per audio minute for the slide’s segment timestamps. OpenAI marks this model deprecated and lists February 26, 2027 as the shutdown date. [Whisper pricing](https://developers.openai.com/api/docs/models/whisper-1) · [Speech-to-text timestamps](https://developers.openai.com/api/docs/guides/speech-to-text) · [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)

Exercise 3 creates its four pictures with `gpt-image-1-mini` at low quality (1024×1024). Image output is billed at US$8 per million tokens; one picture used 272 output tokens, so about US$0.002 per picture and US$0.01 for all four. OpenAI marks this model deprecated; if it is shut down, change `IMAGE_MODEL` in `helper.py`. [gpt-image-1-mini model and pricing](https://developers.openai.com/api/docs/models/gpt-image-1-mini)

One pass through all four required exercises is designed to stay below **CHF 1 per student**. With 27 short GPT-6 Luna calls and about eight seconds of Whisper audio, the OpenAI usage estimate is below US$0.10 at the listed rates, before exchange rates or taxes. Creating the four exercise 3 pictures adds about US$0.01. Exercise 1 shows how to estimate the cost of a batch before you run it: tokens per image × number of images × price. Run each paid block once for the required work, since reruns also incur charges. Google Flow credits, Codespaces usage and any initial API billing top-up are separate from per-run API usage.
