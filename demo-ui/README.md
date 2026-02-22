# Aura Demo UI

Presentation-focused UI/UX wrapper for the `extractor` pipeline.

This app does **not** modify or replace `extractor` logic.  
It only calls existing scripts:
- `extractor/run_ingest.py`
- `extractor/run_query.py`

## Why this folder exists

You asked for a completely separate demo platform so presentation polish can improve without risking backend regressions.  
This folder is isolated and can be deleted independently.

## Features

- Ingest video with the same options used by `extractor`.
- Ask natural-language queries (e.g., `where is my black bottle`).
- Show matched event details and thumbnail.
- View timeline of matching events.
- Browse labels and `instance_id` values.
- Display exact command used (good for transparency during demos).

## Quick Start

From repository root (`aura/`):

```bash
cd demo-ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install extractor dependencies if not already installed in this environment:

```bash
pip install -r ../extractor/requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Usage flow

1. Open **Ingest** tab and provide a video path.
2. Click **Run ingestion** (keep reset enabled for clean stage reruns).
3. Open **Ask** tab and query:
   - `where is my black bottle`
4. Use **Timeline** and **Explore** tabs for richer demo storytelling.

## Notes

- UI default output path is `aura/demo_data/`.
- If your Python executable differs from your extractor env, set it in the sidebar.
- The app remains separate from extractor internals and does not patch extractor files.
