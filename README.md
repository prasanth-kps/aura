<div align="center">

# AURA — Mind Palace
### Augmented Understanding & Relational Archive

**Feature branch: `feature/mind-palace-memory`**

*Extends AURA's object memory with semantic retrieval and LLM-synthesised answers.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm%20AI%20Hub-NPU--first-3253DC?style=flat-square)](https://aihub.qualcomm.com/)

</div>

---

## What's new on this branch

The base AURA project ([`main`](https://github.com/prasanth-kps/aura)) can track objects across video and answer keyword queries like *"where is my laptop?"*

Mind Palace replaces the keyword-matching query layer with a full **retrieval-augmented generation (RAG)** pipeline:

1. **Semantic retrieval** — queries and stored event contexts are embedded with `all-MiniLM-L6-v2` and ranked by cosine similarity, so natural phrasings find the right events even when the words don't exactly match
2. **Visual inference on demand** — the top retrieved thumbnails are passed through a VIT-GPT2 image captioner and a ViLT VQA model to enrich the evidence
3. **LLM synthesis** — retrieved evidence rows are sent to an OpenAI-compatible LLM with a strict grounding prompt; the answer is always tied to timestamps and spatial context from the memory log

The result: instead of *"Last seen at second 84: laptop is on dining table"*, you get a natural, reasoned answer with evidence citations.

---

## New files on this branch

| File | Purpose |
|---|---|
| `extractor/src/local_rag.py` | Semantic embedding, image captioning, visual QA |
| `extractor/src/llm_reasoner.py` | LLM synthesis from grounded evidence |
| `extractor/src/color_utils.py` | Dominant colour classification |
| `demo-ui/app.py` | Mind Palace Streamlit UI |

---

## Running Mind Palace

```bash
git checkout feature/mind-palace-memory
pip install -r requirements.txt

# 1. Ingest a video (same as base)
python extractor/run_ingest.py --video path/to/video.mp4

# 2. Launch the Mind Palace UI
streamlit run demo-ui/app.py
```

Set your LLM credentials before starting:

```bash
export AURA_LLM_API_KEY=sk-...
export AURA_LLM_BASE_URL=https://api.openai.com/v1   # or Ollama /v1 endpoint
export AURA_LLM_MODEL=gpt-4o-mini
```

---

## How a query flows

```
User: "Where did I leave my blue bag?"
    │
    ▼
Label extraction → "bag" (canonicalised)
Colour extraction → "blue"
    │
    ▼
Keyword filter: all "bag" events from memory JSONL
    │
    ▼
Semantic reranking: MiniLM-L6-v2 scores each event context
    against the full query string
Top-k events selected
    │
    ▼
Optional: caption thumbnails (VIT-GPT2)
Optional: run VQA on thumbnails ("what color is this?")
    │
    ▼
Evidence block assembled:
  second=42, label=bag, color=blue, context=on dining table,
  confidence=0.83, thumbnail_path=...
    │
    ▼
LLM (gpt-4o-mini / Ollama):
  "At second 42, I can see a blue bag on the dining table..."
```

---

## For the full project overview

See the [main branch README](https://github.com/prasanth-kps/aura) for the complete Aura architecture, setup instructions, hardware acceleration details, and all three lenses.

The Mind Palace extractor module is documented in detail at [extractor/README.md](extractor/README.md).

---

## Built by

[Prasanth KPS](https://github.com/prasanth-kps) · [@prasanth-kps](https://github.com/prasanth-kps)
