# extractor — Find-My-Object Lens + Mind Palace

This branch extends the base object memory system with a full retrieval-augmented generation (RAG) pipeline. The extractor can now answer open-ended questions about what it has seen — not just "where is X" but "what was I doing when I last had my keys?", synthesised from grounded visual evidence.

---

## What's new on this branch

| File | What it adds |
|---|---|
| `src/local_rag.py` | Semantic similarity retrieval over the memory log |
| `src/llm_reasoner.py` | LLM-synthesised answers from retrieved evidence |
| `src/color_utils.py` | Dominant colour classification for thumbnails |

The base ingestion and query engine from `main` is unchanged. Mind Palace layers on top as an optional reasoning step.

---

## Mind Palace — how it works

```
Natural language question
    │
    ▼
[1] Retrieve candidate events from memory JSONL
    │  Keyword-filtered by label (same as base extractor)
    ▼
[2] local_rag.py — Semantic reranking
    │  Each event's context string is embedded with all-MiniLM-L6-v2
    │  Cosine similarity against the query selects the top-k events
    │
    │  Optional: caption thumbnails with VIT-GPT2 image captioning
    │  Optional: answer sub-questions with ViLT visual QA
    ▼
[3] llm_reasoner.py — Evidence synthesis
    │  Top evidence rows (second, label, colour, context, thumbnail path)
    │  are passed to an LLM with a grounded system prompt
    │  LLM is instructed to cite timestamps and limit speculation
    ▼
Natural language answer with supporting evidence bullets
```

---

## local_rag.py

Provides semantic retrieval and on-thumbnail inference over stored events.

**Semantic similarity search**

```python
from extractor.src.local_rag import semantic_similarity_scores

scores = semantic_similarity_scores(
    query="blue bag near the door",
    texts=["blue backpack on dining table at second 42", "red chair beside sofa"],
)
# Returns cosine similarity scores for each context string
```

**Image captioning**

```python
from extractor.src.local_rag import caption_image

caption = caption_image("data/crops/video/s000042_000_backpack.jpg")
# → "a blue backpack sitting on a wooden table"
```

**Visual question answering**

```python
from extractor.src.local_rag import visual_qa_answers

results = visual_qa_answers(
    question="What color is this?",
    image_paths=["path/to/thumbnail.jpg"],
)
# → [{"answer": "blue", "score": 0.92}]
```

**Models used**

| Task | Model | Backend |
|---|---|---|
| Semantic embedding | `all-MiniLM-L6-v2` | Sentence Transformers |
| Image captioning | `nlpconnect/vit-gpt2-image-captioning` | Transformers / ONNX/QNN |
| Visual QA | `dandelin/vilt-b32-finetuned-vqa` | Transformers / ONNX/QNN |

All models prefer cached local artifacts and fall back to HuggingFace Hub download on first use. Set `AURA_FORCE_OFFLINE=1` to disable network access entirely.

---

## llm_reasoner.py

Takes a question and a list of retrieved evidence events, calls an OpenAI-compatible API, and returns a grounded answer.

```python
from extractor.src.llm_reasoner import synthesize_answer

answer = synthesize_answer(
    question="Where did I leave my laptop?",
    evidence=[
        {
            "video_second": 84,
            "label": "laptop",
            "detected_color": "gray",
            "context": "on dining table",
            "confidence": 0.87,
            "thumbnail_path": "data/crops/...",
        }
    ],
)

if answer["ok"]:
    print(answer["answer"])
```

The LLM is given a strict system prompt: answer using only the provided evidence rows, cite at least one timestamp, keep the answer to 2–4 sentences.

**Configuration via environment variables**

| Variable | Default | Description |
|---|---|---|
| `AURA_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `AURA_LLM_MODEL` | `gpt-4o-mini` | Model to use |
| `AURA_LLM_API_KEY` | *(none)* | API key |

Works with any OpenAI-compatible endpoint — OpenAI, Ollama (via `/v1` compatibility mode), Groq, etc.

---

## Base extractor (unchanged from main)

Everything from the base `extractor/README.md` applies here — ingestion, tracking, JSONL storage, and keyword-based queries still work identically.

```bash
# Ingest
python extractor/run_ingest.py --video path/to/video.mp4

# Basic keyword query (no LLM)
python extractor/run_query.py --video path/to/video.mp4 --query "where is my laptop"
```

---

## Mind Palace UI

`demo-ui/app.py` on this branch is a dedicated Streamlit interface for the full Mind Palace experience:

```bash
streamlit run demo-ui/app.py
```

The UI adds a **Ask Aura** panel where you type a free-form question. The system retrieves evidence from memory, optionally captions relevant thumbnails, and displays the LLM-synthesised answer alongside the supporting evidence rows.

---

## Requirements

```bash
pip install -r extractor/requirements.txt
```

Additional deps on this branch: `sentence-transformers`, `transformers`, `optimum[onnxruntime]`

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `AURA_PREFER_QNN` | `1` | Prefer QNN execution provider for ONNX models |
| `AURA_FORCE_OFFLINE` | `0` | Disable all network model downloads |
| `AURA_LLM_BASE_URL` | OpenAI | LLM API endpoint for answer synthesis |
| `AURA_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `AURA_LLM_API_KEY` | *(none)* | API key |
