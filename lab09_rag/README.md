# Lab 09 — Building a Simple RAG System

DSA 8401 Applied Machine Learning. A beginner-friendly lab that builds a working RAG system
using only `requests` and `numpy` — no LangChain, no vector database, nothing hidden.

**Core question:** can we make an AI answer questions about *our own* course notes — notes
it has never seen before?

## Contents

| Path | What it is |
|---|---|
| `lab09_rag.ipynb` | The lab notebook (22 short code cells) |
| `corpus/` | Our notes: 10 small files on the course outline, Labs 02–09, and a glossary |
| `requirements.txt` | The libraries needed |
| `.env.example` | Template for your API key — copy this to `.env` |
| `.venv/` | Virtual environment (already set up) |

## Setup

The virtual environment and Jupyter kernel are already created. You only need an API key:

1. Get a free key from https://openrouter.ai/keys
2. Copy `.env.example` to `.env` and paste your key inside.
3. Open `lab09_rag.ipynb` and choose the kernel **Python (AML Lab09 RAG)**.
4. Run **Step 3** first — it sends one tiny message to the AI. If that works, everything
   else will.

**Never upload your `.env` file or your API key.** `.env` is already gitignored here.

## Models

Both come through one [OpenRouter](https://openrouter.ai) key, and both are set in Step 2
of the notebook.

| Role | Model |
|---|---|
| Embeddings (text → numbers) | `perplexity/pplx-embed-v1-0.6b` |
| Chat (writes the answer) | `google/gemini-3.8-flash` |

## The 5 steps the notebook builds

```
   1. LOAD     read our notes from files
   2. CHUNK    cut them into pieces of 150 words
   3. EMBED    turn each piece into a list of numbers
   ------------------------------------------------ (done once)
   4. SEARCH   find the pieces that match the question
   5. ASK      give those pieces + the question to the AI
```

Steps 1–3 happen once. Steps 4–5 happen every time someone asks a question.

The notebook then checks whether it actually works:

- **A question the notes can't answer** — the system should refuse, not invent an answer.
- **With RAG vs without RAG** — the same AI, the same question, one with notes and one
  without. This is the cell that shows why RAG is worth building.
- **Hit rate** — 8 test questions where we already know the correct file. How often does
  the search find it? What happens as we retrieve more chunks?

## Main takeaway

The AI does not get smarter. It is the exact same model both times. RAG just hands it the
right page to read — and because the answer cites a file, the student can open that file
and check it.

## Cost

A full run costs well under a dollar. Most of the calls are embeddings, which are very
cheap, plus about ten calls to Gemini Flash.

## Submission

The notebook with all cells run and output visible, Exercises 1–4 completed, and a sentence
saying whether you used an AI assistant and what for.
