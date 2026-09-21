# Lab 09 — Retrieval-Augmented Generation

Lab 09 builds a Retrieval-Augmented Generation (RAG) system from scratch, using no RAG
framework — no LangChain, no LlamaIndex, no vector database. Everything is numpy and HTTP
calls, so that every moving part stays visible.

## The problem RAG solves

A large language model knows only what was in its training data. It does not know your
course notes, your company's policies, or anything written after its knowledge cutoff.
Asked about such material it will often produce a fluent, confident and wrong answer — a
hallucination.

Fine-tuning is one answer but it is expensive, slow, needs retraining whenever the
documents change, and still does not let the model cite a source. RAG is the cheaper
answer: leave the model's weights alone and instead **fetch the relevant text at question
time and paste it into the prompt**.

## The pipeline

RAG has an offline indexing phase and an online query phase.

**Indexing (done once):**
1. Load the documents.
2. Chunk them into passages of a few hundred words.
3. Embed each chunk into a vector with an embedding model.
4. Store the vectors.

**Querying (done per question):**
5. Embed the user's question with the *same* embedding model.
6. Retrieve the top-k chunks by cosine similarity to the question vector.
7. Build a prompt containing those chunks plus the question.
8. Generate an answer with the LLM, instructed to use only the supplied context.

## Chunking

Chunk size is the central trade-off. Chunks that are too small get retrieved without enough
surrounding context to answer from. Chunks that are too large dilute the embedding — a
single vector has to represent several unrelated ideas — and waste prompt tokens. A few
hundred words with a modest overlap is the usual starting point; the overlap stops a fact
from being split across a boundary and lost.

## Why cosine similarity

Embedding vectors are compared by the cosine of the angle between them, not by Euclidean
distance, because direction carries the meaning and magnitude does not. If the vectors are
L2-normalised to unit length, cosine similarity is simply the dot product, which makes
retrieval over the whole index one matrix multiplication.

## Evaluating RAG

RAG has two failure modes and they must be measured separately. **Retrieval failure** is
when the right chunk was never fetched — measured with hit rate at k, or recall at k.
**Generation failure** is when the right chunk was fetched but the model still answered
badly — measured with faithfulness, meaning whether every claim in the answer is supported
by the retrieved context. Fixing a retrieval failure by editing the prompt is wasted
effort, which is why you diagnose which of the two you have before changing anything.
