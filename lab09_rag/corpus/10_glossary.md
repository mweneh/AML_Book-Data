# DSA 8401 — Glossary of Terms

**Chunk** — a passage of a document small enough to embed as a single vector and to paste
into a prompt.

**Cosine similarity** — the cosine of the angle between two vectors, ranging from -1 to 1.
Equal to the dot product when both vectors are normalised to unit length.

**Data leakage** — information available to the model during training that would not be
available at prediction time. Produces optimistic scores that collapse in production.

**Embedding** — a dense numeric vector representing a piece of text, positioned so that
texts with similar meaning sit close together in the vector space.

**Epoch** — one complete pass of the training algorithm over the entire training set.

**Hallucination** — a fluent, confident and factually wrong statement produced by a
language model, typically about something absent from its training data.

**Hit rate at k** — the fraction of evaluation questions for which the correct source
document appears among the top k retrieved chunks. The standard retrieval metric.

**Learning rate** — the step size taken along the gradient at each update. Too high and
training diverges; too low and it crawls or gets stuck.

**MAE (Mean Absolute Error)** — the mean of the absolute differences between prediction and
truth, reported in the units of the target. More robust to outliers than RMSE.

**Overfitting** — the model has memorised the training data, including its noise. Diagnosed
by a large gap between training and validation performance.

**PR-AUC** — area under the precision-recall curve. Preferred to ROC-AUC when the positive
class is rare.

**Prompt** — the full text sent to a language model, including any system instruction,
retrieved context and the user's question.

**RAG (Retrieval-Augmented Generation)** — fetching relevant documents at question time and
supplying them to a language model as context, instead of relying on its trained weights.

**ROC-AUC** — the probability that a randomly chosen positive example is ranked above a
randomly chosen negative one. 0.5 is random guessing; 1.0 is perfect ranking.

**Temperature** — a sampling parameter controlling randomness in language model output. A
temperature of 0 makes output near-deterministic and is the right choice for RAG, where you
want the answer pinned to the retrieved context rather than creative.

**Token** — the unit a language model reads and bills in, roughly three quarters of an
English word.

**Top-k retrieval** — returning the k most similar chunks to the query vector.

**Vanishing gradient** — the shrinking of the gradient signal as it is backpropagated
through many layers or time steps, which stops early layers from learning. Addressed by
ReLU activations, residual connections, and gated recurrent units such as the LSTM.
