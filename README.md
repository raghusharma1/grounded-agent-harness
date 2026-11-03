# grounded-agent-harness

A small agent harness in Python that answers questions **only from evidence it retrieved**, cites it line by line, and abstains when the knowledge base does not cover the question. It combines hybrid RAG, a bounded tool-calling loop on Claude, scoped tools with approval gates, a citation verifier, and an eval suite.

Illustrative output:

```
$ grounded ask "How does a token bucket allow bursts?"

[grounded] A token bucket refills at a steady rate but can hold up to a fixed number of
tokens, so a full bucket can be spent at once [rate-limiting#2].
- Over time the average rate is still enforced, because tokens only come back at the
  refill rate [rate-limiting#2].

steps 2 | tool calls 1 | repairs 0 | ... tokens | $...
```

## Why

The usual failure of a documentation assistant is not a wrong retrieval. It is a fluent answer whose citations point at nothing, or at a passage that was never retrieved. This harness makes "answer from the evidence" a mechanical check instead of a hope in the prompt:

- every claim line must cite a chunk id;
- every cited id must be one a tool actually returned **in this run**;
- an answer that fails gets one repair turn with the verifier's findings, and if it fails again it is **withheld**, not shown;
- "the notes don't cover this" is a first-class outcome (`INSUFFICIENT_EVIDENCE`), scored as correct in the evals.

Passing verification doesn't prove an answer is right. It proves every claim points at text the model was shown, which is what lets a person check it.

## How it works

```
question
   │
   ▼
┌───────────────────────── agent loop (max_steps) ─────────────────────────┐
│  Claude  ──tool_use──▶  ToolRegistry ──▶ search_knowledge / read_chunk    │
│    ▲                    (schema check,    │   HybridRetriever: BM25 + vectors, RRF
│    │                     risk levels,     │   every returned chunk id → evidence ledger
│    │                     write approval)  │
│    └──────────── tool_result (all results in one message) ◀────────────── │
│                                                                           │
│  final text ──▶ verify(answer, evidence ledger)                           │
│                   grounded / abstained ──▶ return                         │
│                   rejected ──▶ one repair turn ──▶ else withhold          │
└───────────────────────────────────────────────────────────────────────────┘
   │
   ▼
RunResult: status, answer, citations, steps, tokens, cost, latency, trace
```

## Design decisions

**Hybrid retrieval, measured.** BM25 catches exact identifiers ("LRU", "429"); embeddings catch paraphrase ("evict whatever nobody has read for the longest time"). Reciprocal rank fusion merges the two rank lists without calibrating their score scales. Results on the bundled eval set (22 answerable questions, 6 of them deliberate paraphrases):

| Retriever | recall@1 | MRR@3 |
|---|---|---|
| BM25 only | 0.955 | 0.977 |
| Vectors, hashing embedder (offline baseline) | 0.773 | 0.856 |
| Hybrid with hashing embedder | 0.909 | 0.955 |
| Vectors, `bge-small-en-v1.5` via fastembed | **1.000** | **1.000** |
| Hybrid with `bge-small-en-v1.5` | **1.000** | **1.000** |

The hashing embedder is deliberately weak (it matches vocabulary, not meaning) and keeps the tests dependency-free; the table shows why the real model matters, and that fusing a weak vector signal can *lower* precision. Reproduce with `grounded --embedder hashing|fastembed eval retrieval -k 1` for recall@1 and `-k 3` for MRR@3.

**Scoped tools.** Each tool declares a risk level. Inputs are validated against the JSON schema even though tools are sent with `strict: true`. Write tools (here, `save_flashcard`) run only with explicit approval, and a flashcard may only cite chunks retrieved in the same run. Errors go back to the model as `is_error` tool results it can explain, never as crashes.

**A bounded, append-only loop.** A step budget caps every run. A turn cut off by `max_tokens` never has its half-written tool call executed. A refusal ends the run with a clear status. Parallel tool calls are answered in a single message. The conversation is only ever appended to, so prompt caching keeps working across turns, and the trace is an exact record of what the model saw.

**Cost and latency are part of the result.** Every run reports steps, tool calls, tokens, an estimated dollar cost and latency, and saves a JSON trace under `runs/`. The quality-versus-cost dial is `--effort` (Claude Opus 5.5 defaults to `medium`).

**Testable without an API key.** `ScriptedModel` replays canned responses with the same shape as the SDK's, so the loop, tools, verifier and evals are covered by 27 offline tests that run in seconds.

**Why a hand-written loop and not a framework.** The point of the project is the harness itself: where the budget is enforced, where verification interrupts, what goes into the trace. Each piece maps onto a LangGraph node or a LangChain tool if you want to port it.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,embeddings]"                 # embeddings is optional (local, no API key)
pytest                                             # 27 offline tests

grounded search "what is a dead-letter queue"    # see retrieval, no model call
grounded eval retrieval -k 1                     # free retrieval eval

export ANTHROPIC_API_KEY=...                       # or `ant auth login`
grounded ask "Why does hash mod N break when a cache node is added?"
grounded ask "Make me a flashcard about token buckets"   # asks before writing
grounded eval agent --yes                        # end-to-end eval, calls the model
```

Point it at your own notes with `--kb path/to/folder` (Markdown or text files).

## Evals

`grounded eval agent` scores behaviour, not just text: did the agent give a verified, cited answer when the notes cover the question, abstain when they don't, and cite the expected source? It also reports unverified answers, repair turns, mean steps, mean latency and total cost. The bundled set has 25 questions, three of them deliberately outside the knowledge base.

## Layout

```
grounded/
  corpus.py     load notes, chunk by heading with overlap, stable chunk ids
  retrieval.py  BM25, vector index (numpy or FAISS), embedders, RRF hybrid
  tools.py      tool registry: schemas, risk levels, approval, evidence ledger
  verify.py     citation and abstention checks
  model.py      Claude adapter (Anthropic SDK), cost accounting, ScriptedModel
  agent.py      the bounded loop with verify-and-repair
  evals.py      retrieval and agent evals
  cli.py        search / ask / eval
examples/kb     sample knowledge base (system-design study notes)
examples/evals  eval questions, including unanswerable ones
tests/          offline tests
```

## Limitations and next steps

- The bundled knowledge base is small, so retrieval scores saturate; it is there to make the harness runnable, not to benchmark retrievers.
- The verifier checks that claims point at retrieved text, not that the text supports them. Next: a claim-support check (an LLM judge or an entailment model) scored in the agent eval.
- Streaming output, a LangGraph port, and exposing the knowledge tools as an MCP server.

## License

MIT
