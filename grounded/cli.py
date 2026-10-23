"""Command line: search the knowledge base, ask the agent, or run the evals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .corpus import load_corpus
from .retrieval import HybridRetriever, make_embedder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB = ROOT / "examples" / "kb"
DEFAULT_CASES = ROOT / "examples" / "evals" / "questions.jsonl"


def _retriever(args: argparse.Namespace) -> HybridRetriever:
    return HybridRetriever(load_corpus(args.kb), make_embedder(args.embedder))


def _agent(args: argparse.Namespace, retriever: HybridRetriever):
    from .agent import Agent
    from .model import ClaudeModel

    def approve(name: str, data: dict) -> bool:
        if args.yes_writes:
            return True
        if not sys.stdin.isatty():
            return False
        print(f"\nThe agent wants to run {name} with:\n{json.dumps(data, indent=2)}")
        return input("Allow? [y/N] ").strip().lower() == "y"

    model = ClaudeModel(model=args.model, effort=args.effort)
    return Agent(model, retriever, notes_path=Path(args.notes), approve=approve)


def _save_trace(result, runs: Path) -> Path:
    runs.mkdir(exist_ok=True)
    path = runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = asdict(result)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def cmd_search(args: argparse.Namespace) -> None:
    for hit in _retriever(args).search(args.query, k=args.k):
        ranks = ", ".join(f"{k} #{v}" for k, v in hit.ranks.items())
        print(f"{hit.chunk.id:<24} {hit.score:.4f}  ({ranks})  {hit.chunk.heading}")


def cmd_ask(args: argparse.Namespace) -> None:
    result = _agent(args, _retriever(args)).run(args.question)
    print(f"\n[{result.status}] {result.answer or '(no answer)'}\n")
    for p in result.problems:
        print(f"  problem: {p}")
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "n/a"
    print(f"steps {result.steps} | tool calls {result.tool_calls} | repairs {result.repairs} | "
          f"{result.latency_s}s | {result.usage.input_tokens} in / {result.usage.output_tokens} out tokens | {cost}")
    print(f"trace: {_save_trace(result, Path(args.runs))}")


def cmd_eval(args: argparse.Namespace) -> None:
    from .evals import agent_eval, load_cases, retrieval_eval

    retriever = _retriever(args)
    cases = load_cases(args.cases)
    if args.kind == "retrieval":
        print(json.dumps(retrieval_eval(retriever, cases, k=args.k), indent=2))
        return
    if args.limit:
        cases = cases[: args.limit]
    if not args.yes:
        sys.exit(f"The agent eval calls the model {len(cases)} times and costs money. Re-run with --yes to proceed.")
    results, summary = agent_eval(_agent(args, retriever).run, cases)
    for r in results:
        mark = "ok  " if r.correct_behaviour else "MISS"
        print(f"{mark} {r.case.id:<22} {r.run.status:<10} cited={sorted(r.run.sources)}")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="grounded", description=__doc__)
    p.add_argument("--kb", default=str(DEFAULT_KB), help="folder of .md/.txt notes")
    p.add_argument("--embedder", default="auto", choices=["auto", "fastembed", "hashing"],
                   help="vector embeddings: fastembed (semantic, optional extra) or hashing (offline baseline)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="show what retrieval returns for a query (no model call)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    def model_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default="claude-opus-5-5")
        sp.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
        sp.add_argument("--notes", default="notes/flashcards.jsonl", help="where save_flashcard writes")
        sp.add_argument("--yes-writes", action="store_true", help="approve write tools without asking")
        sp.add_argument("--runs", default="runs", help="folder for run traces")

    a = sub.add_parser("ask", help="ask the agent a question")
    a.add_argument("question")
    model_opts(a)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="run the retrieval eval (free) or the agent eval (calls the model)")
    e.add_argument("kind", choices=["retrieval", "agent"])
    e.add_argument("--cases", default=str(DEFAULT_CASES))
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--yes", action="store_true", help="confirm the agent eval may spend money")
    model_opts(e)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
