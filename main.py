"""Command-line interface for the RAG search tool.

Usage:
  python main.py ingest <folder>      # build the index from .txt/.md files
  python main.py ask "<question>"     # ask a one-off question
  python main.py chat                 # interactive Q&A loop
"""

import sys

import rag


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    command = args[0]

    if command == "ingest":
        if len(args) < 2:
            sys.exit("Usage: python main.py ingest <folder>")
        rag.ingest(args[1])

    elif command == "ask":
        if len(args) < 2:
            sys.exit('Usage: python main.py ask "<question>"')
        print(rag.ask(" ".join(args[1:])))

    elif command == "chat":
        print("RAG chat — type a question, or 'exit' to quit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                print(f"\n{rag.ask(q)}\n")

    else:
        sys.exit(f"Unknown command {command!r}. Use: ingest | ask | chat")


if __name__ == "__main__":
    main()
