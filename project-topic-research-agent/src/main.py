"""
Topic Research Agent — Main Entry Point
=========================================
Run the research agent from the command line.

Usage (from project-topic-research-agent):
    python src/main.py --framework langgraph --topic "AI in healthcare"
    python src/main.py --framework adk --topic "Climate change solutions"
    python src/main.py --framework langgraph --topic "AI agents" --no-trace
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Resolve paths: src/ for imports, repo root for shared config/.env
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)

sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(_REPO_ROOT, "config", ".env"))
load_dotenv()


def run_research_with_validation(topic: str, framework: str):
    """Validate + sanitize topic, then run LangGraph or ADK research agent."""
    from middlewares.safety_guard import sanitize_input, validate_input

    ok, reason = validate_input(topic)
    if not ok:
        raise ValueError(reason)
    clean = sanitize_input(topic)

    if framework == "langgraph":
        from agents.langgraph_agent import run_research

        return run_research(clean)

    if framework == "adk":
        import asyncio

        from agents.adk_agent import run_research

        text = asyncio.run(run_research(clean))
        return {"topic": clean, "research_text": text, "framework": "adk"}

    raise ValueError(f"Unknown framework: {framework}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic Research Agent")
    parser.add_argument(
        "--framework",
        choices=["langgraph", "adk"],
        default="langgraph",
        help="Agent framework to use",
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Research topic to investigate",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable Phoenix tracing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    tracing_active = False

    if not args.no_trace:
        try:
            from config.phoenix_config import setup_tracing

            setup_tracing()
            tracing_active = True
        except ImportError:
            print("Phoenix not available. Running without tracing.")
        except Exception as e:
            print(f"Phoenix setup failed: {e}. Running without tracing.")

    print("\nTopic Research Agent")
    print("=" * 50)
    print(f"Framework: {args.framework}")
    print(f"Topic:     {args.topic}")
    print(f"Tracing:   {'enabled' if tracing_active else 'disabled'}")
    print("=" * 50 + "\n")

    try:
        result = run_research_with_validation(args.topic, args.framework)

        print("=" * 50)
        print("RESEARCH COMPLETE")
        print("=" * 50)
        if isinstance(result, dict):
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)

    except ValueError as e:
        print(f"\n[Validation] {e}")
    except Exception as e:
        print(f"\n[Error] {type(e).__name__}: {e}")

    if tracing_active:
        print("\n" + "=" * 50)
        print("Phoenix dashboard is running at http://localhost:6006")
        print("Open the URL above to inspect LLM call traces.")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("> ").strip().lower()
                if user_input in ("exit", "quit"):
                    print("Shutting down...")
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down...")
                break

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
