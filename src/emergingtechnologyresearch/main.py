#!/usr/bin/env python
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

try:
    from langfuse import get_client
except ImportError:
    get_client = None  # type: ignore[assignment]

from .crew import Emergingtechnologyresearch

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

if load_dotenv:
    load_dotenv()


def _configure_environment() -> None:
    """Set hard-coded defaults without overwriting the user's configuration."""
    os.environ.setdefault("MODEL", "bedrock/us.amazon.nova-pro-v1:0")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")


def _build_inputs(topic: str = "AI LLMs") -> Dict[str, str]:
    """Generate the shared prompt/context payload for Crew calls."""
    return {
        "topic": topic,
        "current_year": str(datetime.now().year),
    }


def _require_args(count: int) -> list[str]:
    """Raise early when the caller does not supply the required CLI parameters."""
    if len(sys.argv) - 1 < count:
        raise ValueError(f"expected at least {count} CLI argument(s); got {len(sys.argv) - 1}")
    return sys.argv[1:]


def _launch_crew() -> Emergingtechnologyresearch:
    return Emergingtechnologyresearch()


def _init_langfuse_client() -> Optional[Any]:
    """Create the Langfuse client if the dependency is available."""
    if not get_client:
        return None
    return get_client()


_configure_environment()

langfuse_client = _init_langfuse_client()

if langfuse_client:
    if langfuse_client.auth_check():
        print("Langfuse client authenticated successfully.")
    else:
        print("Langfuse authentication failed. Check credentials.")
else:
    print("Langfuse is not installed; install langfuse or pin it in pyproject.toml and rerun.")


def run():
    inputs = _build_inputs()

    try:
        _launch_crew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e


def train():
    inputs = _build_inputs()
    iter_arg, filename = _require_args(2)

    try:
        _launch_crew().crew().train(
            n_iterations=int(iter_arg),
            filename=filename,
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while training the crew: {e}") from e


def replay():
    (task_id,) = _require_args(1)

    try:
        _launch_crew().crew().replay(task_id=task_id)
    except Exception as e:
        raise RuntimeError(f"An error occurred while replaying the crew: {e}") from e


def test():
    inputs = _build_inputs()
    iter_arg, eval_llm = _require_args(2)

    try:
        _launch_crew().crew().test(
            n_iterations=int(iter_arg),
            eval_llm=eval_llm,
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while testing the crew: {e}") from e


def run_with_trigger():
    import json

    if len(sys.argv) < 2:
        raise ValueError("No trigger payload provided.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload provided.") from exc

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": "",
    }

    try:
        result = _launch_crew().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise RuntimeError(f"An error occurred while running with trigger: {e}") from e
