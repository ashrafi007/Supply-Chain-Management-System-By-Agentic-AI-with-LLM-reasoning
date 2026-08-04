"""Public exports for the orchestrator package: build_executor(), LangGraphExecutor."""

from src.orchestrator.executor import LangGraphExecutor

__all__ = ["LangGraphExecutor", "build_executor"]


def build_executor() -> LangGraphExecutor:
    return LangGraphExecutor()
