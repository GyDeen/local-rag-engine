import logging
from collections.abc import Callable

from rag_engine.logging_config import configure_logging

logger = logging.getLogger(__name__)


def run_cli(main: Callable[[], int]) -> int:
    """Log failed commands and return nonzero, preserving parser exits/interrupts.

    Importable main functions still raise normally for library callers.
    """
    try:
        configure_logging()
        return main()
    except Exception:
        logger.exception("Command failed: %s", main.__module__)
        return 1
