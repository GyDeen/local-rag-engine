"""Compatibility imports for the secondary cosine evaluator."""

from rag_engine.evaluation.cosine.result_evaluation import (
    evaluate_answer,
    evaluate_output_files,
    evaluate_result,
)

__all__ = ["evaluate_answer", "evaluate_output_files", "evaluate_result"]
