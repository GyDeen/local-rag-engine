import json

from rag_engine.engine.llm.llm_callers import call_ollama_llm
from rag_engine.engine.rerank.llm_reranker.strategy import LLMCaller


def make_ollama_reranker(
    *,
    model: str,
    host: str,
    context_length: int = 16384,
) -> LLMCaller:
    """Reuse the LLM transport with a schema and bounded ranking response."""

    def caller(system_prompt: str, user_prompt: str) -> str:
        candidates = json.loads(user_prompt)["candidates"]
        ids = [c["chunk_id"] for c in candidates]
        references = {
            "type": "array",
            "items": {"type": "string", "enum": ids},
            "maxItems": len(ids),
            "uniqueItems": True,
        }
        strings = {"type": "array", "items": {"type": "string"}, "maxItems": 3}
        properties = {
            "ranked_chunk_ids": references,
            "coverage": {
                "type": "object",
                "properties": {"explanation": references, "resolution": references},
                "required": ["explanation", "resolution"],
                "additionalProperties": False,
            },
            "sufficient": {"type": "boolean"},
            "missing": strings,
            "search_queries": {**strings, "maxItems": 2},
            "operator_questions": strings,
            "reason": {"type": "string"},
        }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        return call_ollama_llm(
            system_prompt,
            user_prompt,
            model=model,
            host=host,
            context_length=context_length,
            response_schema=schema,
            max_new_tokens=1200,
        )

    return caller
