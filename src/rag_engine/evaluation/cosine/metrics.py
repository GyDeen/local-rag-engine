"""Secondary semantic-similarity and citation metrics."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, Callable, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

CitationKey: TypeAlias = tuple[str, tuple[str, ...]]
EvaluateField: TypeAlias = str | tuple[str, ...] | CitationKey
Item = TypeVar("Item", bound=EvaluateField)


class ItemMatch(TypedDict):
    """One golden item and its strongest generated match."""

    reference_index: int
    generated_index: int | None
    reference: EvaluateField
    generated: EvaluateField | None
    score: float


class MatchReport(TypedDict):
    """Golden-item matches and generated items selected by no golden item."""

    score: float
    precision: float
    recall: float
    matches: list[ItemMatch]
    unmatched_generated: Sequence[EvaluateField]


def precision_score(
    reference_items: Collection[Hashable],
    generated_items: Collection[Hashable],
) -> float:
    """Measure how many generated items are present in the reference set"""

    reference_set = set(reference_items)
    generated_set = set(generated_items)
    if not generated_set:
        return 1.0 if not reference_set else 0.0
    return len(reference_set & generated_set) / len(generated_set)


def recall_score(
    reference_items: Collection[Hashable],
    generated_items: Collection[Hashable],
) -> float:
    """Measure how many reference items are present in the generated set"""

    reference_set = set(reference_items)
    generated_set = set(generated_items)
    if not reference_set:
        return 1.0
    return len(reference_set & generated_set) / len(reference_set)


def f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall"""

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match_score(
    reference_item: EvaluateField,
    generated_item: EvaluateField,
) -> float:
    """Return one for identical structured items, otherwise zero"""

    return float(reference_item == generated_item)


def semantic_similarity(
    reference: str,
    generated: str,
    embedder: SentenceTransformer,
) -> float:
    """Compare the meaning of two strings using normalized embeddings.
    Used in field such as: summary, likely causes, resolution steps"""

    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0

    reference_embedding, generated_embedding = embedder.encode(
        [reference, generated],
        normalize_embeddings=True,
    )
    score = sum(
        float(reference_value) * float(generated_value)
        for reference_value, generated_value in zip(
            reference_embedding,
            generated_embedding,
            strict=True,
        )
    )
    return min(1.0, max(0.0, score))


def ordered_list_similarity(
    reference: list[str],
    generated: list[str],
    embedder: SentenceTransformer,
) -> float:
    """Compare ordered text items through a monotonic one-to-one alignment."""

    return ordered_item_matches(reference, generated, embedder)["score"]


def ordered_item_matches(
    reference: list[str],
    generated: list[str],
    embedder: SentenceTransformer,
) -> MatchReport:
    """Match ordered steps one-to-one while preserving their sequence."""

    return _match_reference_items(
        reference,
        generated,
        lambda reference_item, generated_item: semantic_similarity(
            reference_item,
            generated_item,
            embedder,
        ),
        preserve_order=True,
    )


def unordered_list_similarity(
    reference: list[str],
    generated: list[str],
    embedder: SentenceTransformer,
) -> float:
    """Compare unordered text through each golden item's best generated match."""

    return match_items(
        reference,
        generated,
        embedder,
    )["score"]


def citation_key(citation: dict) -> CitationKey:
    """Create an exact citation identity from its path and section hierarchy.
    It will extract the path of the citation file, and the sections being used and flatten into a tupple
    """

    # Get potential multiple sections
    section = citation.get("section", "")
    if isinstance(section, list):
        section_path = tuple(str(heading) for heading in section)
    else:
        section_path = (str(section),) if section else ()
    return str(citation.get("path", "")), section_path


def citation_scores(reference: list[dict], generated: list[dict]) -> dict[str, float]:
    """Return exact structured citation precision, recall and F1"""

    reference_citations = {citation_key(citation) for citation in reference}
    generated_citations = {citation_key(citation) for citation in generated}
    precision = precision_score(reference_citations, generated_citations)
    recall = recall_score(reference_citations, generated_citations)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def citation_match_details(
    reference: list[dict],
    generated: list[dict],
) -> MatchReport:
    """Match every golden citation to its strongest exact generated citation.
    Offer a detail report about which golden standard citation has low matching score.
    Also which generated citation couldn't find a match
    """

    reference_citations = list(
        dict.fromkeys(citation_key(citation) for citation in reference)
    )
    generated_citations = list(
        dict.fromkeys(citation_key(citation) for citation in generated)
    )
    return _match_reference_items(
        reference_citations,
        generated_citations,
        exact_match_score,
    )


def _validate_metric_inputs(
    reference_items: Collection[Item],
    ranked_items: Sequence[Item],
    k: int,
) -> None:
    """Ensure the output of a retrieval model is compatible with ranking metrics"""
    if k < 1:
        raise ValueError("k must be at least 1")


def _resolve_similarity(
    similarity: Callable[[Item, Item], float] | None,
    embedder: SentenceTransformer | None,
) -> Callable[[Item, Item], float]:
    """Use semantic similarity by default and allow explicit exact matching."""

    if similarity is not None:
        return similarity
    if embedder is None:
        raise ValueError("An embedder is required for semantic at-k evaluation")

    def compare_semantically(reference_item: Item, generated_item: Item) -> float:
        if not isinstance(reference_item, str) or not isinstance(generated_item, str):
            raise TypeError("Semantic at-k evaluation requires string items")
        return semantic_similarity(reference_item, generated_item, embedder)

    return compare_semantically


def match_items(
    reference_items: Collection[str],
    generated_items: Sequence[str],
    embedder: SentenceTransformer,
) -> MatchReport:
    """Match every golden text item to its strongest generated text item."""

    return _match_reference_items(
        list(reference_items),
        list(generated_items),
        lambda reference_item, generated_item: semantic_similarity(
            reference_item,
            generated_item,
            embedder,
        ),
    )


def _match_reference_items(
    reference_items: Sequence[Item],
    generated_items: Sequence[Item],
    similarity: Callable[[Item, Item], float],
    *,
    preserve_order: bool = False,
) -> MatchReport:
    """Match golden items and report generated items that were not used."""

    if preserve_order:
        return _match_ordered_reference_items(
            reference_items,
            generated_items,
            similarity,
        )

    # Calculate similarity score for each generated and golden answer pair
    # similarity_matrix[i][j] means the similarity between i-th reference answer and j-th generated output
    similarity_matrix = [
        [
            similarity(reference_item, generated_item)
            for generated_item in generated_items
        ]
        for reference_item in reference_items
    ]

    # Give every golden item its highest-scoring generated output
    matches: list[ItemMatch] = []
    for reference_index, reference_item in enumerate(reference_items):
        # If there is no input generated field
        if not generated_items:
            matches.append(
                {
                    "reference_index": reference_index,
                    "generated_index": None,
                    "reference": reference_item,
                    "generated": None,
                    "score": 0.0,
                }
            )
            continue

        # Find the most similar generated output
        generated_index = max(
            range(len(generated_items)),
            key=lambda index: similarity_matrix[reference_index][index],
        )
        matches.append(
            {
                "reference_index": reference_index,
                "generated_index": generated_index,
                "reference": reference_item,
                "generated": generated_items[generated_index],
                "score": similarity_matrix[reference_index][generated_index],
            }
        )

    # Record generated outputs that were not selected by any golden item
    matched_generated = {
        match["generated_index"]
        for match in matches
        if match["generated_index"] is not None
    }
    unmatched_generated = [
        item
        for index, item in enumerate(generated_items)
        if index not in matched_generated
    ]

    # Average golden-side best matches as soft recall
    recall = (
        sum(match["score"] for match in matches) / len(reference_items)
        if reference_items
        else 1.0
    )

    # Count every unused generated output as zero when calculating soft precision
    matched_generated_scores: dict[int, float] = {}
    for match in matches:
        generated_index = match["generated_index"]
        if generated_index is None:
            continue
        matched_generated_scores[generated_index] = max(
            matched_generated_scores.get(generated_index, 0.0),
            match["score"],
        )
    precision = (
        sum(matched_generated_scores.values()) / len(generated_items)
        if generated_items
        else 1.0
    )

    return {
        "score": f1_score(precision, recall),
        "precision": precision,
        "recall": recall,
        "matches": matches,
        "unmatched_generated": unmatched_generated,
    }


def _match_ordered_reference_items(
    reference_items: Sequence[Item],
    generated_items: Sequence[Item],
    similarity: Callable[[Item, Item], float],
) -> MatchReport:
    """Find the highest-scoring monotonic one-to-one item alignment."""

    reference_count = len(reference_items)
    generated_count = len(generated_items)
    similarity_matrix = [
        [
            similarity(reference_item, generated_item)
            for generated_item in generated_items
        ]
        for reference_item in reference_items
    ]

    scores = [
        [0.0 for _ in range(generated_count + 1)] for _ in range(reference_count + 1)
    ]
    choices: list[list[str | None]] = [
        [None for _ in range(generated_count + 1)] for _ in range(reference_count + 1)
    ]

    for reference_index in range(1, reference_count + 1):
        choices[reference_index][0] = "skip_reference"
    for generated_index in range(1, generated_count + 1):
        choices[0][generated_index] = "skip_generated"

    for reference_index in range(1, reference_count + 1):
        for generated_index in range(1, generated_count + 1):
            options = (
                (
                    scores[reference_index - 1][generated_index],
                    "skip_reference",
                    0,
                ),
                (
                    scores[reference_index][generated_index - 1],
                    "skip_generated",
                    1,
                ),
                (
                    scores[reference_index - 1][generated_index - 1]
                    + similarity_matrix[reference_index - 1][generated_index - 1],
                    "match",
                    2,
                ),
            )
            best_score, best_choice, _ = max(
                options,
                key=lambda option: (option[0], option[2]),
            )
            scores[reference_index][generated_index] = best_score
            choices[reference_index][generated_index] = best_choice

    matched_indices: dict[int, int] = {}
    reference_index = reference_count
    generated_index = generated_count
    while reference_index or generated_index:
        choice = choices[reference_index][generated_index]
        if choice == "match":
            matched_indices[reference_index - 1] = generated_index - 1
            reference_index -= 1
            generated_index -= 1
        elif choice == "skip_reference":
            reference_index -= 1
        elif choice == "skip_generated":
            generated_index -= 1
        else:
            break

    matches: list[ItemMatch] = []
    for reference_index, reference_item in enumerate(reference_items):
        generated_index = matched_indices.get(reference_index)
        if generated_index is None:
            matches.append(
                {
                    "reference_index": reference_index,
                    "generated_index": None,
                    "reference": reference_item,
                    "generated": None,
                    "score": 0.0,
                }
            )
            continue
        matches.append(
            {
                "reference_index": reference_index,
                "generated_index": generated_index,
                "reference": reference_item,
                "generated": generated_items[generated_index],
                "score": similarity_matrix[reference_index][generated_index],
            }
        )

    matched_generated = set(matched_indices.values())
    unmatched_generated = [
        item
        for index, item in enumerate(generated_items)
        if index not in matched_generated
    ]
    recall = (
        sum(match["score"] for match in matches) / reference_count
        if reference_count
        else 1.0
    )
    precision = (
        sum(match["score"] for match in matches) / generated_count
        if generated_count
        else 1.0
    )

    return {
        "score": f1_score(precision, recall),
        "precision": precision,
        "recall": recall,
        "matches": matches,
        "unmatched_generated": unmatched_generated,
    }


def f1_at_k(
    reference_items: Collection[Item],
    ranked_items: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
    preserve_order: bool = False,
) -> float:
    """Measure graded precision-recall balance for one field at top (k).

    Every golden item selects its strongest match from the first (k) generated
    items. Generated items selected by no golden item receive no credit. Soft
    precision uses the selected generated scores over (k), while soft recall
    averages the golden-item match scores. Their harmonic mean is returned.

    To use exact match, use None as embedder argument.
    Items after rank (k) are ignored; missing top-(k) positions and missing
    generated positions contribute zero.
    """

    _validate_metric_inputs(reference_items, ranked_items, k)
    top_ranked_items = ranked_items[:k]
    match_report = _match_reference_items(
        list(reference_items),
        top_ranked_items,
        _resolve_similarity(similarity, embedder),
        preserve_order=preserve_order,
    )
    precision = match_report["precision"] * len(top_ranked_items) / k
    recall = match_report["recall"] if reference_items else 0.0
    return f1_score(precision, recall)


def hit_at_k(
    reference: Collection[Item],
    ranking: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
) -> float:
    """Measure whether one query has a strong match in the top (k).

    Every top-(k) output is compared with every reference item. The returned
    value is the largest raw pair score:
    """

    _validate_metric_inputs(reference, ranking, k)
    compare = _resolve_similarity(similarity, embedder)
    return max(
        (
            compare(reference_item, generated_item)
            for generated_item in ranking[:k]
            for reference_item in reference
        ),
        default=0.0,
    )


def accuracy_at_k(
    reference_items: Collection[Item],
    ranked_items: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
) -> float:
    """Measure the strongest top-(k) match for one generated field.

    Every golden item is compared with every generated item in the first (k)
    positions, and the highest raw score is returned. With exact matching, the
    result is one when the field contains an exact top-(k) match and zero when
    it does not. With semantic matching, it is the strength of the best match.

    This function does not average across alarm queries. The future experiment
    evaluator is responsible for aggregating per-answer results.
    """

    return hit_at_k(reference_items, ranked_items, k, similarity, embedder)


def reciprocal_rank_at_k(
    reference: Collection[Item],
    ranking: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
) -> float:
    """Measure how early the strongest useful match occurs for one query.

    A match score is discounted by its one-based rank, and the highest
    discounted value is retained:

    With exact matching, this becomes the reciprocal rank of the first exact
    match. With semantic matching, an earlier moderate match can be compared
    with a later stronger match without discarding either through a threshold.
    """

    _validate_metric_inputs(reference, ranking, k)
    compare = _resolve_similarity(similarity, embedder)
    return max(
        (
            compare(reference_item, generated_item) / rank
            for rank, generated_item in enumerate(ranking[:k], start=1)
            for reference_item in reference
        ),
        default=0.0,
    )


def mrr_at_k(
    reference_items: Collection[Item],
    ranked_items: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
) -> float:
    """Measure rank-discounted match quality for one generated field.

    Each raw match score is divided by its one-based generated-item rank. The
    strongest discounted score within the first (k) positions is returned.
    Exact matching gives the reciprocal rank of the first exact match; semantic
    matching lets an earlier partial match compete with a later stronger match.

    Although the public name remains MRR@(k), this function evaluates one
    answer. Averaging these per-answer values belongs to the experiment runner.
    """

    return reciprocal_rank_at_k(
        reference_items,
        ranked_items,
        k,
        similarity,
        embedder,
    )


def average_precision_at_k(
    reference: Collection[Item],
    ranking: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
    preserve_order: bool = False,
) -> float:
    """Measure graded ranking quality for one query with multiple gold items.

    Every golden item selects its strongest top-(k) generated match. A generated
    item may represent several golden items, but its relevance at rank (i) is
    counted once using the strongest selected score.

    This rewards multiple strong matches and rewards placing them early. With
    exact (0/1) similarity, it reduces to ordinary Average Precision@(k).
    """

    _validate_metric_inputs(reference, ranking, k)
    references = list(reference)
    if not references:
        return 0.0

    match_report = _match_reference_items(
        references,
        ranking[:k],
        _resolve_similarity(similarity, embedder),
        preserve_order=preserve_order,
    )
    relevance_by_index: dict[int, float] = {}
    for match in match_report["matches"]:
        generated_index = match["generated_index"]
        if generated_index is None:
            continue
        relevance_by_index[generated_index] = max(
            relevance_by_index.get(generated_index, 0.0),
            match["score"],
        )
    cumulative_relevance = 0.0
    precision_sum = 0.0
    for generated_index in range(min(k, len(ranking))):
        relevance = relevance_by_index.get(generated_index, 0.0)
        cumulative_relevance += relevance
        precision_at_rank = cumulative_relevance / (generated_index + 1)
        precision_sum += relevance * precision_at_rank
    return precision_sum / len(references)


def map_at_k(
    reference_items: Collection[Item],
    ranked_items: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
    preserve_order: bool = False,
) -> float:
    """Measure multi-item ranking quality for one generated field.

    Every golden item uses its strongest generated match. At every rank, graded
    precision is calculated from cumulative raw similarity, then weighted by
    the current item's similarity. Generated items selected by no golden item
    receive zero relevance.

    Although the public name remains MAP@(k), this function evaluates one
    answer and therefore returns its Average Precision@(k). Averaging across
    answers belongs to the future experiment runner.
    """

    return average_precision_at_k(
        reference_items,
        ranked_items,
        k,
        similarity,
        embedder,
        preserve_order,
    )


def ranking_scores(
    reference_items: Collection[Item],
    ranked_items: Sequence[Item],
    k: int,
    similarity: Callable[[Item, Item], float] | None = None,
    embedder: SentenceTransformer | None = None,
    preserve_order: bool = False,
) -> dict[str, float]:
    """Calculate all top-(k) metrics for one field of one answer.

    The same golden items, generated ranking, cutoff, and similarity policy are
    passed to F1@(k), Accuracy@(k), MAP@(k), and MRR@(k). Semantic similarity is
    used by default; pass ``exact_match_score`` for citations. The returned
    dictionary contains no aggregation across alarm codes or test queries.
    """

    return {
        "f1": f1_at_k(
            reference_items,
            ranked_items,
            k,
            similarity,
            embedder,
            preserve_order=preserve_order,
        ),
        "accuracy": accuracy_at_k(
            reference_items,
            ranked_items,
            k,
            similarity,
            embedder,
        ),
        "map": map_at_k(
            reference_items,
            ranked_items,
            k,
            similarity,
            embedder,
            preserve_order=preserve_order,
        ),
        "mrr": mrr_at_k(
            reference_items,
            ranked_items,
            k,
            similarity,
            embedder,
        ),
    }
