class RagError(Exception):
    """Base class for failures that callers can handle deliberately."""


class ServiceBusyError(RagError):
    """The backend cannot admit another request within its capacity."""


class ServiceUnavailableError(RagError):
    """The backend cannot serve a request in its current dependency state."""


class RetrievalError(ServiceUnavailableError):
    """An operational failure prevented retrieval from completing."""

    public_message = "Retrieval is temporarily unavailable"


class RerankingError(RetrievalError):
    """The ranking model could not complete its operation."""

    public_message = "Evidence ranking is temporarily unavailable"


class GenerationError(ServiceUnavailableError):
    """The answer model could not complete its operation."""

    public_message = "Answer generation is temporarily unavailable"


class ModelResponseError(GenerationError, ValueError):
    """The model returned an invalid transport or answer payload."""

    public_message = "The model returned an invalid response, please retry"


class RerankResponseError(RerankingError, ValueError):
    """The reranking model exhausted its invalid-response correction attempts."""

    public_message = "The ranking model returned an invalid response; please retry"


class IndexingError(RagError):
    """Index construction failed; the destination may be incomplete."""


class SourceDocumentError(RetrievalError, ValueError):
    """Indexed source text can no longer be read consistently."""

    public_message = "Documentation is unavailable or needs reindexing"


class DatabaseError(ServiceUnavailableError):
    """The vector database could not complete a driver operation."""

    public_message = "The documentation database is temporarily unavailable"
