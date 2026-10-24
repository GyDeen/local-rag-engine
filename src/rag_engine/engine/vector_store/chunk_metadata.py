"""Text metadata extraction shared by document ingestion paths."""

import re

# Example catalogue-code format: <origin>.<module>.<sequence>, e.g. demo.svc.0001, demo.net.0004
ALARM_CODE_PATTERN = re.compile(
    r"\b[a-z]{2,6}(?P<separator>[.-])[a-z]{2,6}(?P=separator)\d{3,6}\b",
    re.IGNORECASE,
)

# Inline markdown links pointing at an internal anchor, e.g. [Example Section](#example-section)
INTERNAL_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(#([^\)]+)\)")
SEE_ALSO_LINE_PATTERN = re.compile(r"(?im)^\s*\*\*see\s+also:\*\*\s*(.+)$")

# A short bold "label" line that introduces a following code block/table,
# e.g. **Syntax:**, **Example:**, **Additional information:**
LABEL_LINE_PATTERN = re.compile(r"^\*\*[^*]+\*\*:?\s*$")

RELATED_COMMANDS_PATTERN = re.compile(r"(?im)^\s*related\s+commands?\s*:\s*(.+)$")

# A WARNING callout: "> [!WARNING]" followed by its blockquote body lines,
# e.g.
#   > [!WARNING]
#   > When applying a configuration update ...
WARNING_CALLOUT_LINE_PATTERN = re.compile(
    r"^[ \t]{0,3}>\s*\[!WARNING\]\s*$", re.IGNORECASE
)
BLOCKQUOTE_LINE_PATTERN = re.compile(r"^[ \t]{0,3}>\s?(.*)$")


def extract_alarm_codes(text: str) -> list[str]:
    """Extract unique alarm-code tokens in their first-seen order."""

    matches = (match.group(0) for match in ALARM_CODE_PATTERN.finditer(text))
    return list(dict.fromkeys(matches))


def extract_see_also(text: str) -> list[str]:
    """Extract internal links and labelled See Also command lists."""

    seen: set[str] = set()
    refs: list[str] = []

    def add_reference(reference: str) -> None:
        normalized = reference.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            refs.append(normalized)

    for raw_text, anchor in INTERNAL_LINK_PATTERN.findall(text):
        label = raw_text.strip("*_ ").strip()
        if not label or label.isdigit():
            continue

        add_reference(f"{label} (#{anchor})")

    for raw_line in SEE_ALSO_LINE_PATTERN.findall(text):
        for raw_reference in raw_line.split(","):
            reference = re.sub(r"`([^`]*)`", r"\1", raw_reference)
            add_reference(reference)

    return refs


def extract_related_commands(text: str) -> list[str]:
    """Extract labelled related-command references if the document uses them."""

    values: list[str] = []
    for raw_line in RELATED_COMMANDS_PATTERN.findall(text):
        for token in re.split(r"[,;]", raw_line):
            token = token.strip()
            if token:
                values.append(token)
    return values


def extract_warnings(text: str) -> list[str]:
    """Extract the body text of WARNING callouts, in first-seen order."""

    seen: set[str] = set()
    warnings: list[str] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        if not WARNING_CALLOUT_LINE_PATTERN.match(lines[i]):
            i += 1
            continue

        i += 1
        body_lines: list[str] = []
        while i < len(lines) and BLOCKQUOTE_LINE_PATTERN.match(lines[i]):
            match = BLOCKQUOTE_LINE_PATTERN.match(lines[i])
            body_lines.append(match.group(1).strip())
            i += 1

        body = " ".join(line for line in body_lines if line)
        key = body.casefold()
        if body and key not in seen:
            seen.add(key)
            warnings.append(body)

    return warnings
