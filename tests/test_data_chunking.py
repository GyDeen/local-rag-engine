import tempfile
import unittest
from pathlib import Path

from rag_engine.engine.vector_store.data_chunking import (
    OVERSIZED_CHUNK_THRESHOLD,
    chunk_markdown_file,
)


class MarkdownChunkingTests(unittest.TestCase):
    def make_file(self, text: str, *, filename: str = "guide.md"):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        docs_root = Path(temp_dir.name)
        source = docs_root / filename
        source.write_text(text, encoding="utf-8")
        return docs_root, source

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int = 1000,
        overlap: int = 0,
        filename: str = "guide.md",
    ) -> tuple[list, str]:
        docs_root, source = self.make_file(text, filename=filename)
        chunks = chunk_markdown_file(
            source,
            docs_root,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return chunks, source.read_bytes().decode("utf-8")

    def test_hash_comments_inside_fenced_code_are_not_headings(self):
        chunks, _ = self.chunk_text("""# Guide

## Parameters

```text
# This is a code comment
safe_velocity: 1960
# This is another code comment
```

The comments belong to the Parameters section.
""")

        self.assertEqual([chunk.section for chunk in chunks], ["Guide > Parameters"])

    def test_hash_comments_inside_tilde_fences_are_not_headings(self):
        chunks, _ = self.chunk_text("""# Guide

## Parameters

~~~text
# This is a code comment
safe_velocity: 1960
~~~

The code belongs to Parameters.
""")

        self.assertEqual([chunk.section for chunk in chunks], ["Guide > Parameters"])

    def test_shorter_backtick_fence_does_not_close_a_longer_fence(self):
        chunks, _ = self.chunk_text(
            "# Guide\n\n## Parameters\n\n````text\nsafe\n```\nstill code\n````\n\nBody text.\n"
        )

        self.assertEqual([chunk.section for chunk in chunks], ["Guide > Parameters"])

    def test_backtick_fence_does_not_close_a_tilde_fence(self):
        chunks, _ = self.chunk_text(
            "# Guide\n\n## Parameters\n\n~~~text\nsafe\n```\nstill code\n~~~\n\nBody text.\n"
        )

        self.assertEqual([chunk.section for chunk in chunks], ["Guide > Parameters"])

    def test_oversized_protected_block_is_split_at_line_boundaries(self):
        code_lines = "\n".join(f"process_item({index}, 1960)" for index in range(140))
        text = (
            "# Guide\n\n**Syntax:**\n\n```text\n"
            f"{code_lines}\n```\n\nAfter the block.\n"
        )
        chunks, source = self.chunk_text(text, chunk_size=1000, overlap=100)

        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            size = chunk.end_offset - chunk.start_offset
            self.assertLessEqual(size, OVERSIZED_CHUNK_THRESHOLD)
            self.assertTrue(source[chunk.start_offset : chunk.end_offset].strip())
            if chunk.start_offset > 0:
                self.assertTrue(source[chunk.start_offset - 1].isspace())
            if chunk.end_offset < len(source):
                self.assertTrue(source[chunk.end_offset].isspace())

        all_text = "\n".join(
            source[chunk.start_offset : chunk.end_offset] for chunk in chunks
        )
        self.assertIn("process_item(0, 1960)", all_text)
        self.assertIn("process_item(139, 1960)", all_text)
        self.assertIn("After the block.", all_text)

    def test_single_line_larger_than_threshold_is_flagged_oversized(self):
        long_line = "x" * 3000
        chunks, text = self.chunk_text(
            f"# Guide\n\n```text\n{long_line}\n```\n\nEnd.\n"
        )

        oversized = [chunk for chunk in chunks if chunk.metadata.get("oversized")]
        self.assertEqual(len(oversized), 1)
        # The oversized piece is emitted with its fence and the surrounding
        # context, so the span is deliberately longer than the long line itself.
        span = text[oversized[0].start_offset : oversized[0].end_offset]
        self.assertIn(long_line, span)
        self.assertIn("```text", span)
        self.assertIn("End.", span)


if __name__ == "__main__":
    unittest.main()
