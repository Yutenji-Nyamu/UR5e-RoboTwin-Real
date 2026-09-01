import re
import unittest
from pathlib import Path


class RepositoryHygieneTest(unittest.TestCase):
    def test_no_private_paths_secrets_or_large_files(self):
        root = Path(__file__).resolve().parents[1]
        text_suffixes = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".sh", ".lock", ".xml", ".script"}
        forbidden = [
            re.compile(r"/home/[A-Za-z0-9_.-]+/"),
            re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
        ]
        ignored_parts = {".git", ".third_party", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        violations = []
        for path in root.rglob("*"):
            if not path.is_file() or ignored_parts.intersection(path.parts):
                continue
            if path.stat().st_size > 1_000_000:
                violations.append(f"large file: {path.relative_to(root)}")
            if path.suffix.lower() not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            for pattern in forbidden:
                if pattern.search(content):
                    violations.append(f"{pattern.pattern}: {path.relative_to(root)}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
