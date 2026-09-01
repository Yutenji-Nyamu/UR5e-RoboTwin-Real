import re
import unittest
from pathlib import Path
from urllib.parse import unquote


class RepositoryHygieneTest(unittest.TestCase):
    def test_no_private_paths_secrets_or_large_files(self):
        root = Path(__file__).resolve().parents[1]
        text_suffixes = {
            ".py",
            ".md",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".sh",
            ".lock",
            ".xml",
            ".script",
        }
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

    def test_all_markdown_documents_have_chinese_mirrors(self):
        root = Path(__file__).resolve().parents[1]
        pairs = [(root / "README.md", root / "README.zh-CN.md")]
        pairs.extend(
            (path, root / "docs" / "zh-CN" / path.name)
            for path in (root / "docs").glob("*.md")
        )
        pairs.extend(
            (path, root / "docs" / "zh-CN" / "runbooks" / path.name)
            for path in (root / "docs" / "runbooks").glob("*.md")
        )
        pairs.append(
            (
                root / "integrations" / "robotwin" / "README.md",
                root / "integrations" / "robotwin" / "README.zh-CN.md",
            )
        )
        missing = [
            str(chinese.relative_to(root))
            for english, chinese in pairs
            if english.is_file() and not chinese.is_file()
        ]
        self.assertEqual(missing, [])

    def test_local_markdown_links_resolve(self):
        root = Path(__file__).resolve().parents[1]
        missing = []
        for markdown in root.rglob("*.md"):
            if {".git", ".third_party", ".venv"}.intersection(markdown.parts):
                continue
            content = markdown.read_text(encoding="utf-8")
            for match in re.finditer(r"\[[^]]*\]\(([^)]+)\)", content):
                target = unquote(match.group(1).split("#", 1)[0])
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                if not (markdown.parent / target).exists():
                    missing.append(f"{markdown.relative_to(root)} -> {target}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
