import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote


def repository_files(root: Path):
    """Inspect source candidates, not ignored model/data/runtime caches after local training."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / name for name in result.stdout.split("\0") if name]


class RepositoryHygieneTest(unittest.TestCase):
    def test_portable_source_no_secrets_and_bounded_evidence(self):
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
            ".patch",
            ".jsonl",
            ".csv",
            ".txt",
            ".log",
        }
        private_path = re.compile(r"/home/[A-Za-z0-9_.-]+/")
        secret_patterns = [
            re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
        ]
        ignored_parts = {".git", ".third_party", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        violations = []
        for path in repository_files(root):
            if not path.is_file() or ignored_parts.intersection(path.parts):
                continue
            evidence = path.is_relative_to(root / "docs/experiments/evidence")
            limit = 10 * 1024 * 1024 if evidence else 1_000_000
            if path.stat().st_size > limit:
                violations.append(f"large file: {path.relative_to(root)}")
            if path.suffix.lower() not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            # Historical logs retain their original workstation paths; source remains portable.
            for pattern in secret_patterns + ([] if evidence else [private_path]):
                if pattern.search(content):
                    violations.append(f"{pattern.pattern}: {path.relative_to(root)}")
        self.assertEqual(violations, [])

    def test_all_markdown_documents_have_chinese_mirrors(self):
        root = Path(__file__).resolve().parents[1]
        pairs = [(root / "README.md", root / "README.zh-CN.md")]
        pairs.extend((path, root / "docs" / "zh-CN" / path.name) for path in (root / "docs").glob("*.md"))
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
            str(chinese.relative_to(root)) for english, chinese in pairs if english.is_file() and not chinese.is_file()
        ]
        self.assertEqual(missing, [])

    def test_local_markdown_links_resolve(self):
        root = Path(__file__).resolve().parents[1]
        missing = []
        for markdown in repository_files(root):
            if markdown.suffix != ".md" or not markdown.is_file():
                continue
            if {".git", ".third_party", ".venv"}.intersection(markdown.parts):
                continue
            if markdown.is_relative_to(root / "docs/experiments/evidence"):
                continue  # Source-faithful historical documents may refer to external assets.
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
