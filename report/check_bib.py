"""Check that every citation in the report has an entry in references.bib.

A missing entry is a LaTeX warning, not an error: the build still succeeds
but the reference silently becomes an asterisk.  This script turns that
into a failure, so that a citation can never be added and forgotten.

It also reports the entries that are defined but never cited, which is the
usual symptom of a renamed key.

Usage:
    python check_bib.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

CITE = re.compile(r"\\cite\{([^}]+)\}")
ENTRY = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)

HERE = Path(__file__).resolve().parent


def collect_citations(paths: list[Path]) -> set[str]:
    keys = set()
    for path in paths:
        if not path.is_file():
            continue
        for match in CITE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            for key in match.group(1).split(","):
                key = key.strip()
                if key:
                    keys.add(key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, default=HERE)
    parser.add_argument("--bib", type=Path, default=HERE / "references.bib")
    args = parser.parse_args()

    sources = [args.report_dir / "final_report.tex",
               args.report_dir / "progress_report.tex"]
    sources += sorted((args.report_dir / "sections").glob("*.tex"))

    cited = collect_citations(sources)
    text = args.bib.read_text(encoding="utf-8")
    defined = set(ENTRY.findall(text))

    missing = sorted(cited - defined)
    unused = sorted(defined - cited)

    print(f"{len(cited)} keys cited, {len(defined)} entries defined")
    if missing:
        print("\nCITED BUT NOT DEFINED:")
        for key in missing:
            print(f"  - {key}")
    if unused:
        print("\ndefined but never cited:")
        for key in unused:
            print(f"  - {key}")
    if missing:
        print("\nFAILED: a citation has no bibliography entry")
        return 1
    print("all citations resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
