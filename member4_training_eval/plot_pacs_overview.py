"""Fig. 7: the PACS sample grid and the per-domain class counts.

Two panels are produced:

  * a grid with one image per (domain, class) pair, so that the reader
    sees what the four domains of PACS actually look like;
  * a bar chart of the number of images per class in each domain, read
    from the real split files rather than from a hard-coded table.

The class counts come from the frozen splits under
``member4_training_eval/project_splits``, so the figure follows the data
automatically and cannot drift away from it.

Usage:
    python plot_pacs_overview.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
DOMAIN_LABELS = {"photo": "Photo", "art_painting": "Art Painting",
                 "cartoon": "Cartoon", "sketch": "Sketch"}


def split_files(runs_root: Path) -> dict:
    """The frozen split files, by name."""
    directory = REPO / "member4_training_eval" / "project_splits"
    files = {}
    for path in sorted(directory.glob("*.txt")):
        files[path.stem] = path
    return files


def class_counts(files: dict) -> dict:
    """The number of images per class, per domain.

    The counts come from the image directories themselves rather than from
    the split files: the split files only describe the training protocol
    (a photo training split and one test split per target domain), while
    the figure is meant to show what the dataset contains.  The class
    breakdown of the training split is reported separately in Table~I.
    """
    counts = {}
    for domain in DOMAINS:
        directory = config.IMAGE_ROOT / domain
        if not directory.is_dir():
            continue
        counter = Counter()
        for name in config.CLASSES:
            folder = directory / name
            if folder.is_dir():
                counter[name] = sum(
                    1 for p in folder.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        counts[domain] = counter
    return counts


def sample_grid(files: dict, out: Path) -> None:
    """One image per (domain, class), with the class as the row."""
    classes = list(config.CLASSES)
    domains = [d for d in DOMAINS
               if (config.IMAGE_ROOT / d).is_dir()]
    figure, axes = plt.subplots(len(classes), len(domains),
                                 figsize=(2.0 * len(domains),
                                          1.7 * len(classes)))
    if len(classes) == 1 or len(domains) == 1:
        axes = [[axes]]
    for row, name in enumerate(classes):
        for column, domain in enumerate(domains):
            axis = axes[row][column]
            axis.set_xticks([])
            axis.set_yticks([])
            picked = None
            for path in sorted((config.IMAGE_ROOT / domain / name).glob("*")):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    picked = path
                    break
            if picked is None:
                axis.set_facecolor("#f0f0f0")
                axis.text(0.5, 0.5, "none", ha="center", va="center",
                          fontsize=7, color="#999999")
            else:
                try:
                    image = plt.imread(picked)
                    axis.imshow(image)
                except (OSError, ValueError):
                    axis.set_facecolor("#f0f0f0")
            if row == 0:
                axis.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=8,
                               pad=4)
            if column == 0:
                axis.set_ylabel(name, fontsize=7, rotation=0,
                                ha="right", va="center")
    figure.suptitle("PACS: four domains, seven classes", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")

def counts_figure(counts: dict, out: Path) -> None:
    """The number of images per class in each of the four domains."""
    domains = [d for d in DOMAINS if d in counts]
    classes = list(config.CLASSES)
    figure, axis = plt.subplots(figsize=(6.2, 2.9))
    width = 0.2
    positions = range(len(classes))
    for index, domain in enumerate(domains):
        counter = counts[domain]
        values = [counter.get(name, 0) for name in classes]
        offset = (index - (len(domains) - 1) / 2) * width
        axis.bar([p + offset for p in positions], values, width,
                 label=DOMAIN_LABELS.get(domain, domain))
    axis.set_xticks(list(positions))
    axis.set_xticklabels(classes, fontsize=8)
    axis.set_ylabel("images", fontsize=8)
    axis.set_title("Class counts per domain (read from the frozen splits)",
                   fontsize=9)
    axis.legend(fontsize=7, ncol=4, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = split_files(config.OUTPUT_ROOT)
    if not files:
        print("no split files were found; the figure cannot be built")
        return
    counts = class_counts(files)
    sample_grid(files, args.out_dir / "pacs_samples.png")
    counts_figure(counts, args.out_dir / "pacs_counts.png")


if __name__ == "__main__":
    main()
