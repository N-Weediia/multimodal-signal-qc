"""Validate the compact paired-seven release without recomputing analyses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageStat

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import RELEASE_ROOT
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import RELEASE_ROOT


MANIFEST_REL = Path("provenance/release_file_manifest.csv")
REPORT_REL = Path("provenance/release_validation.md")


class ImageSourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        source = dict(attrs).get("src")
        if source:
            self.sources.append(source)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    manifest = root / MANIFEST_REL
    return sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)


def write_manifest(root: Path) -> None:
    manifest = root / MANIFEST_REL
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        for path in release_files(root):
            writer.writerow(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )


def check_manifest(root: Path) -> tuple[list[str], int]:
    manifest = root / MANIFEST_REL
    errors: list[str] = []
    if not manifest.is_file():
        return [f"missing manifest: {MANIFEST_REL.as_posix()}"], 0
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"] for row in rows}
    actual = {path.relative_to(root).as_posix() for path in release_files(root)}
    for rel in sorted(expected - actual):
        errors.append(f"manifest entry missing on disk: {rel}")
    for rel in sorted(actual - expected):
        errors.append(f"file absent from manifest: {rel}")
    for row in rows:
        path = root / Path(row["relative_path"])
        if not path.is_file():
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['relative_path']}")
        if sha256(path) != row["sha256"]:
            errors.append(f"hash mismatch: {row['relative_path']}")
    return errors, len(rows)


def check_html_images(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    count = 0
    for html_path in root.glob("*.html"):
        parser = ImageSourceParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        for source in parser.sources:
            count += 1
            if "://" in source or source.startswith("data:"):
                errors.append(f"non-local HTML image source: {source}")
                continue
            if not (html_path.parent / source).resolve().is_file():
                errors.append(f"broken HTML image source: {source}")
    return errors, count


def check_pngs(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    paths = sorted((root / "figures").glob("*.png"))
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if width < 800 or height < 400:
                    errors.append(f"undersized PNG ({width}x{height}): {path.name}")
                extrema = ImageStat.Stat(image.convert("RGB")).extrema
                if not any(low < high for low, high in extrema):
                    errors.append(f"blank PNG: {path.name}")
        except Exception as exc:  # pragma: no cover - malformed-image path
            errors.append(f"unreadable PNG {path.name}: {exc}")
    return errors, len(paths)


def check_svgs(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    paths = sorted((root / "figures").glob("*.svg"))
    for path in paths:
        try:
            tree = ElementTree.parse(path)
            text_count = sum(1 for node in tree.iter() if node.tag.rsplit("}", 1)[-1] == "text")
            if text_count == 0:
                errors.append(f"SVG has no editable text nodes: {path.name}")
        except ElementTree.ParseError as exc:
            errors.append(f"invalid SVG {path.name}: {exc}")
    return errors, len(paths)


def check_pdf_fonts(root: Path) -> tuple[list[str], int, bool]:
    errors: list[str] = []
    paths = sorted((root / "figures").glob("*.pdf"))
    executable = shutil.which("pdffonts")
    if not executable:
        return errors, len(paths), False
    for path in paths:
        result = subprocess.run(
            [executable, str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode:
            errors.append(f"pdffonts failed for {path.name}: {result.stderr.strip()}")
        elif "Type 3" in result.stdout:
            errors.append(f"PDF contains Type 3 fonts: {path.name}")
        elif "yes" not in result.stdout:
            errors.append(f"PDF has no detected embedded font: {path.name}")
    return errors, len(paths), True


def render_report(
    manifest_count: int,
    html_count: int,
    png_count: int,
    svg_count: int,
    pdf_count: int,
    pdffonts_available: bool,
    errors: list[str],
) -> str:
    status = "PASS" if not errors else "FAIL"
    lines = [
        "# Paired-seven release validation",
        "",
        f"Overall status: **{status}**",
        "",
        f"- Manifest entries: {manifest_count}",
        f"- Local HTML image links checked: {html_count}",
        f"- PNG files checked for readability, dimensions, and nonblank pixels: {png_count}",
        f"- SVG files checked for editable text nodes: {svg_count}",
        f"- PDF files checked for embedded, non-Type-3 fonts: {pdf_count}",
        f"- `pdffonts` available: {'yes' if pdffonts_available else 'no'}",
        "",
    ]
    if errors:
        lines.extend(["## Errors", "", *[f"- {error}" for error in errors], ""])
    else:
        lines.extend(
            [
                "No missing files, hash mismatches, broken image links, blank/undersized PNGs,",
                "non-editable SVG exports, or Type 3 PDF fonts were detected.",
                "",
            ]
        )
    return "\n".join(lines)


def validate(root: Path, refresh_manifest: bool) -> list[str]:
    errors: list[str] = []
    html_errors, html_count = check_html_images(root)
    png_errors, png_count = check_pngs(root)
    svg_errors, svg_count = check_svgs(root)
    pdf_errors, pdf_count, pdffonts_available = check_pdf_fonts(root)
    errors.extend(html_errors + png_errors + svg_errors + pdf_errors)

    if refresh_manifest:
        provisional_count = len(release_files(root)) + (0 if (root / REPORT_REL).is_file() else 1)
        report = render_report(
            provisional_count,
            html_count,
            png_count,
            svg_count,
            pdf_count,
            pdffonts_available,
            errors,
        )
        report_path = root / REPORT_REL
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        write_manifest(root)

    manifest_errors, manifest_count = check_manifest(root)
    errors.extend(manifest_errors)
    if refresh_manifest and not errors:
        print(f"PASS: {manifest_count} manifest entries; release QA report and manifest refreshed")
    elif not errors:
        print(f"PASS: {manifest_count} manifest entries; release structure and figures validated")
    else:
        for error in errors:
            print(f"ERROR: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Release directory not found: {root}")
    raise SystemExit(1 if validate(root, args.refresh_manifest) else 0)


if __name__ == "__main__":
    main()
