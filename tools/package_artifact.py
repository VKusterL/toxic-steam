"""Export the public code/aggregate-results artifact without data or Git history.

Build from a Git working tree: python tools/package_artifact.py
Verify anywhere (Git and third-party packages unnecessary):
    python tools/package_artifact.py --verify dist/toxic-steam-artifact.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile


ROOT_FILES = {"README.md", "LICENSE", "requirements.txt", ".gitignore"}
CHARACTERIZATION_RESULTS = {
    "README.md", "corpus_verification.json", "tag_toxicity_all.csv",
    "tag_toxicity_top.csv", "tag_toxicity_report.json", "tag_tfidf_means.csv",
    "tag_tfidf_report.json", "user_profile_report.json",
}
CLASSICAL_RESULTS = {
    "balanced_eval.csv", "bucket_metrics.csv", "lto_control.csv",
    "master_table.csv", "metrics_agg.csv", "metrics_cv.csv",
    "pairwise_bootstrap.csv",
}
# These reviewed additions can be exported before they are committed. Other new
# files are never picked up implicitly; review and commit them first.
EXTRA_FILES = {
    "docs/artifact.md", "tools/package_artifact.py",
    "tools/package_paper.py", "tools/check_paper_layout.py",
    "characterization/verify_numbers.py",
    "characterization/figures/check_figures.py",
    "characterization/figures/corpus_vectorizer.py",
} | {f"results/characterization/{name}" for name in CHARACTERIZATION_RESULTS}
BLOCKED_FIELDS = {
    "user_key", "user_id", "steamid", "steamid64", "steam_id", "profile_url",
    "review_id", "review_text", "review", "reviews", "narrative", "narratives",
    "explanation", "rationale", "reasoning", "prompt", "response",
}
# 'reviews' is an aggregate claim in this one reviewed report, not review text.
AGGREGATE_FIELD_EXCEPTIONS = {"results/characterization/corpus_verification.json": {"reviews"}}
SENSITIVE_PATTERNS = {
    "SteamID64": re.compile(r"(?<![\d.])7656119\d{10}(?!\d)"),
    "Steam profile URL": re.compile(r"steamcommunity\.com/(?:profiles/\d|id/[A-Za-z0-9_-])", re.I),
    "API credential": re.compile(r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,})\b"),
}
MANIFEST = "ARTIFACT_MANIFEST.json"
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_ARCHIVE_SIZE = 100 * 1024 * 1024


def safe_name(name: str) -> None:
    parts = PurePosixPath(name).parts
    if (not parts or name != PurePosixPath(name).as_posix()
            or name.startswith("/") or "\\" in name or ":" in name
            or any(part in {".", "..", ".git", "__pycache__"} for part in parts)):
        raise ValueError(f"Unsafe archive path: {name!r}")


def allowed(name: str) -> bool:
    safe_name(name)
    p = PurePosixPath(name)
    if name in ROOT_FILES or name in {"tools/package_artifact.py", "tools/package_paper.py", "tools/check_paper_layout.py"}:
        return True
    if p.parts[0] in {"src", "characterization", "docs"}:
        return p.suffix in {".py", ".md"} or p.name in {"requirements.txt", ".gitignore"}
    if len(p.parts) != 3 or p.parts[0] != "results":
        return False
    group, filename = p.parts[1:]
    if group == "characterization":
        return filename in CHARACTERIZATION_RESULTS
    if group == "replication":
        return filename in CLASSICAL_RESULTS
    if group == "replication_mpnet":
        return filename in {"metrics_agg.csv", "metrics_cv.csv"}
    if group in {"replication_bert", "replication_bert_mpnet", "replication_bert_undersampled"}:
        return filename == "bert_user_metrics.csv" or bool(re.fullmatch(r"fold_\d+(?:_loss)?\.json", filename))
    return group == "lente4_llm" and filename == "llm_user_metrics.csv"


def validate_payload(name: str, payload: bytes) -> None:
    if len(payload) > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds the artifact size limit: {name}")
    text = payload.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError(f"Binary content is not allowed: {name}")
    for label, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(text):
            # Never echo the matching identifier or credential.
            raise ValueError(f"Potential {label} detected in {name}")
    if not name.startswith("results/") or PurePosixPath(name).suffix not in {".csv", ".json"}:
        return
    blocked = BLOCKED_FIELDS - AGGREGATE_FIELD_EXCEPTIONS.get(name, set())

    def check_fields(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in blocked:
                    raise ValueError(f"Row-level or free-text field {key!r} is not allowed in {name}")
                check_fields(item)
        elif isinstance(value, list):
            for item in value:
                check_fields(item)

    if name.endswith(".json"):
        check_fields(json.loads(text))
    else:
        reader = csv.reader(io.StringIO(text))
        header = next(reader, [])
        if not header:
            raise ValueError(f"Empty result table: {name}")
        check_fields(dict.fromkeys(header))
        if any(len(row) != len(header) for row in reader):
            raise ValueError(f"Malformed result table: {name}")


def prepare_payload(name: str, payload: bytes) -> tuple[bytes, list[str]]:
    """Remove machine-specific figure locations only in exported JSON copies."""
    changes = []
    if name.startswith("results/characterization/") and name.endswith(".json"):
        report = json.loads(payload)

        def figure_location(value: str) -> str:
            return "data/figures-output/" + value.replace("\\", "/").rsplit("/", 1)[-1]

        if "figure_path" in report:
            report["figure_path"] = figure_location(report["figure_path"])
            changes.append("figure_path converted to a repository-relative output path")
        if "figures" in report:
            report["figures"] = {key: figure_location(value) for key, value in report["figures"].items()}
            changes.append("figures paths converted to repository-relative output paths")
        if changes:
            payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    validate_payload(name, payload)
    return payload, changes


def exclusion_reason(name: str) -> str:
    if name.startswith("notebooks/"):
        return "Notebook with embedded outputs; supplementary analysis is outside this export"
    if name.endswith(".jsonl") or "/panel/" in name:
        return "User-level narratives/judgments and free text are not redistributed"
    if "preds" in PurePosixPath(name).name:
        return "User-level predictions contain identifiable account keys"
    return "Outside the reviewed public-artifact allowlist"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("Missing manifest or duplicate archive members")
        if sum(entry.file_size for entry in entries) > MAX_ARCHIVE_SIZE:
            raise ValueError("Archive exceeds the uncompressed size limit")
        for entry in entries:
            safe_name(entry.filename)
            if entry.file_size > MAX_FILE_SIZE or entry.is_dir() or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsupported archive member: {entry.filename}")
        manifest = json.loads(bundle.read(MANIFEST))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported artifact manifest version")
        expected = {item["path"]: item for item in manifest["files"]}
        if len(expected) != len(manifest["files"]) or set(names) != set(expected) | {MANIFEST}:
            raise ValueError("Archive members do not match the manifest")
        for name, item in expected.items():
            if not allowed(name):
                raise ValueError(f"Member outside the allowlist: {name}")
            payload = bundle.read(name)
            validate_payload(name, payload)
            if len(payload) != item["bytes"] or digest(payload) != item["sha256"]:
                raise ValueError(f"Integrity check failed: {name}")
        for item in manifest["excluded_tracked_files"]:
            safe_name(item["path"])
        validate_payload(MANIFEST, bundle.read(MANIFEST))
    return manifest


def build(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    # Only ZIP files directly under dist are accepted: no arbitrary overwrites,
    # no staging directory traversal, and no output path eligible as an input.
    destination = (root / "dist").resolve()
    if not destination.is_relative_to(root) or output.parent.resolve() != destination or output.suffix.lower() != ".zip":
        raise ValueError("Output must be a .zip file directly inside this repository's dist directory")
    if output.is_symlink() or not output.resolve().is_relative_to(destination):
        raise ValueError("Output must not be a symlink or escape dist")
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True)
    tracked = set(result.stdout.decode("utf-8").rstrip("\0").split("\0")) - {""}
    # Honor working-tree deletions even before their removal is committed.
    deleted = subprocess.run(["git", "-C", str(root), "ls-files", "--deleted", "-z"],
                             check=True, capture_output=True)
    tracked.difference_update(deleted.stdout.decode("utf-8").rstrip("\0").split("\0"))
    candidates = tracked | {name for name in EXTRA_FILES if (root / name).is_file()}
    files = []
    payloads = {}
    excluded = []
    for name in sorted(candidates):
        if not allowed(name):
            if name in tracked:
                excluded.append({"path": name, "reason": exclusion_reason(name)})
            continue
        source = root / name
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Input is a symlink or escapes the repository: {name}")
        if source.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"File exceeds the artifact size limit: {name}")
        original = source.read_bytes()
        payload, changes = prepare_payload(name, original)
        payloads[name] = payload
        files.append({"path": name, "bytes": len(payload), "sha256": digest(payload),
                      "source_sha256": digest(original), "transformations": changes})
    manifest = {
        "schema_version": 1,
        "artifact": "Toxic Steam: public code and aggregate results",
        "source": "Current working-tree file contents; no Git history included",
        "scope": "Inspection of code and supplied aggregate results; raw-data reproduction requires separately held corpus",
        "not_included": ["Raw corpus and account identifiers", "User-level predictions and narrative judgments",
                         "Model weights, caches and binaries", "Notebooks with embedded outputs", "Manuscript delivery and Git history"],
        "files": files,
        "excluded_tracked_files": excluded,
    }
    payloads[MANIFEST] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, payload in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, payload)
        verify(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, metavar="ARCHIVE", help="Check an existing ZIP without reading the corpus or using Git")
    parser.add_argument("--output", type=Path, help="ZIP destination directly under the repository's dist directory")
    args = parser.parse_args()
    try:
        if args.verify:
            if args.output:
                parser.error("--verify and --output cannot be combined")
            archive = args.verify.resolve(strict=True)
            manifest = verify(archive)
        else:
            root = Path(__file__).resolve().parent.parent
            archive = args.output.absolute() if args.output else root / "dist" / "toxic-steam-artifact.zip"
            manifest = build(root, archive)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Artifact check failed: {exc}\n")
    print(f"Verified {archive}: {len(manifest['files'])} files; {len(manifest['excluded_tracked_files'])} tracked files excluded.")
    print(f"Archive SHA-256: {digest(archive.read_bytes())}")


if __name__ == "__main__":
    main()
