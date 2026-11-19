"""Near-duplicate and low-information image detection and removal."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset, ImageRecord
from ..errors import BlockError
from ..registry import block
from ..utils import strip_split_prefix

_KNOWN_SPLIT_SET = frozenset(KNOWN_SPLITS)
_SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
_MAX_HAMMING = 7
_CHUNKS = 8


@dataclass
class _Entry:
    index: int
    record: ImageRecord
    fingerprint: int
    contrast: float
    scope: str


@block(
    "find_redundant_images",
    "Flag near-duplicate and blank images that add little learning value.",
    aliases=("find_duplicates", "find_uninformative"),
)
def find_redundant_images(
    dataset: Dataset,
    ctx: Context,
    *,
    flag: str = "redundant",
    hash_size: int = 8,
    similarity_threshold: int = 6,
    min_contrast: float = 5.0,
    protect_splits: list[str] | None = None,
    respect_splits: bool = False,
    write_report: bool = True,
    write_list: bool = True,
    max_logged: int = 10,
) -> Dataset:
    """Compare perceptual hashes and image contrast to find candidates.

    Images flagged here are stored in ``dataset.metadata["image_flags"][flag]``.
    A following :func:`remove_flagged_images` block can drop them from the
    dataset. Detection requires Pillow (``pip install 'cnn-trainer[images]'``).
    """
    Image = _import_pillow()
    _validate_hash_options(hash_size, similarity_threshold, min_contrast, max_logged)
    protected = _normalize_splits(protect_splits, "protect_splits")
    if not isinstance(flag, str) or not flag.strip():
        raise BlockError("flag must be a non-empty string")
    flag = flag.strip()
    if not dataset.images:
        raise BlockError("Cannot analyze an empty dataset")

    entries: list[_Entry] = []
    unreadable: list[dict[str, str]] = []
    records = sorted(dataset.images, key=lambda record: record.relative_path.as_posix())
    for record in records:
        try:
            fingerprint, contrast = _fingerprint(Image, record.path, hash_size=hash_size)
        except OSError as error:
            unreadable.append(
                {"relative_path": record.relative_path.as_posix(), "error": str(error)}
            )
            continue
        entries.append(
            _Entry(
                index=len(entries),
                record=record,
                fingerprint=fingerprint,
                contrast=contrast,
                scope=(record.split or "unassigned") if respect_splits else "",
            )
        )

    instance_counts = Counter(
        annotation.class_id for record in dataset.images for annotation in record.annotations
    )
    groups = _duplicate_groups(entries, similarity_threshold, hash_size)

    candidates: dict[int, dict] = {}
    group_reports: list[dict] = []
    for group_number, members in enumerate(groups, start=1):
        keeper = max(
            members,
            key=lambda entry: _keeper_score(entry, instance_counts, protected),
        )
        group_name = f"duplicate_{group_number:03d}"
        splits_in_group = {entry.record.split for entry in members if entry.record.split}
        group_reports.append(
            {
                "id": group_name,
                "keeper": keeper.record.relative_path.as_posix(),
                "members": [
                    entry.record.relative_path.as_posix()
                    for entry in sorted(
                        members, key=lambda item: item.record.relative_path.as_posix()
                    )
                ],
                "size": len(members),
                "cross_split": len(splits_in_group) > 1,
            }
        )
        for entry in members:
            if entry is keeper or entry.record.split in protected:
                continue
            candidates[entry.index] = {
                "relative_path": entry.record.relative_path.as_posix(),
                "split": entry.record.split,
                "reason": "duplicate",
                "similarity": (entry.fingerprint ^ keeper.fingerprint).bit_count(),
                "group": group_name,
                "kept": keeper.record.relative_path.as_posix(),
            }

    for entry in entries:
        if entry.contrast >= min_contrast:
            continue
        if entry.record.annotations:
            continue
        if entry.record.split in protected:
            continue
        candidates[entry.index] = {
            "relative_path": entry.record.relative_path.as_posix(),
            "split": entry.record.split,
            "reason": "blank",
            "contrast": round(entry.contrast, 2),
        }

    candidate_list = sorted(candidates.values(), key=lambda item: item["relative_path"])
    counts = Counter(item["reason"] for item in candidate_list)
    report = {
        "flag": flag,
        "images_analyzed": len(entries),
        "unreadable": unreadable,
        "duplicate_groups": group_reports,
        "cross_split_groups": sum(1 for group in group_reports if group["cross_split"]),
        "candidates": candidate_list,
        "counts": {
            "duplicate": counts.get("duplicate", 0),
            "blank": counts.get("blank", 0),
        },
        "parameters": {
            "hash_size": hash_size,
            "similarity_threshold": similarity_threshold,
            "min_contrast": min_contrast,
            "respect_splits": respect_splits,
            "protect_splits": sorted(protected),
        },
    }

    flags = dataset.metadata.get("image_flags")
    if not isinstance(flags, dict):
        flags = {}
        dataset.metadata["image_flags"] = flags
    flags[flag] = [candidate["relative_path"] for candidate in candidate_list]
    dataset.metadata["redundancy"] = report

    if write_report:
        destination = ctx.ensure_workdir() / "redundancy.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    if write_list:
        lines = [candidate["relative_path"] for candidate in candidate_list]
        (ctx.ensure_workdir() / "candidates.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    ctx.logger.info(
        "Found %d candidate(s) for %r: %d duplicate(s) in %d group(s), %d blank; "
        "%d image(s) unreadable",
        len(candidate_list),
        flag,
        counts.get("duplicate", 0),
        len(group_reports),
        counts.get("blank", 0),
        len(unreadable),
    )
    cross_split = sum(1 for group in group_reports if group["cross_split"])
    if cross_split:
        ctx.logger.warning(
            "%d near-duplicate group(s) span different splits; they can inflate "
            "validation metrics. Run this block before split or review the groups.",
            cross_split,
        )
    for candidate in candidate_list[:max_logged]:
        ctx.logger.info(
            "  %s: %s%s",
            candidate["reason"],
            candidate["relative_path"],
            f" (kept {candidate['kept']})" if candidate.get("kept") else "",
        )
    if write_report:
        ctx.logger.info("Wrote %s", ctx.workdir / "redundancy.json")
    if write_list:
        ctx.logger.info("Wrote %s", ctx.workdir / "candidates.txt")
    return dataset


@block(
    "remove_flagged_images",
    "Remove images flagged as redundant or uninformative.",
    aliases=("remove_redundant", "drop_candidates", "remove_images"),
)
def remove_flagged_images(
    dataset: Dataset,
    ctx: Context,
    *,
    flag: str = "redundant",
    paths: list[str] | None = None,
    reasons: list[str] | None = None,
    protect_splits: list[str] | None = None,
    max_fraction: float = 0.5,
    force: bool = False,
    dry_run: bool = False,
    write_report: bool = True,
    max_logged: int = 10,
) -> Dataset:
    """Drop flagged images from the in-memory dataset.

    Source files are not touched. Downstream ``export_yolo`` / ``train`` blocks
    simply no longer see the removed images or their labels.
    """
    if not isinstance(flag, str) or not flag.strip():
        raise BlockError("flag must be a non-empty string")
    flag = flag.strip()
    if (
        isinstance(max_fraction, bool)
        or not isinstance(max_fraction, (int, float))
        or not 0 <= max_fraction <= 1
    ):
        raise BlockError("max_fraction must be between 0 and 1")
    if isinstance(max_logged, bool) or not isinstance(max_logged, int) or max_logged < 0:
        raise BlockError("max_logged must be a non-negative integer")
    protected = _normalize_splits(protect_splits, "protect_splits")
    reason_filter = _normalize_reasons(reasons)
    if paths is not None and not isinstance(paths, list):
        raise BlockError("paths must be a list of relative image paths")
    if not dataset.images:
        raise BlockError("Cannot remove images from an empty dataset")

    flags = dataset.metadata.get("image_flags")
    if paths is None:
        if not isinstance(flags, dict) or not flags.get(flag):
            raise BlockError(
                f"No images are flagged as {flag!r}. Run find_redundant_images first "
                "or pass an explicit paths list."
            )
        wanted = [str(path) for path in flags[flag]]
    else:
        wanted = [str(path) for path in paths]

    if not wanted:
        ctx.logger.info("Nothing to remove: the path list is empty")
        return dataset

    reason_map = _reason_map(dataset)
    exact, stripped = _record_index(dataset)
    selected: list[ImageRecord] = []
    selected_reasons: Counter[str] = Counter()
    seen: set[int] = set()
    missing: list[str] = []
    skipped_reason = 0
    protected_count = 0
    for raw in wanted:
        matches = _match_records(raw, exact, stripped)
        if not matches:
            missing.append(raw)
            continue
        for record in matches:
            if id(record) in seen:
                continue
            reason = reason_map.get(record.relative_path.as_posix(), "explicit")
            if reason_filter is not None and reason not in reason_filter:
                skipped_reason += 1
                continue
            if record.split in protected:
                protected_count += 1
                continue
            seen.add(id(record))
            selected.append(record)
            selected_reasons[reason] += 1

    total = len(dataset.images)
    if len(selected) > max_fraction * total and not force:
        raise BlockError(
            f"Refusing to remove {len(selected)} of {total} image(s): that exceeds "
            f"max_fraction={max_fraction}. Review the candidates or set force: true."
        )
    if total - len(selected) < 1:
        raise BlockError("Refusing to remove every image from the dataset")

    report = {
        "flag": flag,
        "requested": len(wanted),
        "matched": len(selected),
        "removed": 0 if dry_run else len(selected),
        "protected": protected_count,
        "missing": sorted(missing),
        "skipped_by_reason": skipped_reason,
        "reason_counts": dict(selected_reasons),
        "dry_run": dry_run,
    }

    if dry_run:
        ctx.logger.info("Dry run: would remove %d image(s) flagged as %r", len(selected), flag)
    else:
        removal_ids = {id(record) for record in selected}
        dataset.images = [record for record in dataset.images if id(record) not in removal_ids]
        removed_paths = {record.relative_path.as_posix() for record in selected}
        if isinstance(flags, dict) and flag in flags:
            remaining = [path for path in flags[flag] if str(path) not in removed_paths]
            if remaining:
                flags[flag] = remaining
            else:
                flags.pop(flag, None)
        _prune_redundancy_metadata(dataset, removed_paths)
        dataset.metadata["removed"] = report
        ctx.logger.info(
            "Removed %d image(s) flagged as %r; %d image(s) remain",
            len(selected),
            flag,
            len(dataset.images),
        )

    for record in selected[:max_logged]:
        ctx.logger.info("  - %s", record.relative_path.as_posix())
    if missing:
        ctx.logger.warning(
            "%d requested path(s) were not found, e.g. %s",
            len(missing),
            ", ".join(sorted(missing)[:3]),
        )
    if write_report:
        destination = ctx.ensure_workdir() / "removed.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    return dataset


def _import_pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise BlockError(
            "Image analysis requires Pillow. Install it with: pip install 'cnn-trainer[images]'"
        ) from error
    return Image


def _fingerprint(Image, path: Path, *, hash_size: int) -> tuple[int, float]:
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        gray = image.convert("L")
        small = gray.resize((hash_size + 1, hash_size), resampling.LANCZOS)
        pixels = small.tobytes()
        stats_image = gray.resize((32, 32), resampling.BILINEAR)
        values = stats_image.tobytes()

    fingerprint = 0
    bit = 0
    stride = hash_size + 1
    for row in range(hash_size):
        offset = row * stride
        for column in range(hash_size):
            if pixels[offset + column + 1] > pixels[offset + column]:
                fingerprint |= 1 << bit
            bit += 1

    return fingerprint, statistics.pstdev(values)


def _duplicate_groups(entries: list[_Entry], threshold: int, hash_size: int) -> list[list[_Entry]]:
    parent = list(range(len(entries)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_hash: dict[tuple[str, int], list[_Entry]] = defaultdict(list)
    for entry in entries:
        by_hash[(entry.scope, entry.fingerprint)].append(entry)

    representatives: list[_Entry] = []
    for group in by_hash.values():
        representatives.append(group[0])
        for entry in group[1:]:
            union(group[0].index, entry.index)

    total_bits = hash_size * hash_size
    base_bits = total_bits // _CHUNKS
    buckets: dict[tuple[str, int, int], list[_Entry]] = defaultdict(list)
    for entry in representatives:
        start = 0
        for position in range(_CHUNKS):
            width = base_bits
            if position == _CHUNKS - 1:
                width = total_bits - base_bits * (_CHUNKS - 1)
            mask = (1 << width) - 1
            value = (entry.fingerprint >> start) & mask
            buckets[(entry.scope, position, value)].append(entry)
            start += width

    for bucket in buckets.values():
        for left_index in range(len(bucket)):
            left = bucket[left_index]
            for right in bucket[left_index + 1 :]:
                if find(left.index) == find(right.index):
                    continue
                distance = (left.fingerprint ^ right.fingerprint).bit_count()
                if distance <= threshold:
                    union(left.index, right.index)

    groups: dict[int, list[_Entry]] = defaultdict(list)
    for entry in entries:
        groups[find(entry.index)].append(entry)
    return [members for members in groups.values() if len(members) > 1]


def _keeper_score(entry: _Entry, instance_counts: Counter, protected: set[str]) -> tuple:
    record = entry.record
    class_ids = {annotation.class_id for annotation in record.annotations}
    rare_score = sum(1.0 / instance_counts.get(class_id, 1) for class_id in class_ids)
    return (
        1 if record.split in protected else 0,
        len(record.annotations),
        rare_score,
        _SPLIT_RANK.get(record.split or "", 0),
        record.width * record.height,
    )


def _normalize_splits(value, name: str) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise BlockError(f"{name} must be a list of split names")
    splits = {str(item).strip() for item in value if str(item).strip()}
    unknown = splits - _KNOWN_SPLIT_SET
    if unknown:
        raise BlockError(
            f"Unknown split name(s) in {name}: {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(KNOWN_SPLITS)}"
        )
    return splits


def _normalize_reasons(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise BlockError("reasons must be a list of reason names")
    reasons = {str(item).strip() for item in value if str(item).strip()}
    allowed = {"duplicate", "blank", "explicit"}
    unknown = reasons - allowed
    if unknown:
        raise BlockError(
            f"Unknown reason(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return reasons


def _reason_map(dataset: Dataset) -> dict[str, str]:
    report = dataset.metadata.get("redundancy")
    if not isinstance(report, dict):
        return {}
    return {
        candidate["relative_path"]: candidate["reason"]
        for candidate in report.get("candidates", [])
        if isinstance(candidate, dict) and "relative_path" in candidate
    }


def _record_index(
    dataset: Dataset,
) -> tuple[dict[str, list[ImageRecord]], dict[str, list[ImageRecord]]]:
    exact: dict[str, list[ImageRecord]] = defaultdict(list)
    stripped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in dataset.images:
        exact[record.relative_path.as_posix()].append(record)
        exact[record.path.as_posix()].append(record)
        key = strip_split_prefix(record.relative_path, _KNOWN_SPLIT_SET).as_posix()
        stripped[key].append(record)
    return dict(exact), dict(stripped)


def _match_records(
    raw: str,
    exact: dict[str, list[ImageRecord]],
    stripped: dict[str, list[ImageRecord]],
) -> list[ImageRecord]:
    cleaned = str(raw).replace("\\", "/")
    if cleaned in exact:
        return exact[cleaned]
    key = Path(cleaned)
    if key.is_absolute():
        key = Path(key.name)
    return stripped.get(key.as_posix(), [])


def _prune_redundancy_metadata(dataset: Dataset, removed_paths: set[str]) -> None:
    report = dataset.metadata.get("redundancy")
    if not isinstance(report, dict):
        return
    candidates = [
        candidate
        for candidate in report.get("candidates", [])
        if candidate.get("relative_path") not in removed_paths
    ]
    report["candidates"] = candidates
    counts = Counter(candidate.get("reason", "explicit") for candidate in candidates)
    report["counts"] = {
        "duplicate": counts.get("duplicate", 0),
        "blank": counts.get("blank", 0),
    }

    groups = []
    for group in report.get("duplicate_groups", []):
        members = [member for member in group.get("members", []) if member not in removed_paths]
        if len(members) < 2:
            continue
        group["members"] = members
        group["size"] = len(members)
        groups.append(group)
    report["duplicate_groups"] = groups
    report["cross_split_groups"] = sum(1 for group in groups if group.get("cross_split"))


def _validate_hash_options(
    hash_size: int, similarity_threshold: int, min_contrast: float, max_logged: int
) -> None:
    if isinstance(hash_size, bool) or not isinstance(hash_size, int):
        raise BlockError("hash_size must be an integer between 8 and 16")
    if not 8 <= hash_size <= 16:
        raise BlockError("hash_size must be between 8 and 16")
    if (
        isinstance(similarity_threshold, bool)
        or not isinstance(similarity_threshold, int)
        or not 0 <= similarity_threshold <= _MAX_HAMMING
    ):
        raise BlockError(f"similarity_threshold must be an integer between 0 and {_MAX_HAMMING}")
    if (
        isinstance(min_contrast, bool)
        or not isinstance(min_contrast, (int, float))
        or min_contrast < 0
    ):
        raise BlockError("min_contrast must be a non-negative number")
    if isinstance(max_logged, bool) or not isinstance(max_logged, int) or max_logged < 0:
        raise BlockError("max_logged must be a non-negative integer")
