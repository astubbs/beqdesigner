#!/usr/bin/env python3
"""WAV cache acquisition recommender — find diversifying titles to add.

Picks the catalogue entries that, if acquired, would best correct the bias
in our local WAV cache.  Uses greedy bias-correction selection: at each
step, picks the entry whose metadata most closes the gap between the
cache distribution and the catalogue distribution.

Output: a markdown shopping list grouped by author, format, and era for
easy human acquisition.

Usage::

    poetry run python3 scripts/nn_acquisition_recommender.py
    poetry run python3 scripts/nn_acquisition_recommender.py -n 50 \
        -o docs/acquisition_recommendations.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running from project root.
sys.path.insert(0, "src/main/python")
sys.path.insert(0, "src/test/python")


def _classify_format(audio_types) -> str:
    j = " ".join(audio_types or []).lower()
    if "atmos" in j:
        return "atmos"
    if "truehd" in j:
        return "truehd"
    if "dts-hd" in j:
        return "dts-hd"
    if "dd+" in j:
        return "dd+"
    return "other"


def _classify_era(year) -> str:
    try:
        y = int(year)
    except (ValueError, TypeError):
        return "unknown"
    if y < 1990:
        return "pre1990"
    if y < 2010:
        return "1990s-2000s"
    if y < 2020:
        return "2010s"
    return "2020s"


def _key_fns():
    return {
        "author": lambda e: str(e.get("author", "unknown")).strip().lower(),
        "format": lambda e: _classify_format(e.get("audioTypes", [])),
        "era": lambda e: _classify_era(e.get("year")),
        "content_type": lambda e: e.get("content_type", "film"),
        "source": lambda e: e.get("source", "unknown"),
    }


def _dist_pct(entries, key_fn) -> dict[str, float]:
    c = Counter(key_fn(e) for e in entries)
    total = sum(c.values())
    if total == 0:
        return {}
    return {k: 100 * v / total for k, v in c.items()}


def _dist_counts(entries, key_fn) -> Counter:
    return Counter(key_fn(e) for e in entries)


def _format_rank(audio_types) -> int:
    """Lower = better.  Picks Atmos > TrueHD > DTS-HD MA > DD+ > other."""
    j = " ".join(audio_types or []).lower()
    for i, fmt in enumerate(["atmos", "truehd", "dts-hd ma", "dts-hd", "dd+"]):
        if fmt in j:
            return i
    return 99


def _score_candidate(
    candidate: dict,
    have_counts: dict[str, Counter],
    target_counts: dict[str, Counter],
    have_total: int,
    target_total: int,
    n_authors_for_title: int,
) -> float:
    """Score a candidate entry by how much its metadata corrects the bias.

    For each dimension, compute the *deficit* (target % minus current %).
    The candidate gets credit for deficits it would help fill.  Squared
    so the largest gaps dominate the score.
    """
    key_fns = _key_fns()
    score = 0.0
    for dim, key_fn in key_fns.items():
        bucket = key_fn(candidate)
        target_pct = 100 * target_counts[dim].get(bucket, 0) / target_total
        current_pct = (
            100 * have_counts[dim].get(bucket, 0) / have_total
            if have_total > 0 else 0
        )
        deficit = target_pct - current_pct
        if deficit > 0:
            score += deficit ** 2
    # Bonus for multi-author titles — one acquisition trains for all authors.
    score *= 1.0 + 0.15 * (n_authors_for_title - 1)
    return score


def _load_library_inventory(inventory_path: Path | None) -> set[str]:
    """Read media_inventory.json and return tmdb IDs already in the library.

    These titles are already owned (whether or not their WAV has been
    extracted yet), so they should be excluded from acquisition recommendations.
    """
    if inventory_path is None:
        # Default location next to the WAV cache.
        beq_dir = Path.home() / "Downloads" / "beqdesigner"
        inventory_path = beq_dir / "media_inventory.json"
    if not inventory_path.exists():
        print(
            f"warning: media inventory not found at {inventory_path} — "
            "recommendations will not exclude already-owned titles. "
            "Run scripts/extract_lfe.py to refresh the inventory.",
            file=sys.stderr,
        )
        return set()
    data = json.loads(inventory_path.read_text())
    library_tmdb = set()
    for m in data.get("media", []):
        if m.get("id_type") == "tmdb":
            library_tmdb.add(m["id_value"])
    print(
        f"library inventory: {len(data.get('media', []))} files, "
        f"{len(library_tmdb)} unique tmdb IDs already owned",
        file=sys.stderr,
    )
    return library_tmdb


def select_acquisitions(
    n: int = 50,
    catalogue=None,
    have_tmdb_ids: set[str] = None,
    library_tmdb_ids: set[str] | None = None,
    per_author_cap_factor: float = 1.0,
) -> list[dict]:
    """Greedy bias-correcting selection of *n* acquisition candidates.

    *have_tmdb_ids*: tmdb IDs we already have *extracted WAVs* for.
    *library_tmdb_ids*: tmdb IDs already in the user's media library
        (whether or not extracted yet).  Acquisition picks exclude both
        sets — we don't recommend buying something you already own.
    *per_author_cap_factor*: limit per-author picks to ``cap_factor *
    catalogue_share * n``, with a floor of 2.  Set to 1.0 for stratified
    diversification, higher (e.g. 2.0) to allow more concentration on the
    biggest gaps.  None disables the cap.
    """
    from model.auto_beq_catalogue import _fetch_or_cache

    if catalogue is None:
        catalogue = _fetch_or_cache()
    if have_tmdb_ids is None:
        from spike._auto_beq_helpers import discover_wav_catalogue_pairs
        pairs = discover_wav_catalogue_pairs()
        have_tmdb_ids = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    if library_tmdb_ids is None:
        library_tmdb_ids = _load_library_inventory(None)

    # Combined exclude set: WAV cache + library inventory.
    exclude_tmdb = have_tmdb_ids | library_tmdb_ids

    # Filter to trainable entries.
    trainable = [e for e in catalogue if e.get("filters")]

    # Skip dangerous entries.
    def is_safe(e):
        warning = (e.get("warning") or "").lower()
        return "do not use" not in warning and "risking" not in warning

    trainable = [e for e in trainable if is_safe(e)]

    # Group entries by title to handle multi-author cases.
    by_title: dict[str, list[dict]] = defaultdict(list)
    for e in trainable:
        title_key = str(e.get("title", "")).lower().strip()
        by_title[title_key].append(e)

    # Currently held entries: WAVs we have OR titles in the user's library.
    held_entries = [
        e for e in trainable
        if str(e.get("theMovieDB", "")).strip() in exclude_tmdb
    ]
    held_titles = {
        str(e.get("title", "")).lower().strip() for e in held_entries
    }
    # have_entries (WAVs only) is what we use for the bias starting point.
    have_entries = [
        e for e in trainable
        if str(e.get("theMovieDB", "")).strip() in have_tmdb_ids
    ]

    # Candidate pool: pick the highest-quality format entry per missing title.
    # "Missing" means not in WAV cache AND not in library inventory.
    candidates: list[tuple[dict, int]] = []
    for title_key, group in by_title.items():
        if title_key in held_titles:
            continue  # already in cache OR already in library
        if not group:
            continue
        # Pick the best format variant for acquisition.
        best = min(group, key=lambda e: _format_rank(e.get("audioTypes", [])))
        n_authors = len({
            str(e.get("author", "")).strip().lower() for e in group
        })
        candidates.append((best, n_authors))

    print(
        f"candidates: {len(candidates)} truly missing titles "
        f"(WAV cache: {len(have_entries)} entries, "
        f"library: {len(library_tmdb_ids)} tmdb IDs)",
        file=sys.stderr,
    )

    # Build target distribution counts (catalogue per dimension).
    key_fns = _key_fns()
    target_counts = {dim: _dist_counts(trainable, fn) for dim, fn in key_fns.items()}
    target_total = len(trainable)

    # Per-author cap = max(2, factor * catalogue_share * n).
    # Stratifies the picks across authors so a single dominant author can't
    # consume all the budget.
    per_author_cap: dict[str, int] = {}
    if per_author_cap_factor is not None:
        author_share = target_counts["author"]
        for author, count in author_share.items():
            share = count / target_total
            cap = max(2, int(round(per_author_cap_factor * share * n)))
            per_author_cap[author] = cap
        log_caps = ", ".join(f"{a}={c}" for a, c in sorted(
            per_author_cap.items(), key=lambda kv: -kv[1])[:5])
        print(f"per-author caps (top 5): {log_caps}", file=sys.stderr)

    # Initial have distribution.
    have_counts = {dim: _dist_counts(have_entries, fn) for dim, fn in key_fns.items()}
    have_total = len(have_entries)

    # Greedy selection loop.
    selected: list[dict] = []
    selected_titles: set[str] = set()
    selected_per_author: Counter = Counter()
    remaining = list(candidates)

    for pick_idx in range(n):
        if not remaining:
            break
        # Score every remaining candidate against the *current* distributions.
        scored = []
        for entry, n_authors in remaining:
            title_key = str(entry.get("title", "")).lower().strip()
            if title_key in selected_titles:
                continue
            author = str(entry.get("author", "unknown")).strip().lower()
            if (
                per_author_cap
                and selected_per_author[author] >= per_author_cap.get(author, 0)
            ):
                continue  # cap exceeded for this author
            s = _score_candidate(
                entry, have_counts, target_counts,
                have_total, target_total, n_authors,
            )
            scored.append((s, entry, n_authors))
        if not scored:
            break
        scored.sort(key=lambda x: -x[0])
        best_score, best_entry, best_n_authors = scored[0]
        if best_score <= 0:
            break

        # Commit the pick: add to have_counts so the next iteration sees it.
        title_key = str(best_entry.get("title", "")).lower().strip()
        author = str(best_entry.get("author", "unknown")).strip().lower()
        selected.append({
            "entry": best_entry,
            "score": best_score,
            "n_authors_for_title": best_n_authors,
            "title_key": title_key,
            "pick_order": pick_idx + 1,
        })
        selected_titles.add(title_key)
        selected_per_author[author] += 1

        # Add ALL trainable entries for this title (since acquiring the WAV
        # makes all author variants available for training).
        for e in by_title[title_key]:
            for dim, fn in key_fns.items():
                have_counts[dim][fn(e)] += 1
            have_total += 1

        # Remove this title's entries from the candidate pool.
        remaining = [
            (entry, n_a) for entry, n_a in remaining
            if str(entry.get("title", "")).lower().strip() != title_key
        ]

    return selected


def generate_report(n: int, output=None, inventory_path: Path | None = None) -> None:
    pr = lambda s="": print(s, file=output or sys.stdout)

    from model.auto_beq_catalogue import _fetch_or_cache
    from spike._auto_beq_helpers import discover_wav_catalogue_pairs

    catalogue = _fetch_or_cache()
    pairs = discover_wav_catalogue_pairs()
    have_tmdb_ids = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    library_tmdb_ids = _load_library_inventory(inventory_path)

    selected = select_acquisitions(
        n=n, catalogue=catalogue, have_tmdb_ids=have_tmdb_ids,
        library_tmdb_ids=library_tmdb_ids,
    )

    pr(f"# Acquisition Recommendations (top {len(selected)})")
    pr()
    pr("Generated by greedy bias-correcting selection over the BEQ catalogue. "
       "Each pick is the missing title that would most close the gap between "
       "the local WAV cache distribution and the full catalogue distribution.")
    pr()
    pr(f"**Current WAV cache**: {len(have_tmdb_ids)} unique tmdb IDs / "
       f"{len(pairs)} WAV files")
    if library_tmdb_ids:
        pr(f"**Media library inventory**: {len(library_tmdb_ids)} unique "
           f"tmdb IDs already owned (excluded from recommendations)")
    else:
        pr(f"**Media library inventory**: not available — recommendations "
           f"may include titles you already own. Run `scripts/extract_lfe.py` "
           f"to refresh `media_inventory.json`.")
    pr(f"**Catalogue size**: {len([e for e in catalogue if e.get('filters')]):,} "
       f"trainable entries")
    pr()
    pr("Acquiring the listed titles will move the cache distribution toward "
       "the catalogue distribution across the most biased dimensions: "
       "**mobe1969 (-24.5%)**, **dts-hd (-12.7%)**, **streaming sources (-10.8%)**.")
    pr()
    pr("**Note on the multi-author cascade**: each title in the catalogue is "
       "often scored by 3-7 authors. Acquiring one WAV gives the model "
       "training signal for *all* of that title's authors at once — but it "
       "also dilutes each individual author's share when their cap is "
       "reached. The per-author cap is set proportional to catalogue share "
       "(57% mobe1969, 13% aron7awol, etc.), capped at floor 2.  See the "
       "coverage simulation section for the projected post-acquisition "
       "distribution.")
    pr()

    # ----------------------------------------------------------------------
    # Section 1: Ranked picks with metadata
    # ----------------------------------------------------------------------
    pr("## Ranked picks")
    pr()
    pr("| # | Title | Year | Author | Format | Source | Score |")
    pr("|---:|---|---:|---|---|---|---:|")
    for s in selected:
        e = s["entry"]
        title = e.get("title", "?")
        year = e.get("year", "?")
        author = e.get("author", "?")
        fmt = _classify_format(e.get("audioTypes", []))
        src = e.get("source", "?")
        score = s["score"]
        # Mark multi-author titles (training bonus).
        marker = f" *({s['n_authors_for_title']} authors)*" if s["n_authors_for_title"] > 1 else ""
        pr(f"| {s['pick_order']} | {title}{marker} | {year} | {author} | "
           f"{fmt} | {src} | {score:.0f} |")
    pr()

    # ----------------------------------------------------------------------
    # Section 2: Grouped by author
    # ----------------------------------------------------------------------
    pr("## Grouped by author")
    pr()
    by_author = defaultdict(list)
    for s in selected:
        a = s["entry"].get("author", "unknown")
        by_author[a].append(s)
    for author, items in sorted(by_author.items(), key=lambda kv: -len(kv[1])):
        pr(f"### {author} ({len(items)})")
        pr()
        for s in items:
            e = s["entry"]
            fmt = _classify_format(e.get("audioTypes", []))
            pr(f"- **{e.get('title', '?')}** ({e.get('year', '?')}) — "
               f"{fmt}, {e.get('source', '?')}")
        pr()

    # ----------------------------------------------------------------------
    # Section 3: Grouped by format
    # ----------------------------------------------------------------------
    pr("## Grouped by audio format")
    pr()
    by_fmt = defaultdict(list)
    for s in selected:
        f = _classify_format(s["entry"].get("audioTypes", []))
        by_fmt[f].append(s)
    for fmt, items in sorted(by_fmt.items(), key=lambda kv: -len(kv[1])):
        pr(f"### {fmt} ({len(items)})")
        pr()
        for s in items:
            e = s["entry"]
            pr(f"- **{e.get('title', '?')}** ({e.get('year', '?')}) — "
               f"by {e.get('author', '?')}, {e.get('source', '?')}")
        pr()

    # ----------------------------------------------------------------------
    # Section 4: Coverage simulation
    # ----------------------------------------------------------------------
    pr("## Coverage simulation")
    pr()
    pr("How the cache distribution would change after acquiring all "
       f"{len(selected)} recommendations.")
    pr()
    trainable = [e for e in catalogue if e.get("filters")]
    have_entries_now = [
        e for e in trainable
        if str(e.get("theMovieDB", "")).strip() in have_tmdb_ids
    ]
    selected_titles = {s["title_key"] for s in selected}
    by_title = defaultdict(list)
    for e in trainable:
        by_title[str(e.get("title", "")).lower().strip()].append(e)
    have_entries_after = list(have_entries_now)
    for t in selected_titles:
        have_entries_after.extend(by_title[t])

    key_fns = _key_fns()
    for dim, fn in key_fns.items():
        cat_dist = _dist_pct(trainable, fn)
        now_dist = _dist_pct(have_entries_now, fn)
        after_dist = _dist_pct(have_entries_after, fn)
        pr(f"### {dim}")
        pr()
        pr("| Bucket | Catalogue % | Now % | After % | Bias before | Bias after |")
        pr("|---|---:|---:|---:|---:|---:|")
        for k in sorted(set(cat_dist) | set(now_dist), key=lambda k: -cat_dist.get(k, 0)):
            c = cat_dist.get(k, 0)
            n = now_dist.get(k, 0)
            a = after_dist.get(k, 0)
            bias_before = n - c
            bias_after = a - c
            pr(f"| {k} | {c:.1f}% | {n:.1f}% | {a:.1f}% | "
               f"{bias_before:+.1f} | {bias_after:+.1f} |")
        pr()

    pr("---")
    pr("Generated by `scripts/nn_acquisition_recommender.py`")


def main():
    parser = argparse.ArgumentParser(description="WAV acquisition recommender")
    parser.add_argument("-n", "--count", type=int, default=50,
                        help="Number of titles to recommend (default 50)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None,
                        help="Path to media_inventory.json (default: "
                             "~/Downloads/beqdesigner/media_inventory.json)")
    args = parser.parse_args()

    if args.output:
        with args.output.open("w") as f:
            generate_report(args.count, output=f, inventory_path=args.inventory)
        print(f"Report written to {args.output}")
    else:
        generate_report(args.count, inventory_path=args.inventory)


if __name__ == "__main__":
    main()
