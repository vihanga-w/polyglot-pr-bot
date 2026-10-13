"""Pure, provider-free logic for the parity tool.

Given the PR's changed files, the repository tree and a ``ParityConfig``, work out:

* which language implementations under ``root`` the PR touched and which it did not,
* which changed files are shared (``shared_dir``) or outside the polyglot layout,
* for each untouched language, the most likely files the equivalent change belongs in,
* which services (monorepo microservices) the PR touches.

Nothing here talks to a git provider or a model, so it is unit-testable offline.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass, field

from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
from pr_agent.parity.config import DEFAULT_LANGUAGE_EXTENSIONS, ParityConfig

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_\-\.\s/]+")
_STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "self", "none", "null", "true", "false", "return",
    "import", "export", "const", "let", "var", "func", "def", "pub", "use", "mod", "struct", "impl", "class",
    "async", "await", "new", "string", "int", "bool", "err", "error", "nil", "type", "interface",
    "package", "mut", "match", "else", "then", "void", "static", "public", "private", "test", "tests", "src",
    "index", "main", "lib", "init", "utils", "util",
}
_TEST_HINTS = ("test", "spec", "_test", ".test.", ".spec.")


@dataclass
class ChangedFile:
    path: str
    language: str | None  # None => shared or outside the layout
    relative: str  # path relative to the language dir (or to root/shared/repo root)
    edit_type: EDIT_TYPE
    zone: str  # "language" | "shared" | "outside"


@dataclass
class Candidate:
    path: str
    score: float
    reasons: list[str] = field(default_factory=list)
    identifier_hits: list[str] = field(default_factory=list)


@dataclass
class LanguageReport:
    language: str
    directory: str
    changed_files: list[ChangedFile] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return bool(self.changed_files)


@dataclass
class ParityLayout:
    config: ParityConfig
    languages: list[LanguageReport]
    shared_files: list[ChangedFile]
    outside_files: list[ChangedFile]
    services_touched: dict[str, list[str]]  # service dir -> changed files
    services_all: list[str]
    identifiers: list[str]

    @property
    def touched_languages(self) -> list[LanguageReport]:
        return [lang for lang in self.languages if lang.touched]

    @property
    def missing_languages(self) -> list[LanguageReport]:
        return [lang for lang in self.languages if not lang.touched]

    @property
    def in_layout(self) -> bool:
        """True when the PR touched anything under ``root`` (a language dir or shared)."""
        return bool(self.touched_languages or self.shared_files)


# ---------------------------------------------------------------------------------------------
# Language discovery
# ---------------------------------------------------------------------------------------------

def _top_level_dirs(config: ParityConfig, tree_paths: list[str]) -> dict[str, str]:
    """Lower-cased name -> real name of every directory directly under ``root``."""
    prefix = config.root + "/" if config.root else ""
    found: dict[str, str] = {}
    for path in tree_paths:
        if prefix and not path.startswith(prefix):
            continue
        head, sep, _tail = path[len(prefix):].partition("/")
        if sep and head and head.lower() not in found:
            found[head.lower()] = head
    return found


def _join_root(config: ParityConfig, name: str) -> str:
    return posixpath.join(config.root, name) if config.root else name


def discover_languages(config: ParityConfig, tree_paths: list[str]) -> list[tuple[str, str]]:
    """``(language, directory)`` pairs: configured languages whose directory exists (matching
    aliases such as ``go`` for ``golang`` and ignoring case), then, when ``auto_detect_languages``
    is on, any other directory under ``root`` that holds a recognisable source file.

    Without a tree (provider cannot list one) every configured language is assumed to exist at
    ``root/<language>`` so the PR's own files can still be bucketed.
    """
    if not tree_paths:
        return [(lang, _join_root(config, lang)) for lang in config.languages]

    dirs = _top_level_dirs(config, tree_paths)
    shared_name = posixpath.basename(config.shared_dir).lower() if config.shared_dir else ""
    result: list[tuple[str, str]] = []
    claimed: set[str] = set()
    for lang in config.languages:
        for alias in config.dir_names_for(lang):
            real = dirs.get(alias)
            if real is not None and alias not in claimed:
                claimed.add(alias)
                result.append((lang, _join_root(config, real)))
                break
    if not config.auto_detect_languages:
        return result

    known_exts = set()
    for exts in DEFAULT_LANGUAGE_EXTENSIONS.values():
        known_exts.update(exts)
    for lower, real in sorted(dirs.items()):
        if lower in claimed or lower == shared_name or real.startswith(".") or lower in config.ignore_path_segments:
            continue
        dir_prefix = _join_root(config, real) + "/"
        if any(p.startswith(dir_prefix) and posixpath.splitext(p)[1].lower() in known_exts
               and not _ignored(p, config) for p in tree_paths):
            result.append((lower, _join_root(config, real)))
    return result


def _is_source_file(path: str, config: ParityConfig, language: str | None = None) -> bool:
    """A file worth proposing as a target: the language's own extensions when known, otherwise any
    extension of a language this module knows about (never assets, docs or lockfiles)."""
    ext = posixpath.splitext(path)[1].lower()
    if language:
        exts = config.extensions_for(language)
        if exts:
            return ext in exts
    return any(ext in exts for exts in DEFAULT_LANGUAGE_EXTENSIONS.values())


def _ignored(path: str, config: ParityConfig) -> bool:
    parts = path.split("/")
    return any(part in config.ignore_path_segments for part in parts)


# ---------------------------------------------------------------------------------------------
# Bucketing changed files
# ---------------------------------------------------------------------------------------------

def _match_language_dir(languages: list[tuple[str, str]], path: str) -> tuple[str, str] | None:
    for language, d in languages:
        if path == d or path.startswith(d + "/"):
            return language, path[len(d) + 1:]
    return None


def bucket_changed_files(config: ParityConfig, languages: list[tuple[str, str]],
                         diff_files: list[FilePatchInfo]) -> tuple[dict[str, list[ChangedFile]],
                                                                    list[ChangedFile], list[ChangedFile]]:
    by_language: dict[str, list[ChangedFile]] = {lang: [] for lang, _ in languages}
    shared: list[ChangedFile] = []
    outside: list[ChangedFile] = []
    for f in diff_files:
        path = (f.filename or "").replace("\\", "/").lstrip("/")
        if not path:
            continue
        edit_type = f.edit_type if isinstance(f.edit_type, EDIT_TYPE) else EDIT_TYPE.UNKNOWN
        matched = _match_language_dir(languages, path)
        if matched:
            lang, rel = matched
            by_language[lang].append(ChangedFile(path, lang, rel, edit_type, "language"))
            continue
        if config.shared_dir and (path == config.shared_dir or path.startswith(config.shared_dir + "/")):
            shared.append(ChangedFile(path, None, path[len(config.shared_dir) + 1:], edit_type, "shared"))
            continue
        outside.append(ChangedFile(path, None, path, edit_type, "outside"))
    return by_language, shared, outside


# ---------------------------------------------------------------------------------------------
# Identifier extraction from the diff (used to score candidates)
# ---------------------------------------------------------------------------------------------

def _split_words(token: str) -> list[str]:
    return [w.lower() for w in _CAMEL_SPLIT_RE.split(token) if w]


def extract_identifiers(diff_files: list[FilePatchInfo], limit: int = 40) -> list[str]:
    """Identifiers that appear on added/removed lines, most frequent first. Case-insensitive
    and camel/snake-normalised words are kept separately so ``getUserName`` in TypeScript can
    match ``get_user_name`` in Python."""
    counts: dict[str, int] = {}
    for f in diff_files:
        for line in (f.patch or "").splitlines():
            if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
                continue
            for ident in _IDENT_RE.findall(line[1:]):
                low = ident.lower()
                if low in _STOP_WORDS or len(low) < 4:
                    continue
                counts[ident] = counts.get(ident, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ident for ident, _ in ranked[:limit]]


def identifier_variants(identifier: str) -> set[str]:
    words = _split_words(identifier)
    variants = {identifier.lower()}
    if len(words) > 1:
        variants.add("".join(words))
        variants.add("_".join(words))
        variants.add("-".join(words))
    return variants


# ---------------------------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------------------------

def _stem_words(relative_path: str) -> set[str]:
    no_ext = posixpath.splitext(relative_path)[0]
    words = set(_split_words(no_ext))
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _is_test_path(path: str) -> bool:
    low = path.lower()
    return any(h in low for h in _TEST_HINTS)


def rank_candidates(config: ParityConfig, language: str, tree_paths: list[str],
                    changed: list[ChangedFile], identifiers: list[str],
                    read_file=None, directory: str | None = None) -> list[Candidate]:
    """Rank files under ``language``'s directory as homes for the equivalent change.

    Scoring (highest first):
      +10 identical relative path ignoring extension
      +6  identical basename ignoring extension
      +1..4 overlap between path words (e.g. ``auth/token`` vs ``auth/tokens.rs``)
      +2 per changed identifier found in the file's content (``read_file``, capped by
         ``max_candidate_content_fetches``, only for the top path-scored files)
      -3 if the candidate looks like a test file and none of the changed files do
    An empty list means nothing in the language directory matched by path or (for the few files
    read) by identifier; callers must present that as "no match found", not as proof that a new
    file is required.
    """
    lang_dir = directory or _join_root(config, language)
    prefix = lang_dir + "/"
    files = [p for p in tree_paths
             if p.startswith(prefix) and _is_source_file(p, config, language) and not _ignored(p, config)]
    if not files:
        return []

    source_changes = [c for c in changed if c.zone == "language"]
    changed_rel_noext = {posixpath.splitext(c.relative)[0].lower() for c in source_changes}
    changed_basenames = {posixpath.splitext(posixpath.basename(c.relative))[0].lower() for c in source_changes}
    changed_words: set[str] = set()
    for c in source_changes:
        changed_words |= _stem_words(c.relative)
    changed_has_tests = any(_is_test_path(c.path) for c in source_changes)

    scored: list[Candidate] = []
    for path in files:
        rel = path[len(prefix):]
        rel_noext = posixpath.splitext(rel)[0].lower()
        base = posixpath.splitext(posixpath.basename(rel))[0].lower()
        score = 0.0
        reasons: list[str] = []
        if rel_noext in changed_rel_noext:
            score += 10
            reasons.append("same relative path")
        elif base in changed_basenames:
            score += 6
            reasons.append("same file name")
        overlap = _stem_words(rel) & changed_words
        if overlap:
            bump = min(4, len(overlap))
            score += bump
            reasons.append("path words: " + ", ".join(sorted(overlap)))
        if _is_test_path(path) and not changed_has_tests:
            score -= 3
        if score > 0:
            scored.append(Candidate(path, score, reasons))

    scored.sort(key=lambda c: (-c.score, c.path))

    # Content pass: only for the strongest path matches, bounded by config.
    if read_file is not None and identifiers and config.max_candidate_content_fetches:
        variants = {ident: identifier_variants(ident) for ident in identifiers}
        pool = scored[:config.max_candidate_content_fetches] if scored else []
        # If path scoring found nothing, peek at a few files anyway, shallowest and smallest-named first
        # (library roots such as lib.rs / index.ts / __init__.py are the likeliest homes).
        if not pool:
            plausible = sorted((p for p in files if not _is_test_path(p)),
                               key=lambda p: (p.count("/"), len(posixpath.basename(p)), p))
            pool = [Candidate(p, 0.0) for p in plausible[:config.max_candidate_content_fetches]]
            scored = pool
        for cand in pool:
            try:
                content = read_file(cand.path) or ""
            except Exception:  # noqa: BLE001 - a missing/unreadable file just scores on path alone
                continue
            low = content.lower()
            hits = [ident for ident, vs in variants.items() if any(v in low for v in vs)]
            if hits:
                cand.identifier_hits = hits[:8]
                cand.score += 2 * min(len(hits), 5)
                cand.reasons.append(f"mentions {len(hits)} changed identifier(s)")
        scored.sort(key=lambda c: (-c.score, c.path))

    return [c for c in scored if c.score > 0][:config.max_candidates_per_language]


def suggested_new_paths(config: ParityConfig, language: str, changed: list[ChangedFile],
                        directory: str | None = None) -> list[str]:
    """When no existing file matches (or the change is a new file), the path the equivalent file
    would most naturally take: same relative path, language's primary extension."""
    exts = config.extensions_for(language)
    ext = exts[0] if exts else ""
    lang_dir = directory or _join_root(config, language)
    out: list[str] = []
    for c in changed:
        if c.zone != "language":
            continue
        rel_noext = posixpath.splitext(c.relative)[0]
        out.append(posixpath.join(lang_dir, rel_noext + ext))
    return out


# ---------------------------------------------------------------------------------------------
# Services (monorepo microservices)
# ---------------------------------------------------------------------------------------------

def detect_services(config: ParityConfig, tree_paths: list[str]) -> list[str]:
    """Directories matching ``service_dir_patterns`` that contain at least one source file."""
    found: set[str] = set()
    for path in tree_paths:
        if _ignored(path, config):
            continue
        parts = path.split("/")
        for pattern in config.service_dir_patterns:
            depth = pattern.count("/") + 1
            if len(parts) <= depth:
                continue
            candidate_dir = "/".join(parts[:depth])
            if fnmatch.fnmatchcase(candidate_dir, pattern) and _is_source_file(path, config):
                found.add(candidate_dir)
    return sorted(found)


def services_for_files(services: list[str], paths: list[str]) -> dict[str, list[str]]:
    touched: dict[str, list[str]] = {}
    for p in paths:
        for s in services:
            if p.startswith(s + "/"):
                touched.setdefault(s, []).append(p)
                break
    return touched


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def build_layout(config: ParityConfig, diff_files: list[FilePatchInfo], tree_paths: list[str],
                 read_file=None) -> ParityLayout:
    tree_paths = [p.replace("\\", "/").lstrip("/") for p in tree_paths or []]
    languages = discover_languages(config, tree_paths)
    by_language, shared, outside = bucket_changed_files(config, languages, diff_files)
    identifiers = extract_identifiers(diff_files)
    touched_changed = [c for lang, _ in languages for c in by_language[lang]]

    reports: list[LanguageReport] = []
    for lang, directory in languages:
        report = LanguageReport(language=lang, directory=directory, changed_files=by_language[lang])
        if not report.touched and touched_changed:
            report.candidates = rank_candidates(config, lang, tree_paths, touched_changed, identifiers, read_file,
                                                directory=directory)
        reports.append(report)

    services = detect_services(config, tree_paths) if config.check_distribution else []
    changed_paths = [f.filename for f in diff_files if f.filename]
    return ParityLayout(
        config=config,
        languages=reports,
        shared_files=shared,
        outside_files=outside,
        services_touched=services_for_files(services, changed_paths),
        services_all=services,
        identifiers=identifiers,
    )
