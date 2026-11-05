"""
Model catalogue: which bbox detectors, image encoders and SAM 3D Body checkpoints exist,
where each one comes from, and which of them get fetched.

The shipped catalogue is ``fbxify/models.json``. A user file (``FBXIFY_MODELS_JSON`` or
``CHECKPOINTS_DIR/models.json``) is overlaid on it: matching names patch the shipped entry,
new names are added. Only entries with ``enabled: true`` are fetched at start-up and listed
in the dropdowns; ``default: true`` picks what loads first.

Sources:
    {"kind": "huggingface", "repo_id": ..., "revision": ..., "files": [{"path", "sha256", "size"}]}
    {"kind": "url", "url": ..., "filename": ..., "sha256": ..., "size": ...}
    {"kind": "local", "path": "/workspace/my-model"}            never downloaded, only checked
    {"kind": "none"}                                              nothing on disk

``${CHECKPOINTS_DIR}`` and ``${CACHE_DIR}`` are expanded in paths. Shipped entries must carry
sha256 + size for every file; custom entries may omit them (then only existence is checked).

Command line:
    python -m fbxify.model_catalog list [--json]
    python -m fbxify.model_catalog fetch [--all] [--section S] [--name N]
    python -m fbxify.model_catalog verify [--deep] [--all]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

SCHEMA = "fbxify.models/1"
SECTIONS = ("detectors", "encoders", "body_models")
SOURCE_KINDS = ("huggingface", "url", "local", "none")
BUILTIN_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")

# Files a type needs when a source does not list them (used for kind=local and url).
TYPE_DEFAULT_FILES = {
    "sam_3d_body": ["model.ckpt", "assets/mhr_model.pt"],
    "detectron2_vitdet": ["model_final_f05665.pkl"],
    "ultralytics": [],
    "embedl_int8": [],
    "checkpoint": [],
}
EMBEDL_FILES_BY_MODE = {
    "pt2": ["embedl_sam3dbody_int8.pt2"],
    "trt": ["embedl_sam3dbody_int8.onnx", "embedl_sam3dbody_int8.onnx.data"],
}
# Legacy env/CLI value of FBXIFY_QUANTIZED_BACKBONE -> encoder entry name.
QUANTIZED_MODE_TO_ENCODER = {None: "fp32", "none": "fp32", "pt2": "embedl_int8_pt2", "trt": "embedl_int8_trt"}

_DOWNLOAD_CHUNK = 8 * 1024 * 1024


class CatalogError(ValueError):
    """Invalid catalogue content or an unknown model name."""


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_dirs() -> Dict[str, str]:
    checkpoints = os.environ.get("CHECKPOINTS_DIR", "/fbxify/checkpoints").rstrip("/\\") or "/fbxify/checkpoints"
    cache = os.environ.get("CACHE_DIR", "/fbxify/cache").rstrip("/\\") or "/fbxify/cache"
    return {"CHECKPOINTS_DIR": checkpoints, "CACHE_DIR": cache}


def expand_path(value: str, dirs: Optional[Dict[str, str]] = None) -> str:
    """Expand ${CHECKPOINTS_DIR} / ${CACHE_DIR} (and ordinary env vars, ~) in a catalogue path."""
    dirs = dirs or _default_dirs()
    text = str(value)
    for key, val in dirs.items():
        text = text.replace("${" + key + "}", val).replace("$" + key, val)
    return os.path.normpath(os.path.expanduser(os.path.expandvars(text)))


@dataclass
class ModelFile:
    path: str                       # relative to the entry's local_dir
    sha256: Optional[str] = None
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path}
        if self.sha256:
            out["sha256"] = self.sha256
        if self.size is not None:
            out["size"] = self.size
        return out


@dataclass
class ModelEntry:
    section: str
    name: str
    display_name: str
    type: str
    enabled: bool = False
    default: bool = False
    licence: str = ""
    source: Dict[str, Any] = field(default_factory=lambda: {"kind": "none"})
    local_dir: str = ""             # expanded absolute directory holding ``files``
    files: List[ModelFile] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    requires_body: List[str] = field(default_factory=list)
    notes: str = ""
    builtin: bool = True            # shipped entry with an unmodified source -> checksums required
    origin: str = "builtin"         # "builtin" or the user file the entry came from

    # ---- file bookkeeping -------------------------------------------------
    @property
    def kind(self) -> str:
        return str(self.source.get("kind", "none"))

    def file_path(self, rel: str) -> str:
        return os.path.join(self.local_dir, *rel.split("/"))

    def file_paths(self) -> List[str]:
        return [self.file_path(f.path) for f in self.files]

    def missing_files(self) -> List[str]:
        return [f.path for f in self.files if not os.path.isfile(self.file_path(f.path))]

    def size_mismatches(self) -> List[str]:
        bad = []
        for f in self.files:
            if f.size is None:
                continue
            p = self.file_path(f.path)
            if os.path.isfile(p) and os.path.getsize(p) != f.size:
                bad.append(f.path)
        return bad

    def status(self) -> str:
        """disabled | ready | missing | size-mismatch | ready-unverified (no checksum data)."""
        if not self.enabled:
            return "disabled"
        if not self.files:
            return "ready"
        if self.missing_files():
            return "missing"
        if self.size_mismatches():
            return "size-mismatch"
        if any(f.sha256 is None and f.size is None for f in self.files):
            return "ready-unverified"
        return "ready"

    def is_ready(self) -> bool:
        return self.status() in ("ready", "ready-unverified")

    def verify(self, deep: bool = False) -> Tuple[bool, str]:
        """Check presence and size; with deep=True also sha256 for files that declare one."""
        if not self.files:
            return True, "no files"
        missing = self.missing_files()
        if missing:
            return False, "missing: " + ", ".join(missing)
        bad = self.size_mismatches()
        if bad:
            return False, "size mismatch: " + ", ".join(bad)
        if deep:
            for f in self.files:
                if not f.sha256:
                    continue
                got = _sha256_file(self.file_path(f.path))
                if got.lower() != f.sha256.lower():
                    return False, f"sha256 mismatch for {f.path}: {got}"
            return True, "verified (sha256)"
        return True, "present (size ok)"

    def describe(self, loaded: bool = False) -> Dict[str, Any]:
        """JSON-safe summary for /models and the CLI. Never includes absolute paths."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.type,
            "enabled": self.enabled,
            "default": self.default,
            "licence": self.licence,
            "source_kind": self.kind,
            "status": self.status(),
            "loaded": loaded,
            "requires_body": list(self.requires_body),
            "options": copy.deepcopy(self.options),
            "notes": self.notes,
            "origin": self.origin,
        }


# --------------------------------------------------------------------------- loading


def _parse_files(raw_files: Any) -> List[ModelFile]:
    out: List[ModelFile] = []
    for item in raw_files or []:
        if isinstance(item, str):
            out.append(ModelFile(path=item))
        elif isinstance(item, dict) and item.get("path"):
            size = item.get("size")
            out.append(ModelFile(path=str(item["path"]), sha256=item.get("sha256"), size=int(size) if size is not None else None))
        else:
            raise CatalogError(f"bad file spec {item!r}")
    return out


def _entry_from_dict(section: str, name: str, raw: Dict[str, Any], dirs: Dict[str, str], builtin: bool, origin: str) -> ModelEntry:
    if not isinstance(raw, dict):
        raise CatalogError(f"{section}.{name}: entry must be an object")
    source = dict(raw.get("source") or {"kind": "none"})
    kind = str(source.get("kind", "none"))
    if kind not in SOURCE_KINDS:
        raise CatalogError(f"{section}.{name}: source.kind must be one of {SOURCE_KINDS}, got {kind!r}")
    etype = str(raw.get("type", "")).strip()
    if not etype:
        raise CatalogError(f"{section}.{name}: 'type' is required")
    options = dict(raw.get("options") or {})

    # Resolve local_dir and the file list from the source kind.
    files: List[ModelFile] = []
    local_dir = raw.get("local_dir")
    if kind == "huggingface":
        if not source.get("repo_id"):
            raise CatalogError(f"{section}.{name}: huggingface source needs repo_id")
        files = _parse_files(source.get("files"))
        if not files:
            raise CatalogError(f"{section}.{name}: huggingface source needs a 'files' list")
        local_dir = local_dir or "${CHECKPOINTS_DIR}/" + source["repo_id"].split("/")[-1]
    elif kind == "url":
        if not source.get("url"):
            raise CatalogError(f"{section}.{name}: url source needs url")
        filename = source.get("filename") or os.path.basename(str(source["url"]).split("?")[0])
        size = source.get("size")
        files = [ModelFile(path=filename, sha256=source.get("sha256"), size=int(size) if size is not None else None)]
        local_dir = local_dir or "${CACHE_DIR}/" + name
    elif kind == "local":
        path = source.get("path") or source.get("dir") or local_dir
        if not path:
            raise CatalogError(f"{section}.{name}: local source needs 'path'")
        expanded = expand_path(path, dirs)
        listed = _parse_files(source.get("files"))
        if listed:
            local_dir, files = expanded, listed
        elif os.path.isfile(expanded) or (not os.path.isdir(expanded) and os.path.splitext(expanded)[1]):
            # A single file: directory + that file.
            local_dir, files = os.path.dirname(expanded), [ModelFile(path=os.path.basename(expanded))]
        else:
            local_dir = expanded
            defaults = list(TYPE_DEFAULT_FILES.get(etype, []))
            if etype == "embedl_int8":
                defaults = list(EMBEDL_FILES_BY_MODE.get(str(options.get("mode", "trt")), []))
            files = [ModelFile(path=p) for p in defaults]
    else:  # none
        local_dir = local_dir or ""

    entry = ModelEntry(
        section=section,
        name=name,
        display_name=str(raw.get("display_name") or name),
        type=etype,
        enabled=bool(raw.get("enabled", False)),
        default=bool(raw.get("default", False)),
        licence=str(raw.get("licence") or raw.get("license") or ""),
        source=source,
        local_dir=expand_path(local_dir, dirs) if local_dir else "",
        files=files,
        options=options,
        requires_body=[str(x) for x in (raw.get("requires_body") or [])],
        notes=str(raw.get("notes") or ""),
        builtin=builtin,
        origin=origin,
    )
    if etype == "embedl_int8" and entry.kind == "huggingface":
        # Keep the file list honest for the selected mode: trt needs onnx+data, pt2 needs the pt2.
        needed = EMBEDL_FILES_BY_MODE.get(str(options.get("mode", "trt")), [])
        have = {f.path for f in entry.files}
        for p in needed:
            if p not in have:
                raise CatalogError(f"{section}.{name}: embedl_int8 mode {options.get('mode')!r} needs file {p}")
    return entry


class ModelCatalog:
    def __init__(self, entries: Dict[str, Dict[str, ModelEntry]], sources: List[str]):
        self.entries = entries
        self.sources = sources

    # ---- construction -------------------------------------------------
    @classmethod
    def load(cls, user_paths: Optional[Iterable[str]] = None, builtin_path: str = BUILTIN_CATALOG_PATH,
             dirs: Optional[Dict[str, str]] = None, strict: bool = True) -> "ModelCatalog":
        """Load the shipped catalogue and overlay user files.

        user_paths: explicit overlay files. Default: FBXIFY_MODELS_JSON if set, else
        CHECKPOINTS_DIR/models.json when it exists.
        """
        dirs = dirs or _default_dirs()
        with open(builtin_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("schema") != SCHEMA:
            raise CatalogError(f"{builtin_path}: schema must be {SCHEMA!r}")
        merged: Dict[str, Dict[str, Dict[str, Any]]] = {s: copy.deepcopy(raw.get(s) or {}) for s in SECTIONS}
        builtin_flags: Dict[str, Dict[str, bool]] = {s: {n: True for n in merged[s]} for s in SECTIONS}
        origins: Dict[str, Dict[str, str]] = {s: {n: "builtin" for n in merged[s]} for s in SECTIONS}
        sources = [builtin_path]

        if user_paths is None:
            user_paths = default_user_paths(dirs)
        for upath in user_paths:
            if not upath or not os.path.isfile(upath):
                continue
            with open(upath, "r", encoding="utf-8") as fh:
                user_raw = json.load(fh)
            if user_raw.get("schema", SCHEMA) != SCHEMA:
                raise CatalogError(f"{upath}: schema must be {SCHEMA!r}")
            sources.append(upath)
            for section in SECTIONS:
                for name, patch in (user_raw.get(section) or {}).items():
                    if patch is None or patch is False:
                        merged[section].pop(name, None)
                        continue
                    if not isinstance(patch, dict):
                        raise CatalogError(f"{upath}: {section}.{name} must be an object")
                    if name in merged[section]:
                        base = merged[section][name]
                        base.update(patch)
                        if "source" in patch or "type" in patch or "local_dir" in patch:
                            builtin_flags[section][name] = False
                    else:
                        merged[section][name] = dict(patch)
                        builtin_flags[section][name] = False
                    origins[section][name] = upath
                    if patch.get("default"):
                        for other in merged[section]:
                            if other != name:
                                merged[section][other]["default"] = False

        # A user file may disable the shipped default; the first enabled entry then takes over.
        for section in SECTIONS:
            table = merged[section]
            enabled = [n for n, it in table.items() if it.get("enabled")]
            defaults = [n for n, it in table.items() if it.get("default")]
            if defaults and defaults[0] not in enabled:
                for n in defaults:
                    table[n]["default"] = False
                if enabled:
                    table[enabled[0]]["default"] = True
                    print(f"model_catalog: {section}: default {defaults[0]!r} is disabled; using {enabled[0]!r}", flush=True)

        entries: Dict[str, Dict[str, ModelEntry]] = {}
        for section in SECTIONS:
            entries[section] = {}
            for name, item in merged[section].items():
                entries[section][name] = _entry_from_dict(
                    section, name, item, dirs, builtin_flags[section][name], origins[section][name]
                )
        catalog = cls(entries, sources)
        problems = catalog.validate()
        if problems and strict:
            raise CatalogError("invalid model catalogue:\n  " + "\n  ".join(problems))
        return catalog

    def validate(self) -> List[str]:
        problems: List[str] = []
        body_names = set(self.entries.get("body_models", {}))
        for section in SECTIONS:
            defaults = [e.name for e in self.entries[section].values() if e.default]
            if len(defaults) > 1:
                problems.append(f"{section}: more than one default ({', '.join(defaults)})")
            for e in self.entries[section].values():
                if e.default and not e.enabled:
                    problems.append(f"{section}.{e.name}: default entry must be enabled")
                if e.builtin and e.kind in ("huggingface", "url"):
                    for f in e.files:
                        if not f.sha256 or f.size is None:
                            problems.append(f"{section}.{e.name}: shipped file {f.path} needs sha256 and size")
                for b in e.requires_body:
                    if b not in body_names:
                        problems.append(f"{section}.{e.name}: requires_body references unknown body model {b!r}")
                if section == "encoders" and e.type == "embedl_int8" and str(e.options.get("mode")) not in EMBEDL_FILES_BY_MODE:
                    problems.append(f"{section}.{e.name}: embedl_int8 needs options.mode pt2|trt")
        return problems

    # ---- lookup ---------------------------------------------------------
    def section(self, section: str) -> Dict[str, ModelEntry]:
        if section not in SECTIONS:
            raise CatalogError(f"unknown section {section!r}; expected one of {SECTIONS}")
        return self.entries[section]

    def get(self, section: str, name: Optional[str]) -> ModelEntry:
        """Entry by name; None or "" means the section default. Raises CatalogError when unknown."""
        table = self.section(section)
        if name is None or str(name).strip() == "":
            return self.default(section)
        key = str(name).strip()
        if key not in table:
            raise CatalogError(f"unknown {section[:-1].replace('_', ' ')} {key!r}; known: {', '.join(sorted(table))}")
        return table[key]

    def default(self, section: str) -> ModelEntry:
        table = self.section(section)
        for e in table.values():
            if e.default:
                return e
        for e in table.values():
            if e.enabled:
                return e
        raise CatalogError(f"{section}: no default entry")

    def names(self, section: str, enabled_only: bool = True) -> List[str]:
        return [n for n, e in self.section(section).items() if e.enabled or not enabled_only]

    def enabled_entries(self, sections: Optional[Iterable[str]] = None) -> List[ModelEntry]:
        out = []
        for s in sections or SECTIONS:
            out.extend(e for e in self.section(s).values() if e.enabled)
        return out

    def all_entries(self, sections: Optional[Iterable[str]] = None) -> List[ModelEntry]:
        out = []
        for s in sections or SECTIONS:
            out.extend(self.section(s).values())
        return out

    def force_enable(self, section: str, name: Optional[str]) -> Optional[ModelEntry]:
        """Enable an entry named by an explicit env/CLI choice (legacy FBXIFY_MODEL etc.)."""
        if name is None or str(name).strip() == "":
            return None
        entry = self.get(section, name)
        if not entry.enabled:
            print(f"model_catalog: {section}.{entry.name} was disabled; enabling because it was selected explicitly", flush=True)
            entry.enabled = True
        return entry

    # ---- resolution for loaders ----------------------------------------
    def resolve_body(self, name: Optional[str]) -> Tuple[ModelEntry, str, str]:
        """(entry, checkpoint_path, mhr_path) for a body model."""
        e = self.get("body_models", name)
        if e.type != "sam_3d_body":
            raise CatalogError(f"body model {e.name!r} has unsupported type {e.type!r}")
        ckpt = e.file_path(str(e.options.get("checkpoint_file", "model.ckpt")))
        mhr = e.file_path(str(e.options.get("mhr_file", "assets/mhr_model.pt")))
        return e, ckpt, mhr

    def resolve_encoder(self, name: Optional[str], body: Optional[str] = None) -> Tuple[ModelEntry, Optional[str], Optional[str]]:
        """(entry, quantized_mode or None, quantized_dir or None); checks requires_body."""
        e = self.get("encoders", name)
        if body is not None and e.requires_body and body not in e.requires_body:
            raise CatalogError(
                f"encoder {e.name!r} requires body model {', '.join(e.requires_body)}; selected {body!r}"
            )
        if e.type == "checkpoint":
            return e, None, None
        if e.type == "embedl_int8":
            return e, str(e.options.get("mode", "trt")), e.local_dir
        raise CatalogError(f"encoder {e.name!r} has unsupported type {e.type!r}")

    def resolve_detector(self, name: Optional[str]) -> Tuple[ModelEntry, Dict[str, Any]]:
        """(entry, kwargs for tools.build_detector.HumanDetector)."""
        e = self.get("detectors", name)
        kwargs: Dict[str, Any] = {"options": copy.deepcopy(e.options)}
        if e.type == "detectron2_vitdet":
            kwargs["path"] = e.local_dir
        elif e.type == "ultralytics":
            weights = e.options.get("weights")
            if weights:
                kwargs["weights"] = e.file_path(str(weights))
            elif e.files:
                kwargs["weights"] = e.file_path(e.files[0].path)
            else:
                raise CatalogError(f"detector {e.name!r}: ultralytics entries need a weights file")
            kwargs["options"].pop("weights", None)
        else:
            raise CatalogError(f"detector {e.name!r} has unsupported type {e.type!r}")
        return e, kwargs

    # ---- reporting ------------------------------------------------------
    def describe(self, loaded: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        loaded = loaded or {}
        out: Dict[str, Any] = {"schema": SCHEMA, "sources": [os.path.basename(s) for s in self.sources], "loaded": dict(loaded)}
        key_for = {"detectors": "detector", "encoders": "encoder", "body_models": "body_model"}
        for section in SECTIONS:
            out[section] = [
                e.describe(loaded=(loaded.get(key_for[section]) == e.name)) for e in self.section(section).values()
            ]
        return out


def default_user_paths(dirs: Optional[Dict[str, str]] = None) -> List[str]:
    explicit = os.environ.get("FBXIFY_MODELS_JSON", "").strip()
    if explicit:
        return [expand_path(explicit, dirs)]
    dirs = dirs or _default_dirs()
    return [os.path.join(dirs["CHECKPOINTS_DIR"], "models.json")]


_CACHED: Optional[ModelCatalog] = None


def load_catalog(refresh: bool = False, **kwargs) -> ModelCatalog:
    """Process-wide catalogue (re-read with refresh=True after editing models.json)."""
    global _CACHED
    if _CACHED is None or refresh or kwargs:
        cat = ModelCatalog.load(**kwargs)
        if not kwargs:
            _CACHED = cat
        return cat
    return _CACHED


def encoder_name_for_quantized_mode(mode: Optional[str]) -> str:
    """Legacy ``FBXIFY_QUANTIZED_BACKBONE`` / ``--quantized_backbone`` value -> encoder entry name."""
    key = None if mode is None else str(mode).strip().lower()
    if key in ("", "0", "off", "false", "no"):
        key = "none"
    if key not in QUANTIZED_MODE_TO_ENCODER:
        raise CatalogError(f"quantized backbone must be none|pt2|trt, got {mode!r}")
    return QUANTIZED_MODE_TO_ENCODER[key]


# --------------------------------------------------------------------------- fetching


@dataclass
class FetchResult:
    entry: str
    status: str          # ready | ready-unverified | skipped | missing | failed | disabled
    message: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"entry": self.entry, "status": self.status, "message": self.message}


def _download_url(url: str, dest: str, expected_size: Optional[int]) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".partial"
    req = Request(url, headers={"User-Agent": "fbxify-model-catalog/1"})
    with urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, _DOWNLOAD_CHUNK)
    if expected_size is not None and os.path.getsize(tmp) != expected_size:
        got = os.path.getsize(tmp)
        os.unlink(tmp)
        raise OSError(f"downloaded {got} bytes, expected {expected_size}")
    os.replace(tmp, dest)


def _verify_after_download(entry: ModelEntry, rel: str) -> None:
    spec = next(f for f in entry.files if f.path == rel)
    path = entry.file_path(rel)
    if spec.size is not None and os.path.getsize(path) != spec.size:
        raise OSError(f"{rel}: size {os.path.getsize(path)} != expected {spec.size}")
    if spec.sha256:
        got = _sha256_file(path)
        if got.lower() != spec.sha256.lower():
            os.replace(path, path + ".bad")
            raise OSError(f"{rel}: sha256 {got} != expected {spec.sha256} (moved aside as .bad)")


def fetch_entry(entry: ModelEntry, token: Optional[str] = None, log=print) -> FetchResult:
    """Make an entry's files present. Local entries are only checked. Downloads are verified."""
    if not entry.files:
        return FetchResult(entry.name, "ready", "no files needed")
    missing = entry.missing_files()
    bad = entry.size_mismatches()
    if not missing and not bad:
        return FetchResult(entry.name, entry.status(), "already present")
    label = f"{entry.section}.{entry.name}"
    if bad:
        log(f"model_catalog: {label}: size mismatch for {', '.join(bad)}; re-fetching", flush=True)
        for rel in bad:
            try:
                os.replace(entry.file_path(rel), entry.file_path(rel) + ".bad")
            except OSError:
                pass
        missing = sorted(set(missing) | set(bad))

    kind = entry.kind
    if kind == "local":
        return FetchResult(entry.name, "missing", f"local source; not downloaded. Missing: {', '.join(missing)}")
    if kind == "none":
        return FetchResult(entry.name, "ready", "")

    try:
        if kind == "url":
            url = str(entry.source["url"])
            for rel in missing:
                log(f"model_catalog: {label}: downloading {url}", flush=True)
                spec = next(f for f in entry.files if f.path == rel)
                _download_url(url, entry.file_path(rel), spec.size)
                _verify_after_download(entry, rel)
        elif kind == "huggingface":
            token = token or os.environ.get("HF_TOKEN") or None
            repo_id = str(entry.source["repo_id"])
            if not token:
                return FetchResult(
                    entry.name, "missing",
                    f"{repo_id} is a Hugging Face repo; set HF_TOKEN (with access accepted on the model page) "
                    f"to download. Missing: {', '.join(missing)}",
                )
            from huggingface_hub import hf_hub_download

            os.makedirs(entry.local_dir, exist_ok=True)
            for rel in missing:
                log(f"model_catalog: {label}: downloading {repo_id}/{rel} -> {entry.local_dir}", flush=True)
                hf_hub_download(
                    repo_id=repo_id,
                    filename=rel,
                    revision=entry.source.get("revision") or None,
                    local_dir=entry.local_dir,
                    token=token,
                )
                _verify_after_download(entry, rel)
    except Exception as exc:  # network, auth, checksum
        return FetchResult(entry.name, "failed", f"{type(exc).__name__}: {exc}")

    still = entry.missing_files()
    if still:
        return FetchResult(entry.name, "missing", "still missing after fetch: " + ", ".join(still))
    return FetchResult(entry.name, entry.status(), "fetched")


def fetch_enabled(catalog: ModelCatalog, sections: Optional[Iterable[str]] = None, log=print) -> List[FetchResult]:
    """Start-up hook: fetch every enabled entry (and nothing else)."""
    results = []
    for e in catalog.enabled_entries(sections):
        r = fetch_entry(e, log=log)
        if r.message:
            log(f"model_catalog: {e.section}.{e.name}: {r.status} - {r.message}", flush=True)
        results.append(r)
    return results


# --------------------------------------------------------------------------- CLI


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fbxify.model_catalog", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd, help_text in (("list", "show catalogue entries and status"),
                           ("fetch", "download enabled entries that are missing"),
                           ("verify", "check files (sizes; --deep adds sha256)")):
        p = sub.add_parser(cmd, help=help_text)
        p.add_argument("--all", action="store_true", help="include disabled entries")
        p.add_argument("--section", choices=SECTIONS, action="append")
        p.add_argument("--name", action="append", help="restrict to entry name(s)")
        p.add_argument("--json", action="store_true")
        if cmd == "verify":
            p.add_argument("--deep", action="store_true", help="hash files (slow for multi-GB checkpoints)")
    args = parser.parse_args(argv)

    try:
        catalog = ModelCatalog.load()
    except (CatalogError, OSError, json.JSONDecodeError) as exc:
        print(f"model_catalog: {exc}", file=sys.stderr)
        return 2

    entries = catalog.all_entries(args.section) if args.all else catalog.enabled_entries(args.section)
    if args.name:
        wanted = set(args.name)
        entries = [e for e in catalog.all_entries(args.section) if e.name in wanted]

    rows: List[Dict[str, Any]] = []
    exit_code = 0
    for e in entries:
        if args.cmd == "list":
            rows.append({"section": e.section, **e.describe()})
        elif args.cmd == "fetch":
            r = fetch_entry(e)
            rows.append({"section": e.section, "name": e.name, **r.as_dict()})
            if r.status in ("failed", "missing") and e.default:
                exit_code = 1
        else:
            ok, msg = e.verify(deep=args.deep)
            rows.append({"section": e.section, "name": e.name, "ok": ok, "message": msg})
            if not ok and e.default:
                exit_code = 1

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            tag = r.get("status") if "status" in r else ("ok" if r.get("ok") else "FAIL")
            extra = r.get("message") or r.get("display_name") or ""
            flags = ""
            if args.cmd == "list":
                flags = ("*" if r.get("default") else " ") + ("+" if r.get("enabled") else "-")
            print(f"{r['section']:<12} {r['name']:<18} {flags:<3} {tag:<16} {extra}")
        if args.cmd == "list":
            print("\n(* default, + enabled, - disabled; sources: " + ", ".join(os.path.basename(s) for s in catalog.sources) + ")")
    return exit_code


if __name__ == "__main__":
    sys.exit(_cli())
