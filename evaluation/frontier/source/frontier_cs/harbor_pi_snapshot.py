"""Archive the host-side Pi runtime inputs used by a Harbor run."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from datetime import datetime

try:  # ``datetime.UTC`` was added in Python 3.11; the suite supports 3.10.
    from datetime import UTC
except ImportError:  # pragma: no cover - exercised only on Python 3.10
    from datetime import timezone

    UTC = timezone.utc
from hashlib import sha256
from pathlib import Path
from typing import Any

PI_PACKAGE = "@earendil-works/pi-coding-agent"
PI_VERSION = "0.84.3"

_CORE_CONFIG_FILES = {
    "PI_AUTH_JSON_PATH": "auth.json",
    "PI_MODELS_JSON_PATH": "models.json",
    "PI_SETTINGS_JSON_PATH": "settings.json",
}
_OPTIONAL_CONFIG_FILES = {
    "PI_SYSTEM_PROMPT_PATH": "SYSTEM.md",
    "PI_EXTENSION_PATH": "temperature-extension.js",
    "PI_SYSTEM_PROMPT_RECORDER_PATH": "system-prompt-recorder.js",
}
_FULL_RESOURCE_DIRS = (
    "extensions",
    "skills",
    "prompts",
    "themes",
    "npm",
    "git",
    "local",
)
_FULL_RESOURCE_FILES = (
    "AGENTS.md",
    "models-store.json",
    "trust.json",
    "package.json",
    "package-lock.json",
)
_RESOURCE_IGNORE = shutil.ignore_patterns(".git", ".npmrc", ".env", ".env.*")
_FULL_SNAPSHOT_ENV = "PI_RUNTIME_SNAPSHOT_FULL"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


def _resolve_path(value: str, home: Path) -> Path:
    if value == "~":
        return home
    if value.startswith("~/"):
        return home / value[2:]
    return Path(value).expanduser().resolve()


def _redact_all(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_all(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_all(item) for item in value]
    return "<redacted>"


def _write_redacted_auth(source: Path, destination: Path) -> None:
    value = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(
        json.dumps(_redact_all(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    destination.chmod(0o600)


def _full_snapshot_enabled(values: Mapping[str, str]) -> bool:
    raw = values.get(_FULL_SNAPSHOT_ENV, "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{_FULL_SNAPSHOT_ENV} must be one of: 0, 1, false, true, no, yes, off, on"
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _configured_package_sources(settings_path: Path) -> list[str]:
    settings = _read_json_object(settings_path)
    if settings is None:
        return []
    packages = settings.get("packages")
    if not isinstance(packages, list):
        return []

    sources: list[str] = []
    for package in packages:
        if isinstance(package, str):
            source = package
        elif isinstance(package, dict):
            if package.get("autoload") is False:
                continue
            source = package.get("source")
            if not isinstance(source, str):
                continue
        else:
            continue
        source = source.strip()
        if source:
            sources.append(source)
    return sources


def _parse_npm_spec(source: str) -> tuple[str, str | None]:
    spec = source.removeprefix("npm:").strip()
    separator = spec.rfind("@")
    if separator > spec.rfind("/"):
        return spec[:separator], spec[separator + 1 :] or None
    return spec, None


def _split_git_ref(path: str) -> tuple[str, str | None]:
    separator = path.rfind("@")
    if separator <= path.rfind("/"):
        return path, None
    return path[:separator], path[separator + 1 :] or None


def _git_package_location(
    source: str, agent_dir: Path
) -> tuple[Path | None, str | None]:
    raw = source.removeprefix("git:").strip()
    host: str | None = None
    path_with_ref: str | None = None

    scp_match = re.fullmatch(r"git@([^:]+):(.+)", raw)
    if scp_match:
        host, path_with_ref = scp_match.groups()
    elif "://" in raw:
        protocol_match = re.fullmatch(
            r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^/@]+@)?([^/]+)/(.+)", raw
        )
        if protocol_match:
            host, path_with_ref = protocol_match.groups()
    elif "/" in raw:
        host, path_with_ref = raw.split("/", 1)

    if (
        not host
        or host in {".", ".."}
        or "/" in host
        or "\\" in host
        or not path_with_ref
    ):
        return None, None
    package_path, ref = _split_git_ref(path_with_ref)
    package_path = package_path.removesuffix(".git").strip("/")
    if not package_path or "\\" in package_path or ".." in Path(package_path).parts:
        return None, ref
    return agent_dir / "git" / host / package_path, ref


def _read_git_head(repository: Path) -> str | None:
    git_dir = repository / ".git"
    if git_dir.is_file():
        value = git_dir.read_text(encoding="utf-8", errors="replace").strip()
        if not value.startswith("gitdir:"):
            return None
        git_dir = (repository / value.removeprefix("gitdir:").strip()).resolve()
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8", errors="replace").strip()
    if not head.startswith("ref:"):
        return head or None
    ref = head.removeprefix("ref:").strip()
    ref_path = git_dir / ref
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8", errors="replace").strip() or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith(("#", "^")):
                continue
            commit, _, packed_ref = line.partition(" ")
            if packed_ref == ref:
                return commit or None
    return None


def _copy_lockfile(
    source: Path,
    target: Path,
    destination: Path,
    entries: list[dict[str, str]],
) -> dict[str, str] | None:
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    shutil.copy2(source, target)
    checksum = _sha256_file(target)
    relative_target = str(target.relative_to(destination))
    entries.append(
        {
            "kind": "package-lockfile",
            "source": str(source),
            "destination": relative_target,
            "sha256": checksum,
            "status": "copied",
        }
    )
    return {"path": relative_target, "sha256": checksum}


def _archive_remote_package_metadata(
    settings_path: Path,
    agent_dir: Path,
    resources_dir: Path,
    destination: Path,
    entries: list[dict[str, str]],
) -> None:
    sources = _configured_package_sources(settings_path)
    remote_sources = [
        source
        for source in sources
        if source.startswith(("npm:", "git:"))
        or re.match(r"^(?:https?|ssh|git)://", source, re.IGNORECASE)
    ]
    if not remote_sources:
        return

    metadata_root = resources_dir / "package-metadata"
    npm_lock_source = agent_dir / "npm" / "package-lock.json"
    npm_lock_data = _read_json_object(npm_lock_source) or {}
    npm_packages = npm_lock_data.get("packages")
    if not isinstance(npm_packages, dict):
        npm_packages = {}
    npm_lock: dict[str, str] | None = None
    records: list[dict[str, Any]] = []

    for index, source in enumerate(remote_sources):
        if source.startswith("npm:"):
            name, requested_version = _parse_npm_spec(source)
            installed = npm_packages.get(f"node_modules/{name}")
            installed = installed if isinstance(installed, dict) else {}
            version = installed.get("version")
            if not isinstance(version, str):
                package_json = _read_json_object(
                    agent_dir / "npm" / "node_modules" / name / "package.json"
                )
                version = package_json.get("version") if package_json else None
            if npm_lock is None:
                npm_lock = _copy_lockfile(
                    npm_lock_source,
                    metadata_root / "npm" / "package-lock.json",
                    destination,
                    entries,
                )
            record: dict[str, Any] = {
                "source": source,
                "type": "npm",
                "name": name,
            }
            if isinstance(version, str) and version:
                record["version"] = version
            elif requested_version:
                record["version"] = requested_version
            integrity = installed.get("integrity")
            if isinstance(integrity, str) and integrity:
                record["checksum"] = {
                    "algorithm": "npm-integrity",
                    "value": integrity,
                }
            if npm_lock is not None:
                record["lockfile"] = npm_lock
            records.append(record)
            continue

        repository, requested_ref = _git_package_location(source, agent_dir)
        record = {"source": source, "type": "git"}
        if requested_ref:
            record["version"] = requested_ref
        if repository is not None and repository.is_dir():
            package_json = _read_json_object(repository / "package.json")
            package_version = package_json.get("version") if package_json else None
            if isinstance(package_version, str) and package_version:
                record["version"] = package_version
            commit = _read_git_head(repository)
            if commit:
                record["checksum"] = {
                    "algorithm": "git-commit",
                    "value": commit,
                }
            lockfile = _copy_lockfile(
                repository / "package-lock.json",
                metadata_root
                / "git"
                / f"{index:02d}-{repository.name}-package-lock.json",
                destination,
                entries,
            )
            if lockfile is not None:
                record["lockfile"] = lockfile
        records.append(record)

    metadata_path = metadata_root / "packages.json"
    metadata_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {"schema_version": 1, "packages": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path.chmod(0o600)
    entries.append(
        {
            "kind": "package-metadata",
            "source": str(settings_path),
            "destination": str(metadata_path.relative_to(destination)),
            "status": "written",
        }
    )


def snapshot_pi_runtime(
    destination: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Copy Pi configuration and resources into one immutable run snapshot.

    Authentication values are never copied verbatim. The default lean mode
    archives only explicitly selected resources; full mode additionally
    preserves the complete host-side Pi resource trees.
    """

    values = dict(os.environ if env is None else env)
    full_snapshot = _full_snapshot_enabled(values)
    home = Path(values.get("HOME") or Path.home()).expanduser().resolve()
    configured_agent_dir = values.get("PI_CODING_AGENT_DIR")
    agent_dir = (
        _resolve_path(configured_agent_dir, home)
        if configured_agent_dir
        else home / ".pi" / "agent"
    )

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    config_dir = destination / "config"
    resources_dir = destination / "resources"
    config_dir.mkdir(mode=0o700)
    resources_dir.mkdir(mode=0o700)

    entries: list[dict[str, str]] = []
    errors: list[str] = []

    config_files = dict(_CORE_CONFIG_FILES)
    config_files.update(
        {
            env_name: filename
            for env_name, filename in _OPTIONAL_CONFIG_FILES.items()
            if full_snapshot or values.get(env_name)
        }
    )
    settings_source = _resolve_path(
        values.get("PI_SETTINGS_JSON_PATH", str(agent_dir / "settings.json")), home
    )

    for env_name, filename in config_files.items():
        configured_path = values.get(env_name)
        source = (
            _resolve_path(configured_path, home)
            if configured_path
            else agent_dir / filename
        )
        entry = {
            "kind": "config",
            "environment": env_name,
            "source": str(source),
        }
        if not source.is_file():
            entry["status"] = "missing"
            entries.append(entry)
            if configured_path:
                errors.append(f"{env_name} points to a missing file: {source}")
            continue

        archived_name = "auth.redacted.json" if filename == "auth.json" else filename
        target = config_dir / archived_name
        try:
            if filename == "auth.json":
                _write_redacted_auth(source, target)
                entry["sanitization"] = "all-values"
            else:
                shutil.copy2(source, target)
                target.chmod(0o600)
            entry.update(
                {
                    "destination": str(target.relative_to(destination)),
                    "status": "copied",
                }
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            errors.append(f"failed to archive {source}: {exc}")
        entries.append(entry)

    if full_snapshot:
        for name in (*_FULL_RESOURCE_DIRS, *_FULL_RESOURCE_FILES):
            source = agent_dir / name
            entry = {"kind": "resource", "source": str(source)}
            if not source.exists():
                entry["status"] = "missing"
                entries.append(entry)
                continue
            target = resources_dir / name
            try:
                if source.is_dir():
                    shutil.copytree(
                        source, target, symlinks=True, ignore=_RESOURCE_IGNORE
                    )
                elif source.is_file():
                    shutil.copy2(source, target)
                else:
                    raise OSError("unsupported resource type")
                entry.update(
                    {
                        "destination": str(target.relative_to(destination)),
                        "status": "copied",
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
                errors.append(f"failed to archive {source}: {exc}")
            entries.append(entry)

    _archive_remote_package_metadata(
        settings_source, agent_dir, resources_dir, destination, entries
    )

    configured_extensions = values.get("PI_EXTENSIONS_DIR")
    if configured_extensions:
        source = _resolve_path(configured_extensions, home)
        entry = {
            "kind": "resource",
            "environment": "PI_EXTENSIONS_DIR",
            "source": str(source),
        }
        if source.is_dir():
            target = resources_dir / "mounted-extensions"
            try:
                shutil.copytree(source, target, symlinks=True, ignore=_RESOURCE_IGNORE)
                entry.update(
                    {
                        "destination": str(target.relative_to(destination)),
                        "status": "copied",
                    }
                )
            except OSError as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
                errors.append(f"failed to archive {source}: {exc}")
        else:
            entry["status"] = "missing"
            errors.append(f"PI_EXTENSIONS_DIR points to a missing directory: {source}")
        entries.append(entry)

    configured_skill = values.get("PI_SKILL_DIR")
    if configured_skill:
        source = _resolve_path(configured_skill, home)
        entry = {
            "kind": "resource",
            "environment": "PI_SKILL_DIR",
            "source": str(source),
        }
        if source.is_dir():
            target = resources_dir / "selected-skill"
            try:
                shutil.copytree(source, target, symlinks=True, ignore=_RESOURCE_IGNORE)
                entry.update(
                    {
                        "destination": str(target.relative_to(destination)),
                        "status": "copied",
                    }
                )
            except OSError as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
                errors.append(f"failed to archive {source}: {exc}")
        else:
            entry["status"] = "missing"
            errors.append(f"PI_SKILL_DIR points to a missing directory: {source}")
        entries.append(entry)

    configured_packages_json = values.get("PI_LOCAL_PACKAGES_JSON")
    if configured_packages_json:
        try:
            configured_packages = json.loads(configured_packages_json)
            if not isinstance(configured_packages, list) or not all(
                isinstance(item, str) for item in configured_packages
            ):
                raise ValueError("expected a JSON array of paths")
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid PI_LOCAL_PACKAGES_JSON: {exc}")
            configured_packages = []

        for index, configured_package in enumerate(configured_packages):
            source = _resolve_path(configured_package, home)
            entry = {
                "kind": "resource",
                "environment": "PI_LOCAL_PACKAGES_JSON",
                "source": str(source),
            }
            if not source.is_dir():
                entry["status"] = "missing"
                errors.append(f"local Pi package directory is missing: {source}")
                entries.append(entry)
                continue

            target = (
                resources_dir
                / "selected-local-packages"
                / (f"{index:02d}-{source.name}")
            )
            try:
                target.parent.mkdir(mode=0o700, exist_ok=True)
                shutil.copytree(source, target, symlinks=True, ignore=_RESOURCE_IGNORE)
                entry.update(
                    {
                        "destination": str(target.relative_to(destination)),
                        "status": "copied",
                    }
                )
            except OSError as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
                errors.append(f"failed to archive {source}: {exc}")
            entries.append(entry)

    if full_snapshot:
        global_skills = home / ".agents" / "skills"
        global_entry = {"kind": "resource", "source": str(global_skills)}
        if global_skills.is_dir():
            target = resources_dir / "global-agents-skills"
            try:
                shutil.copytree(
                    global_skills, target, symlinks=True, ignore=_RESOURCE_IGNORE
                )
                global_entry.update(
                    {
                        "destination": str(target.relative_to(destination)),
                        "status": "copied",
                    }
                )
            except OSError as exc:
                global_entry["status"] = "error"
                global_entry["error"] = str(exc)
                errors.append(f"failed to archive {global_skills}: {exc}")
        else:
            global_entry["status"] = "missing"
        entries.append(global_entry)

    manifest = {
        "schema_version": 2,
        "captured_at": datetime.now(UTC).isoformat(),
        "pi": {"package": PI_PACKAGE, "version": PI_VERSION},
        "snapshot_mode": "full" if full_snapshot else "lean",
        "source_agent_dir": str(agent_dir),
        "security": {
            "auth": "all values redacted",
            "other_files": "copied verbatim",
        },
        "entries": entries,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)

    if errors:
        raise RuntimeError("; ".join(errors))
    return destination
