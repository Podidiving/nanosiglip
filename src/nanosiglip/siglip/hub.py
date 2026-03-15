from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF_BASE_URL = "https://huggingface.co"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nanosiglip"


class HuggingFaceDownloadError(RuntimeError):
    pass


def _extract_repo_id(identifier: str) -> str:
    if identifier.startswith("https://") or identifier.startswith("http://"):
        parsed = urllib.parse.urlparse(identifier)
        if parsed.netloc != "huggingface.co":
            raise ValueError(f"Unsupported host in URL: {identifier}")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"Could not parse repo id from URL: {identifier}")
        return f"{parts[0]}/{parts[1]}"
    return identifier


def _repo_cache_dir(repo_id: str, revision: str, cache_dir: Path | None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    safe_repo = repo_id.replace("/", "--")
    safe_revision = revision.replace("/", "--")
    return root / safe_repo / safe_revision


def _download_file(
    repo_id: str, revision: str, filename: str, target_dir: Path
) -> Path | None:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    encoded_filename = "/".join(
        urllib.parse.quote(part) for part in filename.split("/")
    )
    url = f"{HF_BASE_URL}/{repo_id}/resolve/{revision}/{encoded_filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "nanosiglip/0.1"})

    try:
        with urllib.request.urlopen(req) as response:
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with tmp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
            tmp_path.replace(out_path)
            return out_path
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise HuggingFaceDownloadError(
            f"Failed to download '{filename}' from '{repo_id}' at revision '{revision}': HTTP {err.code}"
        ) from err
    except urllib.error.URLError as err:
        raise HuggingFaceDownloadError(
            f"Network error while downloading '{filename}' from '{repo_id}': {err}"
        ) from err


def resolve_pretrained_path(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    include_preprocessor: bool = False,
    include_tokenizer: bool = False,
) -> Path:
    candidate = Path(pretrained_model_name_or_path)
    if candidate.exists():
        return candidate.resolve()

    if local_files_only:
        raise FileNotFoundError(
            f"Local path '{pretrained_model_name_or_path}' not found and local_files_only=True"
        )

    repo_id = _extract_repo_id(str(pretrained_model_name_or_path))
    cache_path = _repo_cache_dir(
        repo_id, revision, Path(cache_dir) if cache_dir is not None else None
    )

    config_path = _download_file(repo_id, revision, "config.json", cache_path)
    if config_path is None:
        raise HuggingFaceDownloadError(f"Missing config.json for repo '{repo_id}'")

    model_file = _download_file(repo_id, revision, "model.safetensors", cache_path)
    if model_file is None:
        index_path = _download_file(
            repo_id, revision, "model.safetensors.index.json", cache_path
        )
        if index_path is None:
            raise HuggingFaceDownloadError(
                f"Could not find model.safetensors or model.safetensors.index.json for '{repo_id}'"
            )

        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_filenames = sorted(set(weight_map.values()))
        if not shard_filenames:
            raise HuggingFaceDownloadError(
                f"Index file for '{repo_id}' has an empty weight_map"
            )

        for shard in shard_filenames:
            shard_path = _download_file(repo_id, revision, shard, cache_path)
            if shard_path is None:
                raise HuggingFaceDownloadError(
                    f"Missing shard '{shard}' for repo '{repo_id}'"
                )

    if include_preprocessor:
        _download_file(repo_id, revision, "preprocessor_config.json", cache_path)
    if include_tokenizer:
        _download_file(repo_id, revision, "spiece.model", cache_path)
        _download_file(repo_id, revision, "tokenizer_config.json", cache_path)
        _download_file(repo_id, revision, "special_tokens_map.json", cache_path)
        _download_file(repo_id, revision, "tokenizer.json", cache_path)

    return cache_path.resolve()
