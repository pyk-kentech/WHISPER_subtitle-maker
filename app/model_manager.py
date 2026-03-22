from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import DryRunFileInfo
from tqdm.auto import tqdm

from .config import DEFAULT_MODEL_KEY, MODEL_PRESETS, MODEL_REQUIRED_FILES, get_hf_cache_dir, get_model_cache_dir_for


ProgressCallback = Callable[[int, int, str, int, int], None]
StatusCallback = Callable[[str], None]


@dataclass(slots=True)
class DownloadPlan:
    files: list[DryRunFileInfo]
    total_bytes: int


class DownloadReporter:
    def __init__(self, total_bytes: int, progress_callback: ProgressCallback) -> None:
        self.total_bytes = total_bytes
        self.progress_callback = progress_callback
        self.downloaded_bytes = 0
        self.current_file = ""
        self.current_file_total = 0
        self.current_file_bytes = 0

    def start_bar(self, desc: str, total: int | None, initial: int | None) -> None:
        self.current_file = Path(desc).name if desc else "model"
        self.current_file_total = int(total or 0)
        self.current_file_bytes = int(initial or 0)
        self.progress_callback(
            self.downloaded_bytes,
            self.total_bytes,
            self.current_file,
            self.current_file_bytes,
            self.current_file_total,
        )

    def update(self, amount: int) -> None:
        if amount <= 0:
            return
        self.downloaded_bytes += amount
        self.current_file_bytes += amount
        self.progress_callback(
            self.downloaded_bytes,
            self.total_bytes,
            self.current_file,
            self.current_file_bytes,
            self.current_file_total,
        )

    def close_bar(self) -> None:
        self.progress_callback(
            self.downloaded_bytes,
            self.total_bytes,
            self.current_file,
            self.current_file_bytes,
            self.current_file_total,
        )


class HubProgressTqdm(tqdm):
    reporter: DownloadReporter | None = None

    def __init__(self, *args, **kwargs) -> None:
        desc = str(kwargs.get("desc") or "")
        total = kwargs.get("total")
        initial = kwargs.get("initial")
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
        if HubProgressTqdm.reporter is not None:
            HubProgressTqdm.reporter.start_bar(desc, total, initial)

    def update(self, n: int = 1) -> bool | None:
        if HubProgressTqdm.reporter is not None:
            HubProgressTqdm.reporter.update(int(n))
        return super().update(n)

    def close(self) -> None:
        if HubProgressTqdm.reporter is not None:
            HubProgressTqdm.reporter.close_bar()
        super().close()


def get_model_repo_id(model_key: str = DEFAULT_MODEL_KEY) -> str:
    preset = MODEL_PRESETS.get(model_key, MODEL_PRESETS[DEFAULT_MODEL_KEY])
    return str(preset["repo_id"])


def get_model_label(model_key: str = DEFAULT_MODEL_KEY) -> str:
    preset = MODEL_PRESETS.get(model_key, MODEL_PRESETS[DEFAULT_MODEL_KEY])
    return str(preset["label"])


def get_model_dir(model_key: str = DEFAULT_MODEL_KEY) -> Path:
    return get_model_cache_dir_for(model_key)


def is_model_ready(model_dir: Path | None = None, model_key: str = DEFAULT_MODEL_KEY) -> bool:
    target_dir = model_dir or get_model_dir(model_key)
    return all((target_dir / filename).is_file() for filename in MODEL_REQUIRED_FILES)


def get_download_plan(model_key: str = DEFAULT_MODEL_KEY) -> DownloadPlan:
    model_dir = get_model_dir(model_key)
    cache_dir = get_hf_cache_dir()
    repo_id = get_model_repo_id(model_key)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files: list[DryRunFileInfo] = []
    total_bytes = 0
    for filename in MODEL_REQUIRED_FILES:
        info = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            cache_dir=cache_dir,
            dry_run=True,
        )
        files.append(info)
        if info.will_download:
            total_bytes += info.file_size
    return DownloadPlan(files=files, total_bytes=total_bytes)


def download_model(model_key: str, progress_callback: ProgressCallback, status_callback: StatusCallback) -> Path:
    plan = get_download_plan(model_key)
    model_dir = get_model_dir(model_key)
    cache_dir = get_hf_cache_dir()
    repo_id = get_model_repo_id(model_key)

    if plan.total_bytes == 0 and is_model_ready(model_dir):
        progress_callback(1, 1, "", 1, 1)
        status_callback("모델 캐시가 이미 준비되어 있습니다.")
        return model_dir

    reporter = DownloadReporter(plan.total_bytes, progress_callback)
    HubProgressTqdm.reporter = reporter

    try:
        for info in plan.files:
            if info.will_download:
                status_callback(f"모델 다운로드 중 {info.filename}")
            else:
                status_callback(f"모델 캐시 확인 중 {info.filename}")

            hf_hub_download(
                repo_id=repo_id,
                filename=info.filename,
                local_dir=model_dir,
                cache_dir=cache_dir,
                tqdm_class=HubProgressTqdm,
            )
    finally:
        HubProgressTqdm.reporter = None

    if not is_model_ready(model_dir):
        raise RuntimeError("모델 다운로드 후에 필수 파일이 누락되어 있습니다.")

    status_callback("모델 다운로드 완료")
    return model_dir
