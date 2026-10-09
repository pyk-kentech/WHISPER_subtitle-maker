from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import get_app_data_dir
from .credential_store import CredentialStoreError


# 리눅스용 저장소: 앱 데이터 폴더(권한 700) 안의 secrets.json(권한 600)에 저장한다.
# 데스크톱마다 키링(Secret Service) 지원이 제각각이라 어디서나 동작하는 파일 방식을 쓴다.
_LOCK = threading.Lock()


def _secrets_path() -> Path:
    return get_app_data_dir() / "secrets.json"


def _read_all() -> dict[str, str]:
    path = _secrets_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CredentialStoreError(f"저장된 키 파일을 읽지 못했습니다: {path} ({exc})") from exc
    if not isinstance(data, dict):
        return {}
    return {str(name): str(value) for name, value in data.items()}


def _write_all(data: dict[str, str]) -> None:
    path = _secrets_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        partial_path = path.with_name(path.name + ".part")
        # 처음 만들 때부터 600으로 만들어 잠깐이라도 다른 사용자가 읽을 수 있는 순간이 없게 한다.
        fd = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.chmod(partial_path, 0o600)
        os.replace(partial_path, path)
    except OSError as exc:
        raise CredentialStoreError(f"키 파일을 저장하지 못했습니다: {path} ({exc})") from exc


def save_secret(target_name: str, secret: str, user_name: str = "DongeumSubMaker") -> None:
    with _LOCK:
        data = _read_all()
        data[target_name] = secret
        _write_all(data)


def load_secret(target_name: str) -> str:
    with _LOCK:
        return _read_all().get(target_name, "")


def delete_secret(target_name: str) -> None:
    with _LOCK:
        data = _read_all()
        if data.pop(target_name, None) is not None:
            _write_all(data)
