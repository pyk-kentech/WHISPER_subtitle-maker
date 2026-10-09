"""리눅스용 파일 자격 증명 저장소(_credential_store_file) 테스트. 임시 앱 데이터 폴더만 쓴다."""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest

# 앱과 같은 순서로 불러온다(리눅스에서는 credential_store가 파일 백엔드를 불러오므로 반대 순서면 순환 import).
from app.credential_store import CredentialStoreError
import app._credential_store_file as store  # noqa: E402


def test_secrets_path_is_inside_temp_app_data(isolated_app_data):
    assert store._secrets_path() == isolated_app_data / "secrets.json"


def test_roundtrip_save_load_delete(isolated_app_data):
    assert store.load_secret("A") == ""  # 파일이 없어도 빈 문자열
    store.save_secret("A", "alpha")
    store.save_secret("B", "베타 🔑")
    assert store.load_secret("A") == "alpha"
    assert store.load_secret("B") == "베타 🔑"
    assert store.load_secret("missing") == ""

    store.save_secret("A", "alpha-2")
    assert store.load_secret("A") == "alpha-2"

    store.delete_secret("A")
    assert store.load_secret("A") == ""
    assert store.load_secret("B") == "베타 🔑"
    data = json.loads((isolated_app_data / "secrets.json").read_text(encoding="utf-8"))
    assert data == {"B": "베타 🔑"}
    assert not (isolated_app_data / "secrets.json.part").exists()


def test_delete_missing_does_not_create_file(isolated_app_data):
    store.delete_secret("nothing")
    assert not (isolated_app_data / "secrets.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows는 POSIX 권한 비트를 지원하지 않는다")
def test_file_permissions_are_private(isolated_app_data):
    store.save_secret("A", "alpha")
    path = isolated_app_data / "secrets.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700


def test_corrupt_file_raises_store_error(isolated_app_data):
    isolated_app_data.mkdir(parents=True, exist_ok=True)
    (isolated_app_data / "secrets.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CredentialStoreError):
        store.load_secret("A")


def test_non_dict_json_is_treated_as_empty(isolated_app_data):
    isolated_app_data.mkdir(parents=True, exist_ok=True)
    (isolated_app_data / "secrets.json").write_text("[1, 2]", encoding="utf-8")
    assert store.load_secret("A") == ""
    store.save_secret("A", "x")
    assert store.load_secret("A") == "x"
