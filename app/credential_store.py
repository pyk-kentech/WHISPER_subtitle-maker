from __future__ import annotations

import sys


class CredentialStoreError(RuntimeError):
    pass


# Windows는 자격 증명 관리자, 그 밖(리눅스)은 본인만 읽을 수 있는 파일에 저장한다.
if sys.platform == "win32":
    from ._credential_store_windows import delete_secret, load_secret, save_secret
else:
    from ._credential_store_file import delete_secret, load_secret, save_secret

__all__ = ["CredentialStoreError", "delete_secret", "load_secret", "save_secret"]
