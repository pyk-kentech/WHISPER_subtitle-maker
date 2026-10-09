from __future__ import annotations

import ctypes
from ctypes import wintypes

from .credential_store import CredentialStoreError


CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
CRED_MAX_CREDENTIAL_BLOB_SIZE = 5 * 512
ERROR_NOT_FOUND = 1168


class FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    ]


class CREDENTIAL_ATTRIBUTEW(ctypes.Structure):
    _fields_ = [
        ("Keyword", wintypes.LPWSTR),
        ("Flags", wintypes.DWORD),
        ("ValueSize", wintypes.DWORD),
        ("Value", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.POINTER(CREDENTIAL_ATTRIBUTEW)),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


PCREDENTIALW = ctypes.POINTER(CREDENTIALW)

# use_last_error=True: 실패 직후의 Windows 오류 코드를 ctypes.get_last_error()로 정확히 읽는다.
_advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)

_CredWriteW = _advapi32.CredWriteW
_CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
_CredWriteW.restype = wintypes.BOOL

_CredReadW = _advapi32.CredReadW
_CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(PCREDENTIALW)]
_CredReadW.restype = wintypes.BOOL

_CredDeleteW = _advapi32.CredDeleteW
_CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
_CredDeleteW.restype = wintypes.BOOL

_CredFree = _advapi32.CredFree
_CredFree.argtypes = [ctypes.c_void_p]
_CredFree.restype = None



def save_secret(target_name: str, secret: str, user_name: str = "DongeumSubMaker") -> None:
    secret_bytes = secret.encode("utf-16-le")
    if len(secret_bytes) > CRED_MAX_CREDENTIAL_BLOB_SIZE:
        raise CredentialStoreError(
            f"저장할 값이 Windows 자격 증명 한도({CRED_MAX_CREDENTIAL_BLOB_SIZE}바이트)를 넘습니다."
        )
    blob = (ctypes.c_ubyte * len(secret_bytes)).from_buffer_copy(secret_bytes) if secret_bytes else None

    credential = CREDENTIALW()
    credential.Flags = 0
    credential.Type = CRED_TYPE_GENERIC
    credential.TargetName = target_name
    credential.Comment = None
    credential.CredentialBlobSize = len(secret_bytes)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte)) if blob is not None else None
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE
    credential.AttributeCount = 0
    credential.Attributes = None
    credential.TargetAlias = None
    credential.UserName = user_name

    if not _CredWriteW(ctypes.byref(credential), 0):
        raise CredentialStoreError(f"CredWriteW failed with error {ctypes.get_last_error()}")


def load_secret(target_name: str) -> str:
    credential_ptr = PCREDENTIALW()
    if not _CredReadW(target_name, CRED_TYPE_GENERIC, 0, ctypes.byref(credential_ptr)):
        error = ctypes.get_last_error()
        if error == ERROR_NOT_FOUND:
            return ""
        raise CredentialStoreError(f"CredReadW failed with error {error}")

    try:
        credential = credential_ptr.contents
        size = int(credential.CredentialBlobSize)
        if size <= 0 or not credential.CredentialBlob:
            return ""
        secret_bytes = ctypes.string_at(credential.CredentialBlob, size)
        return secret_bytes.decode("utf-16-le")
    finally:
        _CredFree(credential_ptr)


def delete_secret(target_name: str) -> None:
    if not _CredDeleteW(target_name, CRED_TYPE_GENERIC, 0):
        error = ctypes.get_last_error()
        if error != ERROR_NOT_FOUND:
            raise CredentialStoreError(f"CredDeleteW failed with error {error}")
