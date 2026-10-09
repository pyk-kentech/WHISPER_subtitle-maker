"""Runpod Pod 안에서 도는 원격 음성 인식(+선택: LLM 번역 서버) 워커. 표준 라이브러리만으로 시작한다.

앱(remote_runpod.py)이 Pod를 만들 때 이 파일을 압축해 환경 변수(DSM_BOOT)로 넘기고, Pod가 뜨면
앱의 음성 인식 코드(app 패키지)를 /bundle로 올린다. 그래서 원격에서도 이 PC와 똑같은 코드(WhisperSeg VAD,
구간별 인식, 모델 프리셋)로 인식한다. 후처리와 번역 파일 저장은 앱(이 PC)에서 한다.

보안·비용 장치
- 모든 요청은 DSM_TOKEN(무작위 토큰)이 있어야 한다(X-DSM-Token 또는 Authorization: Bearer).
- 올린 음성은 인식이 끝나면(성공·실패 모두) 바로 지운다. DELETE /jobs/<id>로 결과까지 지운다.
- 요청이 DSM_IDLE_MINUTES 동안 없거나 Pod가 DSM_MAX_HOURS를 넘기면 Pod가 스스로 삭제된다
  (runpodctl remove pod, 안 되면 stop). 앱이 꺼지거나 인터넷이 끊겨도 요금이 계속 나가지 않게 하려는 것이다.

엔드포인트
  GET  /health                      상태(stage, llm, gpu, jobs)
  PUT  /bundle                      앱 코드 zip
  PUT  /jobs/<id>/audio?offset=N    음성 조각 올리기(이어 붙임)
  POST /jobs/<id>/start             인식 시작(JSON: model_key, language, vad, compute_type)
  GET  /jobs/<id>                   진행률·결과
  DELETE /jobs/<id>                 음성·결과 삭제
  POST /shutdown                    지금 Pod 삭제
  *    /v1/...                      llama-server(OpenAI 호환)로 전달(LLM을 켠 경우)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VERSION = "1"
TOKEN = os.environ.get("DSM_TOKEN", "")
PORT = int(os.environ.get("DSM_PORT", "8000"))
WORKDIR = Path(os.environ.get("DSM_WORKDIR", "/dsm"))
IDLE_MINUTES = float(os.environ.get("DSM_IDLE_MINUTES", "15"))
MAX_HOURS = float(os.environ.get("DSM_MAX_HOURS", "6"))
SELF_DESTRUCT = os.environ.get("DSM_SELF_DESTRUCT", "1") == "1"
INSTALL_DEPS = os.environ.get("DSM_INSTALL_DEPS", "1") == "1"
PIP = os.environ.get("DSM_PIP", f"{sys.executable} -m pip")
DEPS = os.environ.get(
    "DSM_DEPS",
    "faster-whisper>=1.2.1,<1.3 av>=15,<19 numpy huggingface_hub>=1.7,<2.0 tqdm onnxruntime packaging",
)
LLAMA_SERVER = os.environ.get("DSM_LLAMA_SERVER", "")
LLAMA_RELEASE_URL = os.environ.get(
    "DSM_LLAMA_URL",
    "https://github.com/ggml-org/llama.cpp/releases/download/b11518/llama-b11518-bin-ubuntu-cuda-12.8-x64.tar.gz",
)
LLM_REPO = os.environ.get("DSM_LLM_REPO", "")
LLM_FILE = os.environ.get("DSM_LLM_FILE", "")
LLM_ALIAS = os.environ.get("DSM_LLM_ALIAS", "remote-llm")
LLM_CTX = int(os.environ.get("DSM_LLM_CTX", "16384"))
LLM_PARALLEL = int(os.environ.get("DSM_LLM_PARALLEL", "2"))
LLM_PORT = 8080
PART_LIMIT = 64 * 1024 * 1024

STATE = {"stage": "starting", "detail": "", "llm": "off" if not (LLM_REPO or LLAMA_SERVER and LLM_FILE) else "pending", "llm_detail": ""}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()  # GPU에서 인식은 한 번에 하나씩
STARTED = time.time()
LAST_ACTIVITY = [time.time()]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def self_destruct(reason: str) -> None:
    log(f"self-destruct: {reason}")
    if not SELF_DESTRUCT:
        return
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if pod_id and shutil.which("runpodctl"):
        for action in ("remove", "stop"):
            try:
                result = subprocess.run(["runpodctl", action, "pod", pod_id], capture_output=True, text=True, timeout=60)
                log(f"runpodctl {action}: rc={result.returncode} {result.stdout.strip()[:200]} {result.stderr.strip()[:200]}")
                if result.returncode == 0:
                    break
            except Exception as exc:  # noqa: BLE001
                log(f"runpodctl {action} failed: {exc}")
    time.sleep(30)
    os._exit(0)


def watchdog() -> None:
    while True:
        time.sleep(20)
        now = time.time()
        with JOBS_LOCK:
            busy = any(job["status"] in {"queued", "running"} for job in JOBS.values())
        if now - STARTED > MAX_HOURS * 3600:
            self_destruct(f"max lifetime {MAX_HOURS}h reached")
        elif not busy and now - LAST_ACTIVITY[0] > IDLE_MINUTES * 60:
            self_destruct(f"idle for {IDLE_MINUTES} minutes")


def install_deps() -> None:
    if INSTALL_DEPS:
        STATE.update(stage="installing", detail="pip install")
        cmd = f"{PIP} install --no-cache-dir -q " + " ".join(f"'{dep}'" for dep in DEPS.split())
        log(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            STATE.update(stage="error", detail=f"pip failed: {result.stderr[-500:]}")
            return
    STATE.update(stage="waiting_bundle" if not (WORKDIR / "code" / "app").is_dir() else "ready", detail="")


def start_llm() -> None:
    if STATE["llm"] == "off":
        return
    try:
        binary = LLAMA_SERVER
        if not binary:
            STATE.update(llm="downloading", llm_detail="llama.cpp")
            target = WORKDIR / "llama"
            target.mkdir(parents=True, exist_ok=True)
            archive = target / "llama.tar.gz"
            urllib.request.urlretrieve(LLAMA_RELEASE_URL, archive)
            subprocess.run(["tar", "xzf", str(archive), "-C", str(target)], check=True)
            archive.unlink(missing_ok=True)
            binary = str(next(target.rglob("llama-server")))
        model_path = LLM_FILE
        if LLM_REPO:
            STATE.update(llm="downloading", llm_detail=f"{LLM_REPO}/{LLM_FILE}")
            # 의존성 설치가 끝나야 huggingface_hub를 쓸 수 있다.
            while STATE["stage"] in {"starting", "installing"}:
                time.sleep(2)
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(LLM_REPO, LLM_FILE, cache_dir=str(WORKDIR / "hf"))
        STATE.update(llm="starting", llm_detail="loading model")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{Path(binary).parent}:{env.get('LD_LIBRARY_PATH', '')}"
        args = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(LLM_PORT), "-ngl", "999",
                "-c", str(LLM_CTX * LLM_PARALLEL), "--parallel", str(LLM_PARALLEL), "--alias", LLM_ALIAS, "--jinja", "--reasoning", "off"]
        log("llama-server: " + " ".join(args))
        subprocess.Popen(args, env=env, stdout=open(WORKDIR / "llama.log", "ab"), stderr=subprocess.STDOUT)
        for _ in range(900):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{LLM_PORT}/health", timeout=5) as response:
                    if response.status == 200:
                        STATE.update(llm="ready", llm_detail=LLM_ALIAS)
                        log("llama-server ready")
                        return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        STATE.update(llm="error", llm_detail="llama-server did not become ready")
    except Exception as exc:  # noqa: BLE001
        STATE.update(llm="error", llm_detail=str(exc)[:300])
        log(traceback.format_exc())


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else ""
    except Exception:  # noqa: BLE001
        return ""


def job_dir(job_id: str) -> Path:
    safe = "".join(ch for ch in job_id if ch.isalnum() or ch in "-_")[:64]
    if not safe:
        raise ValueError("bad job id")
    return WORKDIR / "jobs" / safe


def run_job(job_id: str, options: dict) -> None:
    job = JOBS[job_id]
    audio_path = job_dir(job_id) / "audio"
    try:
        with RUN_LOCK:
            job.update(status="running", detail="loading")
            code_dir = str(WORKDIR / "code")
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
            from app.config import VAD_MODEL_KEY
            from app.model_manager import download_model, is_model_ready
            from app.transcriber import (
                RuntimeTuningOptions,
                TranscriptionEngine,
                VADSettings,
                build_runtime_config,
                needs_whisperseg_model,
            )

            model_key = options.get("model_key", "medium")
            vad_options = options.get("vad") or {}
            vad = VADSettings(**{k: v for k, v in vad_options.items() if k in VADSettings.__slots__})

            def on_download(done, total, name, *_args):
                job.update(detail=f"model download {name} {done // 1_000_000}/{max(total, 1) // 1_000_000}MB")

            if not is_model_ready(model_key=model_key):
                download_model(model_key, on_download, lambda m: job.update(detail=m))
            if needs_whisperseg_model(model_key, vad) and not is_model_ready(model_key=VAD_MODEL_KEY):
                download_model(VAD_MODEL_KEY, on_download, lambda m: job.update(detail=m))
            tuning = RuntimeTuningOptions(compute_type=options.get("compute_type", "auto"))
            engine = TranscriptionEngine(build_runtime_config(options.get("device", "cuda"), tuning), model_key)
            job.update(detail="transcribing")

            def on_progress(percent, position, duration):
                job.update(progress=int(percent), position=float(position), duration=float(duration))
                LAST_ACTIVITY[0] = time.time()

            started = time.time()
            segments = engine.transcribe_file(audio_path, on_progress, language_code=options.get("language"), vad_settings=vad)
            job.update(
                status="done",
                progress=100,
                detail="",
                elapsed=time.time() - started,
                language=engine.last_detected_language,
                segments=[{"start": s.start, "end": s.end, "text": s.text} for s in segments],
            )
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", error=f"{exc.__class__.__name__}: {exc}"[:2000])
        log(traceback.format_exc())
    finally:
        # 음성은 인식이 끝나면 바로 지운다(성공·실패 모두).
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass
        job["audio_deleted"] = not audio_path.exists()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DSMRemote/" + VERSION

    def log_message(self, fmt, *args):  # noqa: N802
        if not self.path.startswith("/health") and "/jobs/" not in self.path:
            log(f"{self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}")

    def _authorized(self) -> bool:
        header = self.headers.get("X-DSM-Token") or ""
        bearer = self.headers.get("Authorization") or ""
        if bearer.lower().startswith("bearer "):
            bearer = bearer[7:]
        return bool(TOKEN) and (header == TOKEN or bearer == TOKEN)

    def _send(self, code: int, payload, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > PART_LIMIT:
            raise ValueError("body too large")
        return self.rfile.read(length) if length else b""

    def _handle(self) -> None:
        LAST_ACTIVITY[0] = time.time()
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if parts and parts[0] == "v1":
                self._proxy_llm()
            elif self.command == "GET" and parts == ["health"]:
                with JOBS_LOCK:
                    jobs = {k: v["status"] for k, v in JOBS.items()}
                self._send(200, {**STATE, "version": VERSION, "gpu": gpu_name(), "jobs": jobs,
                                 "uptime": time.time() - STARTED, "max_hours": MAX_HOURS, "idle_minutes": IDLE_MINUTES})
            elif self.command == "PUT" and parts == ["bundle"]:
                data = self._body()
                target = WORKDIR / "code"
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.namelist():
                        if member.startswith("/") or ".." in Path(member).parts:
                            raise ValueError("bad bundle path")
                    archive.extractall(target)
                # 새 코드가 올라오면 이미 불러온 앱 모듈을 버려 다음 작업부터 새 코드를 쓴다.
                with RUN_LOCK:
                    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
                        del sys.modules[name]
                if STATE["stage"] == "waiting_bundle":
                    STATE.update(stage="ready")
                self._send(200, {"ok": True, "stage": STATE["stage"]})
            elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "audio" and self.command == "PUT":
                folder = job_dir(parts[1])
                folder.mkdir(parents=True, exist_ok=True)
                offset = int(parse_qs(url.query).get("offset", ["0"])[0])
                data = self._body()
                path = folder / "audio"
                size = path.stat().st_size if path.exists() else 0
                if offset != size:
                    self._send(409, {"error": "offset mismatch", "size": size})
                    return
                with open(path, "ab") as handle:
                    handle.write(data)
                with JOBS_LOCK:
                    JOBS.setdefault(parts[1], {"status": "uploading", "progress": 0})
                self._send(200, {"size": size + len(data)})
            elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "start" and self.command == "POST":
                if STATE["stage"] != "ready":
                    self._send(503, {"error": f"not ready: {STATE['stage']}"})
                    return
                options = json.loads(self._body() or b"{}")
                with JOBS_LOCK:
                    job = JOBS.setdefault(parts[1], {})
                    job.update(status="queued", progress=0, detail="", audio_deleted=False)
                threading.Thread(target=run_job, args=(parts[1], options), daemon=True).start()
                self._send(200, {"ok": True})
            elif len(parts) == 2 and parts[0] == "jobs" and self.command == "GET":
                job = JOBS.get(parts[1])
                self._send(200 if job else 404, job or {"error": "no such job"})
            elif len(parts) == 2 and parts[0] == "jobs" and self.command == "DELETE":
                folder = job_dir(parts[1])
                shutil.rmtree(folder, ignore_errors=True)
                with JOBS_LOCK:
                    JOBS.pop(parts[1], None)
                self._send(200, {"deleted": True, "audio_exists": (folder / "audio").exists()})
            elif self.command == "POST" and parts == ["shutdown"]:
                self._send(200, {"ok": True})
                threading.Thread(target=self_destruct, args=("requested by client",), daemon=True).start()
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            log(traceback.format_exc())
            self._send(500, {"error": f"{exc.__class__.__name__}: {exc}"})

    def _proxy_llm(self) -> None:
        if STATE["llm"] != "ready":
            self._send(503, {"error": f"llm not ready: {STATE['llm']} {STATE['llm_detail']}"})
            return
        data = self._body() if self.command in {"POST", "PUT"} else None
        request = urllib.request.Request(f"http://127.0.0.1:{LLM_PORT}{self.path}", data=data, method=self.command,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                self._send(response.status, response.read(), response.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            self._send(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"))

    do_GET = do_POST = do_PUT = do_DELETE = _handle  # noqa: N815


def main() -> None:
    if not TOKEN:
        print("DSM_TOKEN is required", file=sys.stderr)
        sys.exit(2)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # 앱 코드가 모델을 받는 위치(XDG)를 작업 폴더 아래로 둔다.
    os.environ.setdefault("XDG_DATA_HOME", str(WORKDIR / "data"))
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=install_deps, daemon=True).start()
    threading.Thread(target=start_llm, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log(f"remote worker v{VERSION} listening on :{PORT} (idle {IDLE_MINUTES} min, max {MAX_HOURS} h)")
    server.serve_forever()


if __name__ == "__main__":
    main()
