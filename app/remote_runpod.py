"""Runpod 원격 실행: 사용자의 Runpod 계정에 GPU Pod를 잠깐 만들어 음성 인식(+선택: LLM 번역)을 돌리고 지운다.

왜 서버리스가 아니라 Pod인가
- 서버리스 엔드포인트는 미리 만든 Docker 이미지(레지스트리에 올린 것)가 있어야 하는데, 이 앱은 구간별 인식·
  WhisperSeg VAD 등 앱 코드 그대로 인식해야 해서 공개 이미지로는 안 된다.
- 그래서 공식 PyTorch 이미지로 Pod를 만들고, 시작 명령에서 작은 워커 서버(remote_worker.py)를 띄운 뒤 앱 코드를
  올려 쓴다. 작업이 끝나면(실패·취소·앱 종료 포함) Pod를 **삭제**하므로 쓰지 않을 때 요금은 0이다.
- 앱이 갑자기 꺼지거나 인터넷이 끊겨도, Pod 안의 워커가 요청이 없으면(기본 15분) 또는 최대 시간이 지나면 스스로
  Pod를 삭제한다. 다음에 앱을 켜면 남은 Pod(이름이 dsm-remote-로 시작)를 찾아 지울 수 있다.

수동 연결: 이미 떠 있는 워커(직접 만든 Pod 등)의 주소와 토큰을 넣으면 Pod를 만들거나 지우지 않고 그 워커만 쓴다.
"""
from __future__ import annotations

import base64
import io
import json
import secrets
import time
import uuid
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .config import get_project_root, get_runtime_base_dir
from .srt_writer import SubtitleSegment

REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
POD_NAME_PREFIX = "dsm-remote-"
WORKER_PORT = 8000
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
UPLOAD_PART_BYTES = 16 * 1024 * 1024
# 고를 수 있는 GPU(24GB 이상이면 Gemma 26B Q4도 함께 올라간다). 가격은 실시간 조회가 실패할 때 쓰는 참고값($/h, 2026-10).
GPU_CHOICES = [
    ("NVIDIA GeForce RTX 3090", "RTX 3090 24GB", {"COMMUNITY": 0.22, "SECURE": 0.50}),
    ("NVIDIA RTX A5000", "RTX A5000 24GB", {"COMMUNITY": 0.16, "SECURE": 0.27}),
    ("NVIDIA GeForce RTX 4090", "RTX 4090 24GB", {"COMMUNITY": 0.34, "SECURE": 0.89}),
    ("NVIDIA L4", "L4 24GB", {"COMMUNITY": 0.44, "SECURE": 0.59}),
]
# 원격 LLM 번역용 모델(GGUF). 번역 비교 실험(eval/RESULTS.md) 결과로 기본값을 정한다.
LLM_CHOICES = {
    "none": {"label": "사용 안 함 (번역은 Gemini)", "repo": "", "file": ""},
    # 번역 비교(eval/RESULTS.md 6-1): 블라인드 품질 Gemini와 같거나 약간 높고 차단 0. 24GB GPU에서 음성 인식 모델과
    # 함께 올라가도록 Q4_K_M(16.8GB) 대신 IQ4_XS(13.9GB)를 쓴다.
    "gemma4-26b": {
        "label": "Gemma4 26B-A4B Uncensored (IQ4_XS, 14GB) — 추천",
        "repo": "HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced",
        "file": "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf",
    },
    "command-r7b": {
        "label": "Command R7B abliterated (Q6_K, 비상업 라이선스, 실험에서 번역 품질 낮음)",
        "repo": "bartowski/c4ai-command-r7b-12-2024-abliterated-GGUF",
        "file": "c4ai-command-r7b-12-2024-abliterated-Q6_K.gguf",
    },
}
# 비용 추정에 쓰는 대략값(3090 기준 실측, eval/RESULTS.md)
BOOT_MINUTES = 8.0  # Pod 시작 + 패키지 설치 + 모델 다운로드
ASR_RTF = 0.05  # 음성 1시간당 GPU 3분(구간별 인식, 업로드 대기 포함 여유)
LLM_MINUTES_PER_AUDIO_HOUR = 6.0


class RemoteError(RuntimeError):
    pass


@dataclass(slots=True)
class RemoteSettings:
    mode: str = "auto"  # "auto": Pod를 만들고 끝나면 삭제 / "manual": 이미 떠 있는 워커 주소 사용
    api_key: str = ""
    gpu_type_id: str = GPU_CHOICES[0][0]
    cloud_type: str = "COMMUNITY"
    # 원격 실행을 고른 사용자는 Gemini 차단·한도를 피하도록 원격 LLM 번역을 기본으로 쓴다(끄면 Gemini).
    llm_choice: str = "gemma4-26b"
    idle_minutes: int = 15
    max_hours: float = 6.0
    manual_url: str = ""
    manual_token: str = ""


@dataclass(slots=True)
class CostEstimate:
    price_per_hour: float
    price_source: str
    minutes: float
    cost: float
    max_cost: float

    def describe(self, gpu_label: str, cloud: str) -> str:
        return (
            f"GPU: {gpu_label} ({'커뮤니티' if cloud == 'COMMUNITY' else '보안'} 클라우드)\n"
            f"시간당 요금: ${self.price_per_hour:.2f} ({self.price_source})\n"
            f"예상 사용 시간: 약 {self.minutes:.0f}분 → 예상 비용 약 ${self.cost:.2f}\n"
            f"최대 비용(문제가 생겨 자동 종료 시간까지 켜져 있을 때): ${self.max_cost:.2f}"
        )


def gpu_label(gpu_type_id: str) -> str:
    return next((label for gid, label, _ in GPU_CHOICES if gid == gpu_type_id), gpu_type_id)


def fallback_price(gpu_type_id: str, cloud: str) -> float:
    for gid, _label, prices in GPU_CHOICES:
        if gid == gpu_type_id:
            return prices.get(cloud, max(prices.values()))
    return 1.0


def estimate_cost(price: float, source: str, audio_hours: float, use_llm: bool, max_hours: float) -> CostEstimate:
    minutes = BOOT_MINUTES + audio_hours * 60 * ASR_RTF + (audio_hours * LLM_MINUTES_PER_AUDIO_HOUR if use_llm else 0)
    if use_llm:
        minutes += 5  # LLM 모델 다운로드·로딩
    return CostEstimate(price, source, minutes, price * minutes / 60, price * max_hours)


class RunpodApi:
    """Runpod REST v1(https://rest.runpod.io/v1) + 가격 조회용 GraphQL."""

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        if not api_key.strip():
            raise RemoteError("Runpod API 키가 비어 있습니다.")
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}
        self._http = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0))

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code == 401:
            raise RemoteError("Runpod API 키가 거부되었습니다(401). 키를 확인하세요.")
        if response.status_code >= 400:
            raise RemoteError(f"Runpod API 오류 {response.status_code}: {response.text[:300]}")
        return response

    def create_pod(self, body: dict) -> dict:
        return self._check(self._http.post(f"{REST_URL}/pods", json=body, headers=self._headers)).json()

    def get_pod(self, pod_id: str) -> dict | None:
        response = self._http.get(f"{REST_URL}/pods/{pod_id}", headers=self._headers)
        if response.status_code == 404:
            return None
        return self._check(response).json()

    def list_pods(self) -> list[dict]:
        data = self._check(self._http.get(f"{REST_URL}/pods", headers=self._headers)).json()
        return data if isinstance(data, list) else data.get("pods", [])

    def delete_pod(self, pod_id: str) -> None:
        response = self._http.delete(f"{REST_URL}/pods/{pod_id}", headers=self._headers)
        if response.status_code not in {200, 204, 404}:
            self._check(response)

    def gpu_price(self, gpu_type_id: str, cloud: str) -> float | None:
        query = 'query { gpuTypes(input: {id: "%s"}) { id communityPrice securePrice } }' % gpu_type_id
        try:
            response = self._http.post(GRAPHQL_URL, json={"query": query}, headers=self._headers)
            items = response.json()["data"]["gpuTypes"]
            if items:
                price = items[0]["communityPrice" if cloud == "COMMUNITY" else "securePrice"]
                return float(price) if price else None
        except Exception:  # noqa: BLE001
            return None
        return None


def _app_source_dir() -> Path:
    """앱 코드(.py) 폴더. exe 빌드에는 소스가 없으므로 빌드 스크립트가 remote_bundle/app으로 함께 넣는다."""
    candidates = [get_runtime_base_dir() / "remote_bundle" / "app", get_project_root() / "app", Path(__file__).parent]
    source_dir = next((path for path in candidates if (path / "transcriber.py").is_file()), None)
    if source_dir is None:
        raise RemoteError("원격 실행용 앱 코드를 찾지 못했습니다(빌드에 remote_bundle이 빠졌을 수 있음).")
    return source_dir


def worker_source() -> str:
    return (_app_source_dir() / "remote_worker.py").read_text(encoding="utf-8")


def build_code_bundle() -> bytes:
    """원격 워커가 import할 앱 코드(app 패키지의 .py)를 zip으로 묶는다."""
    source_dir = _app_source_dir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.glob("*.py")):
            archive.write(path, f"app/{path.name}")
    return buffer.getvalue()


def build_pod_body(settings: RemoteSettings, token: str) -> dict:
    boot = base64.b64encode(zlib.compress(worker_source().encode("utf-8"), 9)).decode("ascii")
    env = {
        "DSM_TOKEN": token,
        "DSM_BOOT": boot,
        "DSM_IDLE_MINUTES": str(settings.idle_minutes),
        "DSM_MAX_HOURS": str(settings.max_hours),
        "DSM_PORT": str(WORKER_PORT),
    }
    llm = LLM_CHOICES.get(settings.llm_choice, LLM_CHOICES["none"])
    if llm["repo"]:
        env.update(DSM_LLM_REPO=llm["repo"], DSM_LLM_FILE=llm["file"], DSM_LLM_ALIAS=settings.llm_choice)
    start = (
        'python3 -c "import base64,os,zlib;'
        "open('/dsm_worker.py','w').write(zlib.decompress(base64.b64decode(os.environ['DSM_BOOT'])).decode())\" "
        "&& exec python3 /dsm_worker.py"
    )
    return {
        "name": f"{POD_NAME_PREFIX}{time.strftime('%m%d-%H%M')}",
        "imageName": DEFAULT_IMAGE,
        "gpuTypeIds": [settings.gpu_type_id],
        "gpuCount": 1,
        "cloudType": settings.cloud_type,
        "containerDiskInGb": 40 if llm["repo"] else 25,
        "volumeInGb": 0,
        "ports": [f"{WORKER_PORT}/http"],
        "env": env,
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [start],
        "allowedCudaVersions": ["12.8", "12.9", "13.0", "13.1", "13.2"],
    }


class WorkerClient:
    """remote_worker.py와 HTTP로 이야기한다. 네트워크가 끊기면 network(NetworkWaiter)가 기다렸다가 다시 보낸다."""

    def __init__(self, base_url: str, token: str, network=None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.network = network
        self.sleep = sleep
        self._http = httpx.Client(timeout=httpx.Timeout(300.0, connect=20.0), headers={"X-DSM-Token": token})

    @property
    def llm_base_url(self) -> str:
        return f"{self.base_url}/v1"

    def _call(self, method: str, path: str, **kwargs) -> httpx.Response:
        def request():
            return self._http.request(method, f"{self.base_url}{path}", **kwargs)

        response = self.network.run(request, self.sleep) if self.network is not None else request()
        if response.status_code == 401:
            raise RemoteError("원격 워커가 토큰을 거부했습니다(401).")
        return response

    def health(self) -> dict:
        response = self._call("GET", "/health")
        if response.status_code != 200:
            raise RemoteError(f"워커 상태 확인 실패 {response.status_code}: {response.text[:200]}")
        return response.json()

    def upload_bundle(self, data: bytes) -> None:
        response = self._call("PUT", "/bundle", content=data)
        if response.status_code != 200:
            raise RemoteError(f"앱 코드 업로드 실패 {response.status_code}: {response.text[:200]}")

    def transcribe(self, audio_path: Path, options: dict, progress: Callable[[int, float, float], None] | None,
                   log: Callable[[str], None], cancel_check: Callable[[], bool] | None = None) -> tuple[list[SubtitleSegment], str | None]:
        job_id = uuid.uuid4().hex
        data = audio_path.read_bytes()
        offset = 0
        try:
            while offset < len(data):
                part = data[offset : offset + UPLOAD_PART_BYTES]
                response = self._call("PUT", f"/jobs/{job_id}/audio", params={"offset": offset}, content=part)
                if response.status_code == 409:
                    offset = int(response.json().get("size", 0))  # 끊겼다 이어서 올릴 때
                    continue
                if response.status_code != 200:
                    raise RemoteError(f"음성 업로드 실패 {response.status_code}: {response.text[:200]}")
                offset = int(response.json()["size"])
            response = self._call("POST", f"/jobs/{job_id}/start", json=options)
            if response.status_code != 200:
                raise RemoteError(f"원격 인식 시작 실패 {response.status_code}: {response.text[:200]}")
            last_detail = ""
            while True:
                if cancel_check is not None and cancel_check():
                    raise RemoteError("취소됨")
                job = self._call("GET", f"/jobs/{job_id}").json()
                status = job.get("status")
                detail = job.get("detail") or ""
                if detail and detail != last_detail and not detail.startswith("transcribing"):
                    log(f"원격: {detail}")
                    last_detail = detail
                if progress is not None and job.get("duration"):
                    progress(int(job.get("progress", 0)), float(job.get("position", 0.0)), float(job["duration"]))
                if status == "done":
                    segments = [SubtitleSegment(start=s["start"], end=s["end"], text=s["text"]) for s in job.get("segments", [])]
                    if not job.get("audio_deleted", False):
                        log("경고: 원격 음성 파일이 지워졌는지 확인하지 못했습니다. 아래 삭제 요청으로 다시 지웁니다.")
                    return segments, job.get("language")
                if status == "error":
                    raise RemoteError(f"원격 인식 실패: {job.get('error')}")
                self.sleep(2.0)
        finally:
            try:
                result = self._call("DELETE", f"/jobs/{job_id}").json()
                if result.get("audio_exists"):
                    log("경고: 원격 음성 파일 삭제를 확인하지 못했습니다.")
            except Exception as exc:  # noqa: BLE001
                log(f"원격 작업 정리 요청 실패(Pod 삭제 때 함께 지워집니다): {exc}")

    def shutdown(self) -> None:
        try:
            self._call("POST", "/shutdown")
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self._http.close()


@dataclass
class RemoteSession:
    """한 번의 작업(대기열) 동안 원격 워커를 쓰고, 끝나면 Pod를 지운다."""

    settings: RemoteSettings
    log: Callable[[str], None]
    network: object = None
    cancel_check: Callable[[], bool] | None = None
    api: RunpodApi | None = None
    pod_id: str | None = None
    worker: WorkerClient | None = None
    cost_per_hour: float = 0.0
    started_at: float = field(default_factory=time.time)

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while True:
            if self.cancel_check is not None and self.cancel_check():
                raise RemoteError("취소됨")
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    def start(self, progress: Callable[[str], None] | None = None, max_price: float | None = None) -> WorkerClient:
        report = progress or self.log
        if self.settings.mode == "manual":
            if not self.settings.manual_url.strip() or not self.settings.manual_token.strip():
                raise RemoteError("수동 연결에는 워커 주소와 토큰이 필요합니다.")
            self.worker = WorkerClient(self.settings.manual_url, self.settings.manual_token, self.network, self._sleep)
        else:
            self.api = self.api or RunpodApi(self.settings.api_key)
            token = secrets.token_urlsafe(24)
            report("Runpod Pod 만드는 중")
            pod = self.api.create_pod(build_pod_body(self.settings, token))
            self.pod_id = str(pod["id"])
            self.cost_per_hour = float(pod.get("costPerHr") or pod.get("adjustedCostPerHr") or 0.0)
            self.log(f"Runpod Pod 생성: {self.pod_id} (시간당 ${self.cost_per_hour:.2f})")
            if max_price is not None and self.cost_per_hour > max_price * 1.25 + 0.01:
                raise RemoteError(f"Pod 요금(${self.cost_per_hour:.2f}/h)이 확인한 요금보다 높아 취소합니다.")
            url = f"https://{self.pod_id}-{WORKER_PORT}.proxy.runpod.net"
            self.worker = WorkerClient(url, token, self.network, self._sleep)
        self._wait_ready(report)
        return self.worker

    def _wait_ready(self, report: Callable[[str], None], timeout_minutes: float = 25.0) -> None:
        assert self.worker is not None
        deadline = time.monotonic() + timeout_minutes * 60
        bundle_sent = False
        last = ""
        while time.monotonic() < deadline:
            try:
                state = self.worker.health()
            except Exception as exc:  # noqa: BLE001  (Pod가 뜨는 중이면 프록시가 502/연결 오류를 낸다)
                state = {"stage": "booting", "detail": str(exc)[:80]}
            stage = state.get("stage")
            message = f"원격 준비: {stage} {state.get('detail') or ''}".strip()
            if message != last:
                report(message)
                last = message
            if stage == "error":
                raise RemoteError(f"원격 워커 준비 실패: {state.get('detail')}")
            if stage in {"waiting_bundle", "ready"} and not bundle_sent:
                report("앱 코드 올리는 중")
                self.worker.upload_bundle(build_code_bundle())
                bundle_sent = True
                continue
            if stage == "ready" and bundle_sent:
                self.log(f"원격 워커 준비 완료: {state.get('gpu', '')}")
                return
            self._sleep(5)
        raise RemoteError(f"원격 워커가 {timeout_minutes:.0f}분 안에 준비되지 않았습니다.")

    def wait_llm(self, report: Callable[[str], None], timeout_minutes: float = 30.0) -> str:
        """원격 LLM(llama-server)이 뜰 때까지 기다리고 OpenAI 호환 주소를 돌려준다."""
        assert self.worker is not None
        deadline = time.monotonic() + timeout_minutes * 60
        last = ""
        while time.monotonic() < deadline:
            state = self.worker.health()
            llm = state.get("llm")
            message = f"원격 LLM: {llm} {state.get('llm_detail') or ''}".strip()
            if message != last:
                report(message)
                last = message
            if llm == "ready":
                return self.worker.llm_base_url
            if llm in {"error", "off"}:
                raise RemoteError(f"원격 LLM을 쓸 수 없습니다: {state.get('llm_detail') or llm}")
            self._sleep(5)
        raise RemoteError("원격 LLM이 제시간에 준비되지 않았습니다.")

    def close(self) -> None:
        """Pod를 지우고(자동 모드) 지워졌는지 확인한다. 실패하면 워커에게 스스로 지우라고 요청한다."""
        if self.worker is not None and self.settings.mode == "manual":
            self.worker.close()
            return
        if self.pod_id and self.api is not None:
            for attempt in range(5):
                try:
                    self.api.delete_pod(self.pod_id)
                    if self.api.get_pod(self.pod_id) is None:
                        minutes = (time.time() - self.started_at) / 60
                        self.log(
                            f"Runpod Pod 삭제 완료: {self.pod_id} (약 {minutes:.0f}분 사용, 약 ${self.cost_per_hour * minutes / 60:.2f})"
                        )
                        self.pod_id = None
                        break
                except Exception as exc:  # noqa: BLE001
                    self.log(f"Pod 삭제 재시도 {attempt + 1}/5: {exc}")
                time.sleep(3 * (attempt + 1))
            if self.pod_id:
                if self.worker is not None:
                    self.worker.shutdown()
                self.log(
                    f"경고: Pod {self.pod_id} 삭제를 확인하지 못했습니다. Pod가 스스로 삭제를 시도하지만, "
                    "Runpod 콘솔(https://console.runpod.io/pods)에서 꼭 확인하세요."
                )
        if self.worker is not None:
            self.worker.close()


def find_leftover_pods(api_key: str) -> list[dict]:
    api = RunpodApi(api_key)
    return [pod for pod in api.list_pods() if str(pod.get("name", "")).startswith(POD_NAME_PREFIX)]


def encode_for_upload(source: Path, target: Path) -> float:
    """원격으로 보낼 음성을 16kHz 모노 Opus(48kbps)로 만든다(1시간에 약 21MB). 길이(초)를 돌려준다."""
    import av

    duration = 0.0
    with av.open(str(source)) as src, av.open(str(target), "w", format="ogg") as sink:
        stream = sink.add_stream("libopus", rate=16000, layout="mono")
        stream.bit_rate = 48000
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in src.decode(audio=0):
            for out in resampler.resample(frame):
                duration += out.samples / 16000
                for packet in stream.encode(out):
                    sink.mux(packet)
        for out in resampler.resample(None):
            for packet in stream.encode(out):
                sink.mux(packet)
        for packet in stream.encode(None):
            sink.mux(packet)
    return duration


def save_remote_settings(path: Path, settings: RemoteSettings) -> None:
    data = {k: getattr(settings, k) for k in RemoteSettings.__slots__ if k not in {"api_key", "manual_token"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_remote_settings(path: Path) -> RemoteSettings:
    settings = RemoteSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return settings
    for key in RemoteSettings.__slots__:
        if key in data and key not in {"api_key", "manual_token"}:
            setattr(settings, key, type(getattr(settings, key))(data[key]))
    return settings


def probe_audio_seconds(paths: list[Path]) -> float:
    """파일 머리말(메타데이터)만 읽어 전체 길이를 구한다(비용 추정용)."""
    import av

    total = 0.0
    for path in paths:
        try:
            with av.open(str(path)) as container:
                if container.duration is not None:
                    total += float(container.duration / av.time_base)
        except Exception:  # noqa: BLE001
            continue
    return total


def prepare_cost_confirmation(
    settings: RemoteSettings,
    audio_seconds: float,
    price_lookup: Callable[[str, str], float | None] | None = None,
) -> tuple[CostEstimate | None, str]:
    """시작 전에 보여 줄 비용 안내문. 수동 연결이면 추정 없이 안내만 한다."""
    hours = audio_seconds / 3600
    if settings.mode == "manual":
        return None, (
            f"이미 떠 있는 원격 워커({settings.manual_url})로 음성 {hours:.1f}시간 분량을 처리합니다.\n"
            "요금은 그 Pod를 만든 계정에 그 Pod의 시간당 요금으로 나갑니다. 이 앱은 그 Pod를 지우지 않습니다."
        )
    price = price_lookup(settings.gpu_type_id, settings.cloud_type) if price_lookup else None
    source = "Runpod 실시간 가격"
    if price is None:
        price = fallback_price(settings.gpu_type_id, settings.cloud_type)
        source = "참고 가격(실시간 조회 실패)"
    estimate = estimate_cost(price, source, hours, settings.llm_choice != "none", settings.max_hours)
    message = (
        f"Runpod에 GPU Pod를 만들어 음성 {hours:.1f}시간 분량을 처리하고, 끝나면 Pod를 삭제합니다.\n\n"
        + estimate.describe(gpu_label(settings.gpu_type_id), settings.cloud_type)
        + f"\n\n안전장치: 요청이 {settings.idle_minutes}분 동안 없거나 {settings.max_hours:g}시간이 지나면 Pod가 스스로 삭제됩니다."
        "\n음성은 Opus로 변환해 올리고, 인식이 끝나면 원격에서 바로 지웁니다.\n\n시작할까요?"
    )
    return estimate, message
