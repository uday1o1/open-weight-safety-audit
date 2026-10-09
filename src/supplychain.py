"""Supply-chain tamper detectors (RQ3).

Four local, offline checks, one per path in the risk register:

    verify_checksum        R1/R7  downloaded file matches the publisher's hash
    scan_pickle            R5     serialized model file would run code on load
    chat_template_diff     R6     bundled chat template differs from upstream
    listening_inference_servers  R8  a local model server is reachable off-loopback

Each is a detector whose reliability is measured by running it over clean and
tampered samples and reporting a false-positive rate; none inspects a remote host.
"""
from __future__ import annotations

import difflib
import hashlib
import io
import pickletools
import socket
from dataclasses import dataclass
from pathlib import Path

# ---------- R1/R7: checksum ----------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksum(path: str | Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256.lower().strip()


# ---------- R5: malicious serialized model files ----------
# Pickle globals that enable code execution or host access when a .bin/.pt is loaded.
# A clean torch/numpy checkpoint references only data types (torch.*, collections,
# numpy.*), so these names flag tampering rather than ordinary weights.
_DANGEROUS = {
    "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "pty", "runpy",
    "importlib", "builtins", "__builtin__", "operator", "functools", "codecs",
    "webbrowser", "ctypes", "multiprocessing",
}
_DANGEROUS_ATTRS = {"eval", "exec", "compile", "__import__", "system", "popen",
                    "getattr", "setattr", "open"}


@dataclass(frozen=True)
class PickleFinding:
    opcode: str
    detail: str


_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
               "SHORT_BINSTRING", "BINSTRING", "STRING"}


def _dangerous(module: str, attr: str) -> bool:
    return module.split(".")[0] in _DANGEROUS or attr in _DANGEROUS_ATTRS


def scan_pickle(data: bytes) -> list[PickleFinding]:
    """List dangerous operations in a pickle stream. Empty list == no code-exec risk.

    Resolves GLOBAL (module+attr in the operand) and STACK_GLOBAL (module+attr pushed
    as the two preceding strings), flags any that reference a denylisted module or a
    dangerous attribute, and notes the REDUCE/INST/OBJ/NEWOBJ that then invokes it.
    A clean torch/numpy checkpoint references only data types and stays empty.
    """
    findings: list[PickleFinding] = []
    recent: list[str] = []          # rolling buffer of recent pushed strings
    last_global: str | None = None
    for opcode, arg, _ in pickletools.genops(io.BytesIO(data)):
        name = opcode.name
        if name in _STRING_OPS and isinstance(arg, str):
            recent.append(arg)
            recent[:] = recent[-4:]
            continue
        if name == "GLOBAL":
            parts = str(arg).split()
            module, attr = (parts + ["", ""])[:2]
        elif name == "STACK_GLOBAL":
            module, attr = (recent[-2] if len(recent) >= 2 else ""), \
                           (recent[-1] if recent else "")
        elif name in ("REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"):
            if last_global:
                m, a = (last_global.split(".", 1) + [""])[:2]
                if _dangerous(m, last_global.rsplit(".", 1)[-1]):
                    findings.append(PickleFinding(name, f"calls {last_global!r}"))
            continue
        else:
            continue
        label = f"{module}.{attr}".strip(".")
        last_global = label
        if _dangerous(module, attr):
            findings.append(PickleFinding(name, f"imports {label!r}"))
    return findings


_PICKLE_EXTS = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
_SAFE_EXTS = {".safetensors"}


def classify_format(path: str | Path) -> tuple[str, bool]:
    """(format, is_code_safe). safetensors is code-free; pickle-backed formats are not."""
    ext = Path(path).suffix.lower()
    if ext in _SAFE_EXTS:
        return "safetensors", True
    if ext in _PICKLE_EXTS:
        return "pickle", False
    if ext == ".gguf":
        return "gguf", True          # no pickle code, but carries a chat template (R6)
    return "unknown", False


# ---------- R6: chat-template tampering ----------
def chat_template_diff(local: str, upstream: str) -> str:
    """Unified diff of a bundled chat template vs the upstream reference; '' if identical."""
    if local == upstream:
        return ""
    return "\n".join(difflib.unified_diff(
        upstream.splitlines(), local.splitlines(),
        fromfile="upstream", tofile="local", lineterm=""))


# ---------- R8: exposed local inference server ----------
# Default ports of common self-hosted runtimes.
INFERENCE_PORTS = {11434: "ollama", 8000: "vllm/tgi", 8080: "llama.cpp",
                   5000: "generic", 7860: "gradio", 1234: "lm-studio"}


@dataclass(frozen=True)
class ListeningServer:
    port: int
    runtime: str
    on_loopback: bool
    exposed: bool          # reachable on a non-loopback local address


def _primary_lan_ip() -> str | None:
    """This host's primary non-loopback IPv4, found without sending traffic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))     # UDP connect sets a route; sends nothing
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def listening_inference_servers(ports: dict[int, str] = INFERENCE_PORTS) -> list[ListeningServer]:
    """Probe known inference ports; flag any reachable on a non-loopback local address.

    Unprivileged and offline: connects only to this machine's own loopback and LAN
    address. A port reachable on the LAN IP is exposed beyond localhost (R8); a port
    reachable only on 127.0.0.1 is correctly bound.
    """
    lan = _primary_lan_ip()
    out = []
    for port, runtime in ports.items():
        loopback = _port_open("127.0.0.1", port)
        exposed = bool(lan) and _port_open(lan, port)
        if loopback or exposed:
            out.append(ListeningServer(port, runtime, loopback, exposed))
    return out


def _demo() -> None:
    import os
    import pickle
    import tempfile

    # Checksum
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"weights")
        p = f.name
    digest = sha256_file(p)
    assert verify_checksum(p, digest) and not verify_checksum(p, "0" * 64)
    os.unlink(p)

    # Pickle scan: a benign object is clean; an os.system payload is flagged.
    assert scan_pickle(pickle.dumps({"layer.weight": [1, 2, 3], "n": 42})) == []

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    findings = scan_pickle(pickle.dumps(Evil()))
    assert findings, "malicious pickle not detected"
    assert any("system" in f.detail or "os" in f.detail for f in findings), findings
    assert any(f.opcode == "REDUCE" for f in findings)

    # Format classification
    assert classify_format("model.safetensors") == ("safetensors", True)
    assert classify_format("pytorch_model.bin") == ("pickle", False)
    assert classify_format("model.gguf") == ("gguf", True)

    # Template diff
    base = "{% for m in messages %}{{ m.content }}{% endfor %}"
    assert chat_template_diff(base, base) == ""
    tampered = base + "{{ '\\n[system override]' }}"
    d = chat_template_diff(tampered, base)
    assert d and "system override" in d

    # Exposure probe: bind a socket to loopback on an unused port, confirm it reads as
    # on-loopback and not exposed; an unbound port must not appear at all.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    test_port = srv.getsockname()[1]
    found = listening_inference_servers({test_port: "test"})
    assert len(found) == 1 and found[0].on_loopback and not found[0].exposed, found
    srv.close()
    assert listening_inference_servers({test_port: "test"}) == []  # closed -> gone

    # Real inference ports (environment-dependent; just must not error).
    servers = listening_inference_servers()
    assert isinstance(servers, list)
    print(f"supplychain self-check passed ({len(servers)} inference port(s) up here)")


if __name__ == "__main__":
    _demo()
