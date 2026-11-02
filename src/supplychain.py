"""Supply-chain tamper detectors.

Four local, offline checks, one per tampering path:

    verify_checksum              downloaded file matches the publisher's hash
    scan_pickle                  serialized model file would run code on load
    template_render_diff         bundled chat template renders differently from upstream
    listening_inference_servers  a local model server is reachable off-loopback

Each is a detector whose reliability is measured by running it over clean and
tampered samples and reporting a false-positive rate; none inspects a remote host.
"""
from __future__ import annotations

import difflib
import hashlib
import io
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from picklescan.scanner import scan_bytes, SafetyLevel

# ---------- checksum ----------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksum(path: str | Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256.lower().strip()


# ---------- malicious serialized model files ----------
# A pickle (.bin/.pt) runs code on load. Global resolution is delegated to picklescan,
# which simulates the pickle stack (including memoized globals); on top of its verdicts
# we apply the policy below. A clean torch/
# numpy checkpoint references only data reconstructors, so this flags tampering, not
# ordinary weights -- measured as a false-positive rate in supplychain_eval.
#
# Module roots a legitimate weight file may use even when picklescan marks them
# Suspicious (array/tensor/string reconstruction helpers like numpy._core._frombuffer
# and _codecs.encode). Dangerous verdicts are always flagged regardless of module.
_SAFE_RECONSTRUCT_MODULES = {"numpy", "_codecs", "torch", "collections", "copyreg"}


@dataclass(frozen=True)
class PickleFinding:
    module: str
    name: str
    level: str          # picklescan SafetyLevel name: Suspicious | Dangerous


def scan_pickle(data: bytes | BinaryIO) -> list[PickleFinding]:
    """List code-execution risks in a pickle stream. Empty list == no risk under policy.

    Flags every Dangerous global, and every Suspicious global whose module is not a known
    weight-reconstruction helper. Catches os/subprocess/builtins.eval/__import__ gadgets
    across pickle protocols while clearing legitimate numpy/torch state_dicts. Accepts
    bytes or an open binary file; a zip checkpoint is scanned member by member, so a
    multi-GB file is never read into memory.
    """
    result = scan_bytes(io.BytesIO(data) if isinstance(data, bytes) else data, "model")
    findings = []
    for g in result.globals:
        root = g.module.split(".")[0]
        if g.safety == SafetyLevel.Dangerous or (
            g.safety == SafetyLevel.Suspicious and root not in _SAFE_RECONSTRUCT_MODULES
        ):
            findings.append(PickleFinding(g.module, g.name, g.safety.name))
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
        return "gguf", True          # no pickle code, but carries a chat template
    return "unknown", False


# ---------- chat-template tampering ----------
def chat_template_diff(local: str, upstream: str) -> str:
    """Unified diff of a bundled chat template vs the upstream reference; '' if identical."""
    if local == upstream:
        return ""
    return "\n".join(difflib.unified_diff(
        upstream.splitlines(), local.splitlines(),
        fromfile="upstream", tofile="local", lineterm=""))


# A text diff flags harmless edits (comments, spacing inside tags). What matters is whether
# the template turns the same conversation into a different prompt, so render both the way
# Hugging Face does (sandboxed Jinja, trim/lstrip blocks) on fixed probe conversations.
# Each probe exercises a different branch: system prompt, multi-turn, closed turn, the
# tool-calling path, and the no-thinking switch (how Qwen3 is run in this audit). A change
# confined to a branch no probe reaches would go unseen, so every branch needs a probe.
_TOOL = {"type": "function", "function": {
    "name": "get_time", "description": "Current time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
PROBE_CONVERSATIONS = {
    "single_turn": ([{"role": "user", "content": "Hello."}], True, {}),
    "with_system": ([{"role": "system", "content": "Be concise."},
                     {"role": "user", "content": "What is 2+2?"},
                     {"role": "assistant", "content": "4."},
                     {"role": "user", "content": "And 3+3?"}], True, {}),
    "closed_turn": ([{"role": "user", "content": "Hi."},
                     {"role": "assistant", "content": "Hello!"}], False, {}),
    "with_tools": ([{"role": "user", "content": "Time in Paris?"}], True, {"tools": [_TOOL]}),
    "tools_in_system": ([{"role": "user", "content": "Time in Paris?"}], True,
                        {"tools": [_TOOL], "tools_in_user_message": False}),     # Llama 3.1 switch
    "no_thinking": ([{"role": "user", "content": "Hello."}], True, {"enable_thinking": False}),
}


def _jinja_env():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, separators=None, \
        sort_keys=False: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii,
                                    separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "01 Jan 2026"     # fixed: renders must be comparable
    return env


def render_template(template: str, messages: list[dict], add_generation_prompt: bool,
                    extra: dict | None = None) -> str:
    try:
        return _jinja_env().from_string(template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token="<bos>", eos_token="<eos>", **(extra or {}))
    except Exception as e:                       # a template that now fails is also a change
        return f"<render error: {type(e).__name__}: {e}>"


def template_render_diff(local: str, upstream: str) -> list[str]:
    """Probe conversations whose rendered prompt differs; [] if the templates behave the same."""
    return [name for name, (msgs, gen, extra) in PROBE_CONVERSATIONS.items()
            if render_template(local, msgs, gen, extra) != render_template(upstream, msgs, gen, extra)]


def load_chat_template(model_dir: str | Path) -> str | None:
    """The template a runtime would use: chat_template.jinja, else tokenizer_config.json."""
    d = Path(model_dir)
    if (d / "chat_template.jinja").exists():
        return (d / "chat_template.jinja").read_text()
    cfg = d / "tokenizer_config.json"
    if not cfg.exists():
        return None
    t = json.loads(cfg.read_text()).get("chat_template")
    if isinstance(t, list):                      # named templates: use "default"
        t = next((x["template"] for x in t if x.get("name") == "default"), None)
    return t


# ---------- exposed local inference server ----------
# Default ports of common self-hosted runtimes.
INFERENCE_PORTS = {11434: "ollama", 8000: "vllm", 8080: "llama.cpp",
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
    address. A port reachable on the LAN IP is exposed beyond localhost; a port
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
    for proto in (0, 2, 4, 5):
        findings = scan_pickle(pickle.dumps(Evil(), protocol=proto))
        assert findings, f"malicious pickle (proto {proto}) not detected"
        assert any(f.name == "system" for f in findings), findings
        assert all(f.level in ("Suspicious", "Dangerous") for f in findings)

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
    # Render diff: a comment or extra spacing inside a tag is not a behavior change; an
    # injected instruction is.
    assert template_render_diff("{# reviewed #}" + base.replace("{% for", "{%  for"), base) == []
    assert template_render_diff(tampered, base) == list(PROBE_CONVERSATIONS)

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
