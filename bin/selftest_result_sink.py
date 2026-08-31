#!/usr/bin/env python3
"""T26  the answer gets off the box -- for every verb, not just published roots.

The defect this covers was found by renting: a container-native run ended with
"receipts under /workspace/fidelity/receipts", on a RunPod pod whose volume is
pod-scoped, whose image runs no sshd, and whose REST API exposes no logs and no
files. ROOT-1's --publish-root-to covered a multi-GB root capture and nothing
else -- not `measure`, whose 4-40 KB receipt IS the submission object, not
`stage`, and not a FAILED run, whose evidence is the hardest to reach and the
most wanted.

Every rung here runs offline: the http rungs drive a stub server on loopback.
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import sys
import tarfile
import tempfile
import threading
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fidelity import resultsink as RS          # noqa: E402

PASS = FAIL = 0
FAILED = []


def check(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  PASS  %s" % label)
    else:
        FAIL += 1
        FAILED.append(label)
        print("  FAIL  %s%s" % (label, ("  -- " + detail) if detail else ""))


def _run_root(tmp, *, receipt_bytes=200, failed=False):
    root = Path(tmp) / "run"
    (root / "receipts").mkdir(parents=True)
    (root / ".secrets").mkdir(parents=True)
    (root / "receipts" / ".stream-work").mkdir(parents=True)
    (root / "receipts" / "measurement-receipt.json").write_text(
        json.dumps({"schema": "submission-receipt.v1", "pad": "x" * receipt_bytes}),
        encoding="utf-8")
    (root / "receipts" / ".stream-work" / "huge.bin").write_text("z" * 4096,
                                                                encoding="utf-8")
    (root / ".secrets" / "hf_token").write_text("hf_TOKEN_MUST_NEVER_LEAVE",
                                                encoding="utf-8")
    (root / "job.json").write_text('{"role":"quant"}', encoding="utf-8")
    return root


def con(_text):
    pass


class _Collector(http.server.BaseHTTPRequestHandler):
    received = []

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Collector.received.append({
            "method": "PUT", "body": self.rfile.read(n),
            "status": self.headers.get("X-Fidelity-Status"),
            "auth": self.headers.get("Authorization"),
        })
        self.send_response(200); self.end_headers()

    do_POST = do_PUT

    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/collect" % srv.server_address[1]


def rung_parse():
    print("[T26.1] the sink list is parsed before anything is spent")
    s = RS.parse_sinks([])
    check("R1 stdout is unconditional and needs no flag",
          len(s) == 1 and s[0].scheme == "stdout")
    s = RS.parse_sinks(["file:/tmp/a", "https://h/u"])
    check("R2 stdout stays FIRST, so a later sink's failure cannot eat the "
          "answer", [x.scheme for x in s] == ["stdout", "file", "http"])
    s = RS.parse_sinks([], env={"FIDELITY_RESULT_SINK": "file:/tmp/a,https://h/u"})
    check("R3 the environment is a sink channel -- the one providers do NOT "
          "echo back in their console", [x.scheme for x in s[1:]] == ["file", "http"])
    s = RS.parse_sinks(["file:/tmp/a"], env={"FIDELITY_RESULT_SINK": "file:/tmp/a"})
    check("R4 the same sink named twice is delivered once", len(s) == 2)
    for bad, why in (("ftp://h/x", "unknown scheme"),
                     ("/tmp/plain", "a bare path is not a URI")):
        try:
            RS.parse_sinks([bad])
            check("R5 %s is refused (%s)" % (bad, why), False, "accepted it")
        except RS.SinkError:
            check("R5 %s is refused (%s)" % (bad, why), True)
    try:
        RS.parse_sinks(["hf://me/repo"])
        check("R6 hf:// is refused, naming --publish-root-to instead", False)
    except RS.SinkError as exc:
        check("R6 hf:// is refused, naming --publish-root-to instead",
              "publish-root-to" in str(exc))


def rung_content():
    print("[T26.2] what leaves the box, and what never does")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(root, "measure", "ok", ["setup", "seal"])
        paths = [f["path"] for f in summary["files"]]
        check("R7 the receipt and the job document are carried",
              "receipts/measurement-receipt.json" in paths and "job.json" in paths)
        check("R8 .secrets/ is NEVER in the manifest",
              not any(".secrets" in p for p in paths), "%s" % paths)
        check("R9 .stream-work/ (the multi-GB scratch tree) is not either",
              not any(".stream-work" in p for p in paths), "%s" % paths)
        check("R10 every carried file is identified by sha256",
              all(len(f["sha256"]) == 64 for f in summary["files"]))
        blob = RS._bundle(root, summary)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            names = tar.getnames()
        check("R11 the tar.gz carries the summary alongside the receipts",
              "result-summary.json" in names)
        check("R12 ... and no secret rides along in the tar either",
              not any(".secrets" in n for n in names), "%s" % names)


def rung_http():
    print("[T26.3] the https sink, against a real server on loopback")
    srv, url = _serve()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(tmp)
            summary = RS.build_summary(root, "measure", "ok", ["seal"])
            res = RS.deliver(root, RS.parse_sinks([url]), summary, con)
            got = [r for r in res if r["scheme"] == "http"]
            check("R13 the bundle is PUT and the endpoint answers 200",
                  got and got[0].get("ok") and got[0].get("code") == 200,
                  "%s" % got)
            body = _Collector.received[-1]["body"] if _Collector.received else b""
            check("R14 the body is a readable gzip tar of the receipts",
                  b"hf_TOKEN_MUST_NEVER_LEAVE" not in body and len(body) > 0)
            with tarfile.open(fileobj=io.BytesIO(body)) as tar:
                names = tar.getnames()
            check("R15 ... carrying the receipt the registry ingests",
                  "receipts/measurement-receipt.json" in names, "%s" % names)
            check("R16 the status rides in a header a collector can route on",
                  _Collector.received[-1]["status"] == "ok")
            os.environ["FIDELITY_RESULT_SINK_AUTH"] = "Bearer TESTVALUE"
            try:
                RS.deliver(root, RS.parse_sinks([url]), summary, con)
                check("R17 an Authorization header comes from the environment, "
                      "never argv",
                      _Collector.received[-1]["auth"] == "Bearer TESTVALUE")
            finally:
                os.environ.pop("FIDELITY_RESULT_SINK_AUTH", None)
    finally:
        srv.shutdown()

    print("[T26.4] a sink that fails does not become a measurement result")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        # A port nothing listens on: the connection is refused, not hung.
        s = socket.socket(); s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]; s.close()
        res = RS.deliver(root, RS.parse_sinks(["http://127.0.0.1:%d/x" % dead]),
                         summary, con)
        http_r = [r for r in res if r["scheme"] == "http"][0]
        stdout_r = [r for r in res if r["scheme"] == "stdout"][0]
        check("R18 the unreachable sink is reported as failed", not http_r["ok"])
        check("R19 ... and stdout still delivered the answer anyway",
              stdout_r["ok"])
        check("R20 the failure names the host but NOT the query string, which "
              "on a presigned URL is the credential",
              "127.0.0.1" in http_r["error"] and "?" not in http_r["error"])


def rung_cap():
    print("[T26.5] a payload too big for a log frame says so")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, receipt_bytes=RS.STDOUT_CAP_BYTES + 10)
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            RS._deliver_stdout(root, summary, con)
        text = buf.getvalue()
        check("R21 the oversize receipt is withheld, not dumped",
              "WITHHELD" in text)
        check("R22 ... and its sha256 is in the frame regardless, so the "
              "artifact is still identifiable",
              summary["files"][0]["sha256"] in text)
        check("R23 the frame markers are present and greppable",
              RS.BEGIN in text and RS.END in text)


def rung_wired():
    print("[T26.6] the entrypoint actually uses it, and the image ships it")
    entry = (HERE / "container_entry.py").read_text(encoding="utf-8")
    check("R24 --result-sink is on the common parser", "--result-sink" in entry)
    # Anchored on the LAST delivery site, not the first: `doctor` also
    # delivers, and it is defined earlier in main(), so a naive index() reads
    # the wrong one and both rungs go green for the wrong reason.
    check("R25 the stage run's delivery is in the finally, so a FAILED run "
          "still reports",
          "RS.deliver" in entry
          and entry.rindex("RS.deliver") > entry.rindex("finally:"))
    shred = entry.index("HF token shredded from the run root")
    check("R26 the token is shredded BEFORE any result leaves the box",
          "RS.deliver" in entry[shred:],
          "no delivery follows the shred")
    check("R27b doctor takes the common flags, so 'rent a pod and check the "
          "image sees the GPU' has an answer you can retrieve",
          'd = sub.add_parser("doctor"' in entry and "add_common(d)" in entry)
    bundle = (HERE / "BUNDLE.txt").read_text(encoding="utf-8").split()
    check("R27 bin/fidelity/resultsink.py is bundled -- an unbundled module is "
          "an image that dies at the last line of a paid run",
          "bin/fidelity/resultsink.py" in bundle)


def main():
    print("== T26 result sinks: getting the answer off the box ==")
    for rung in (rung_parse, rung_content, rung_http, rung_cap, rung_wired):
        rung()
    print("\nT26: %d passed, %d failed" % (PASS, FAIL))
    for f in FAILED:
        print("  - %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
