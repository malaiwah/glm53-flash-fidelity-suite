#!/usr/bin/env python3
"""The bearer token must never follow a redirect off the original origin.

WHY THIS EXISTS
---------------
urllib's default redirect handler copies every header except
content-length/content-type onto the redirected request -- including
`Authorization`.  Hugging Face `/resolve/` URLs 302 to pre-signed CDN/Xet
hosts, so any stdlib client using the default handler hands the Hub token to
whatever host the endpoint names.  The 2026-08-31 peer review demonstrated
the leak against a local adversarial redirect: `dshub` had the correct
defense (`_NoCrossHostAuth`), while `hfmeta` and
`engines/tools/fetch_truncated_ckpt.py` were still on the default handler.

The one correct implementation now lives in
`fidelity.common.make_no_cross_origin_auth_handler()` (with a documented
standalone copy in the truncation fetcher, which ships to remote boxes as a
single file).  These rungs drive REAL redirects through a local stub server
-- no network, no Hugging Face, no token that exists.

  R1  same-origin redirect KEEPS Authorization (the canonical-case 307).
  R2  cross-host redirect STRIPS it (the CDN hop).
  R3  cross-port redirect STRIPS it (origin is scheme+host+port, not host).
  R4  https->http downgrade on the same host STRIPS it (unit-level: local
      stub servers cannot speak TLS, so the handler is driven directly).
  R5  a redirect loop terminates with an error after a bounded number of
      hops instead of spinning.
  R6  hfmeta.fetch_file: the token reaches the configured endpoint and does
      NOT survive that endpoint's cross-host redirect.
  R7  hfmeta._get REFUSES to attach the token to a URL that is not the
      configured endpoint at all (no request is even sent).
  R8  fetch_truncated_ckpt.Fetcher: same keep/strip pair through its own
      retrying request path.
"""
import http.server
import importlib.util
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

FAILED = []

# The token below is a fixture, not a credential.
TOKEN = "hf_selftest_not_a_real_token_000000"


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:8]:
            print("        %s" % line)


class _Handler(http.server.BaseHTTPRequestHandler):
    """Redirects and an auth-echo, plus a request journal."""

    # set per-server after construction
    journal = None       # list of (path, auth_header_or_None)
    cross_to = None      # absolute URL for /cross

    def log_message(self, *a):                     # noqa: D102
        pass

    def do_GET(self):                              # noqa: N802
        auth = self.headers.get("Authorization")
        self.journal.append((self.path, auth))
        if self.path.startswith("/keep"):
            self.send_response(302)
            self.send_header("Location", "/echo")
            self.end_headers()
            return
        if self.path.startswith("/cross"):
            self.send_response(302)
            self.send_header("Location", self.cross_to)
            self.end_headers()
            return
        if self.path.startswith("/loop"):
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
            return
        body = ("auth=%s" % (auth or "NONE")).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host):
    journal = []
    handler = type("H", (_Handler,), {"journal": journal, "cross_to": None})
    srv = http.server.ThreadingHTTPServer((host, 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler, journal


def get(opener, url, token=TOKEN):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with opener.open(req, timeout=10) as resp:
        return resp.read().decode()


def main():
    from fidelity import common

    handler_cls = common.make_no_cross_origin_auth_handler()
    opener = urllib.request.build_opener(handler_cls())

    a_srv, a_handler, a_journal = serve("127.0.0.1")
    a_base = "http://127.0.0.1:%d" % a_srv.server_address[1]

    # A second ORIGIN: a different hostname for the same loopback interface.
    try:
        socket.getaddrinfo("localhost", None)
        b_host = "localhost"
    except OSError:
        b_host = "127.0.0.1"
    b_srv, b_handler, b_journal = serve("127.0.0.1")
    b_base = "http://%s:%d" % (b_host, b_srv.server_address[1])
    a_handler.cross_to = b_base + "/echo"

    # R1: same-origin redirect keeps the header.
    body = get(opener, a_base + "/keep")
    check("R1 same-origin redirect keeps Authorization",
          body == "auth=Bearer " + TOKEN, body)

    # R2: cross-host redirect strips it.
    b_journal.clear()
    body = get(opener, a_base + "/cross")
    check("R2 cross-host redirect strips Authorization",
          body == "auth=NONE" and b_journal and b_journal[-1][1] is None,
          "body=%r journal=%r" % (body, b_journal))

    # R3: same host, different port is a different origin.
    a_handler.cross_to = "http://127.0.0.1:%d/echo" % b_srv.server_address[1]
    b_journal.clear()
    body = get(opener, a_base + "/cross")
    check("R3 cross-port redirect strips Authorization",
          body == "auth=NONE", body)

    # R4: https->http downgrade, unit-level against the handler itself.
    req = urllib.request.Request("https://127.0.0.1/x")
    req.add_header("Authorization", "Bearer " + TOKEN)
    new = handler_cls().redirect_request(
        req, None, 302, "Found", {}, "http://127.0.0.1/echo")
    hdrs = {k.lower() for k in (new.headers or {})}
    hdrs |= {k.lower() for k in getattr(new, "unredirected_hdrs", {})}
    check("R4 https->http downgrade strips Authorization",
          new is not None and "authorization" not in hdrs, sorted(hdrs))
    # ... and the same-origin control keeps it, so R4 is not vacuous.
    req2 = urllib.request.Request("https://127.0.0.1/x")
    req2.add_header("Authorization", "Bearer " + TOKEN)
    new2 = handler_cls().redirect_request(
        req2, None, 302, "Found", {}, "https://127.0.0.1/echo")
    hdrs2 = {k.lower() for k in (new2.headers or {})}
    hdrs2 |= {k.lower() for k in getattr(new2, "unredirected_hdrs", {})}
    check("R4b same-origin control keeps Authorization",
          "authorization" in hdrs2, sorted(hdrs2))

    # R5: a loop terminates with an error, boundedly.
    a_journal.clear()
    try:
        get(opener, a_base + "/loop")
        bounded, hops = False, len([p for p, _ in a_journal if p == "/loop"])
    except urllib.error.HTTPError:
        hops = len([p for p, _ in a_journal if p == "/loop"])
        bounded = hops <= 15
    except urllib.error.URLError:
        hops = len([p for p, _ in a_journal if p == "/loop"])
        bounded = hops <= 15
    check("R5 redirect loop is bounded (%d hops)" % hops, bounded)

    # R6/R7: hfmeta, with the endpoint pointed at the stub.
    os.environ["HF_ENDPOINT"] = a_base
    os.environ["HF_TOKEN"] = TOKEN
    import fidelity.hfmeta as hfmeta
    importlib.reload(hfmeta)

    a_handler.cross_to = b_base + "/echo"
    a_journal.clear()
    b_journal.clear()
    body = hfmeta.fetch_file("cross", "x", revision="r").decode()
    endpoint_saw = next((auth for path, auth in a_journal
                         if path.startswith("/cross")), None)
    check("R6 hfmeta sends the token to the endpoint and not past its "
          "cross-host redirect",
          endpoint_saw == "Bearer " + TOKEN and body == "auth=NONE",
          "endpoint_saw=%r body=%r" % (endpoint_saw, body))

    try:
        hfmeta._get(b_base + "/echo")
        check("R7 hfmeta refuses to attach the token off-endpoint", False,
              "no exception")
    except hfmeta.HFError as exc:
        check("R7 hfmeta refuses to attach the token off-endpoint",
              "refusing to send" in str(exc), str(exc))

    # R8: the standalone truncation fetcher.
    spec = importlib.util.spec_from_file_location(
        "fetch_truncated_ckpt",
        str(ROOT / "engines" / "tools" / "fetch_truncated_ckpt.py"))
    ftc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ftc)
    fetcher = ftc.Fetcher("stub/repo", "r", TOKEN)
    fetcher.base = a_base + "/"
    a_journal.clear()
    b_journal.clear()
    kept = fetcher.whole("echo").decode()
    crossed = fetcher.whole("cross").decode()
    check("R8 truncation fetcher keeps auth on-origin, strips it across",
          kept == "auth=Bearer " + TOKEN and crossed == "auth=NONE",
          "kept=%r crossed=%r" % (kept, crossed))

    a_srv.shutdown()
    b_srv.shutdown()

    print()
    if FAILED:
        print("selftest_hf_redirect: %d FAILED" % len(FAILED))
        return 1
    print("selftest_hf_redirect: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
