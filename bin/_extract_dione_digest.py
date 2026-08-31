#!/usr/bin/env python3
"""Extract `tensor_digest`'s embedded python out of engines/tools/measure_dione.sh.

Test-support for bin/selftest_shell_guards.sh (SH-06 / CC-09). The guard must drive
the SHIPPED snippet, not a copy of it: the defect was that a digest over zero
`logits/*.safetensors` files returned the sha256 of nothing -- a constant, equal to
itself -- so the Dione escalation trigger could not fire, and two runs that were never
compared were reported as bitwise identical.

    _extract_dione_digest.py <measure_dione.sh> <out.py>
    _extract_dione_digest.py --write-window <path.safetensors>

The second form writes a one-tensor safetensors file so the guard can also prove the
digest still ANSWERS for a real window; a refusal that refuses everything is not a fix.
"""

import json
import re
import struct
import sys

SNIPPET = re.compile(
    r"tensor_digest\(\) \{ \$V/python - \"\$1\" <<'PY'\n(?P<body>.*?)\nPY\n\}",
    re.S,
)


def write_window(path):
    payload = bytes(range(256)) * 8                       # 2048 bytes -> [4, 256] F16
    header = {"logits": {"dtype": "F16", "shape": [4, 256],
                         "data_offsets": [0, len(payload)]}}
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(payload)


def main(argv):
    if len(argv) == 3 and argv[1] == "--write-window":
        write_window(argv[2])
        return 0
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    text = open(argv[1], encoding="utf-8").read()
    found = SNIPPET.search(text)
    if not found:
        sys.stderr.write("tensor_digest snippet not found in %s\n" % argv[1])
        return 1
    with open(argv[2], "w", encoding="utf-8") as fh:
        fh.write(found.group("body") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
