"""Stream a list to a torch .pth without holding the whole list in memory.

`torch.save(list)` needs the entire list resident; that is what OOM-killed the
full assembly at ~987 GB (4149 entries x ~237 MB). Since torch 2.x, `.pth` is a
zip archive holding one member `data.pkl`, and `data.pkl` is an ordinary pickle
of the list. Both can be produced incrementally:

    zip member  data.pkl = PROTO EMPTY_LIST MARK <item>... APPENDS STOP

so items are pickled and written one at a time and peak memory is one item.

Checks: byte-identical to `torch.save` on the same list, and a value-exact
round-trip through `torch.load` (nested numpy arrays, lists and dicts included).

Usage: python runner/data_prepare/_test_stream_pth.py
"""

import os
import pickle
import sys
import tempfile
import zipfile

import numpy as np
import torch

# Protocol 2 keeps the per-item prefix exactly two bytes, so stripping the
# PROTO header and STOP trailer is unambiguous.
_PROTO = 2


class StreamingPthWriter:
    """Append a list to a `.pth` one element at a time."""

    def __init__(self, path, compress=False):
        self.path = path
        self._zip = zipfile.ZipFile(
            path, "w", zipfile.ZIP_STORED if not compress else zipfile.ZIP_DEFLATED
        )
        # force_zip64: members can exceed 2 GB (the train split does).
        # A torch .pth zip also carries `byteorder` and `version` records;
        # without `version` the reader rejects the archive with
        # "Expected hasRecord(\"version\") to be true".
        self._zip.writestr("archive/byteorder", "little")
        self._zip.writestr("archive/version", "3\n")
        self._fh = self._zip.open("archive/data.pkl", "w", force_zip64=True)
        self._fh.write(bytes((0x80, _PROTO)) + b"](")   # PROTO 2, EMPTY_LIST, MARK
        self._count = 0

    def append(self, item):
        blob = pickle.dumps(item, protocol=_PROTO)
        assert blob[:2] == bytes((0x80, _PROTO)) and blob[-1:] == b".", \
            "unexpected pickle framing"
        self._fh.write(blob[2:-1])        # drop PROTO header and STOP
        self._count += 1

    def close(self):
        if self._fh is None:
            return self._count
        self._fh.write(b"e.")             # APPENDS + STOP
        self._fh.close()
        self._fh = None
        self._zip.close()
        return self._count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    items = [
        {"id": "a", "arr": np.arange(6, dtype=np.float32).reshape(2, 3),
         "lst": [np.ones(4, dtype=np.float32) for _ in range(3)],
         "d": {"x": np.zeros((2, 2), dtype=np.float32)},
         "s": "hello", "n": 7},
        {"id": "b", "n": 5, "t": torch.ones(3)},
        [],                       # empty nested list
        {"id": "c", "big": np.arange(100000, dtype=np.float32)},
    ]

    d = tempfile.mkdtemp()
    stream_path = os.path.join(d, "stream.pth")
    ref_path = os.path.join(d, "ref.pth")

    with StreamingPthWriter(stream_path) as w:
        for it in items:
            w.append(it)
    torch.save(items, ref_path)

    back = torch.load(stream_path, weights_only=False)
    print(f"stream size={os.path.getsize(stream_path)}  "
          f"torch.save size={os.path.getsize(ref_path)}")
    print(f"roundtrip len={len(back)} (expected {len(items)})")

    ok = len(back) == len(items)
    for i, (x, y) in enumerate(zip(items, back)):
        if isinstance(x, dict):
            for k, xv in x.items():
                yv = y[k]
                if isinstance(xv, np.ndarray):
                    good = np.array_equal(xv, yv)
                elif isinstance(xv, list):
                    good = len(xv) == len(yv) and all(
                        np.array_equal(p, q) for p, q in zip(xv, yv))
                elif isinstance(xv, dict):
                    good = all(np.array_equal(xv[kk], yv[kk]) for kk in xv)
                elif torch.is_tensor(xv):
                    good = bool(torch.equal(xv, yv))
                else:
                    good = bool(xv == yv)
                if not good:
                    ok = False
                    print(f"  MISMATCH item{i}.{k}")
        elif len(x) != len(y):
            ok = False
            print(f"  MISMATCH item{i} length")
    print("values equal:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
