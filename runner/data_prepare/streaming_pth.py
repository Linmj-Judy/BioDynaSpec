"""Stream a list to a torch .pth without holding the whole list in memory.

`torch.save(list)` needs the entire list resident; that is what OOM-killed the
full assembly at ~987 GB (4149 entries x ~237 MB). Since torch 2.x, `.pth` is a
zip archive holding one member `data.pkl`, and `data.pkl` is an ordinary pickle
of the list. Both can be produced incrementally:

    zip member  data.pkl = PROTO EMPTY_LIST MARK <item>... APPENDS STOP

so items are pickled and written one at a time and peak memory is one item.

The output is read back by plain `torch.load`, byte-compatible with `torch.save`
apart from the optional `serialization_id` record. Round-trip equality over
nested numpy arrays / lists / dicts / tensors is covered by
`runner/data_prepare/_test_stream_pth.py`.
"""

import pickle
import zipfile


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
