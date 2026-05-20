#!/usr/bin/env python3
import hashlib
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def load_source(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as f:
        obs = {}
        for name, ds in f["obs"].items():
            value = ds[()]
            if getattr(value, "dtype", None) is not None and value.dtype.kind in {"S", "O"}:
                value = ds.asstr()[()]
            obs[name] = value
        return {
            "data": f["rna/data"][()],
            "indices": f["rna/indices"][()],
            "indptr": f["rna/indptr"][()],
            "shape": f["rna/shape"][()],
            "obs_names": f["rna/obs_names"].asstr()[()],
            "var_names": f["rna/var_names"].asstr()[()],
            "obs": obs,
        }


def write_copy(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as f:
        rna = f.create_group("rna")
        rna.create_dataset("data", data=payload["data"], chunks=True)
        rna.create_dataset("indices", data=payload["indices"], chunks=True)
        rna.create_dataset("indptr", data=payload["indptr"])
        rna.create_dataset("shape", data=payload["shape"])
        rna.create_dataset("obs_names", data=payload["obs_names"], dtype=STRING_DTYPE)
        rna.create_dataset("var_names", data=payload["var_names"], dtype=STRING_DTYPE)

        obs = f.create_group("obs")
        for name, value in payload["obs"].items():
            array = np.asarray(value)
            if array.dtype.kind in {"U", "S", "O"}:
                obs.create_dataset(name, data=array.astype(object), dtype=STRING_DTYPE)
            else:
                obs.create_dataset(name, data=array)


def output_for(prefix: Path, iteration: int) -> Path:
    return prefix.with_name(prefix.name + f"_{iteration}.h5")


def digest_dataset(ds: h5py.Dataset) -> str:
    h = hashlib.sha256()
    value = ds[()]
    if getattr(value, "dtype", None) is not None and value.dtype.kind in {"S", "O"}:
        for item in ds.asstr()[()].reshape(-1):
            h.update(item.encode("utf-8"))
            h.update(b"\0")
    else:
        arr = np.asarray(value)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def collect_digests(path: Path) -> dict[str, str]:
    digests = {}
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                digests[name] = digest_dataset(obj)

        f.visititems(visitor)
    return digests


def verify(paths: list[Path]) -> None:
    baseline = collect_digests(paths[0])
    for path in paths[1:]:
        current = collect_digests(path)
        if current != baseline:
            missing = sorted(set(baseline) - set(current))
            extra = sorted(set(current) - set(baseline))
            changed = sorted(k for k in baseline.keys() & current.keys() if baseline[k] != current[k])
            raise SystemExit(
                f"parity failed for {path}: missing={missing} extra={extra} changed={changed[:20]}"
            )
    print("parity=ok files=" + ",".join(str(p) for p in paths))


def bench(source: Path, prefix: Path, iterations: int) -> None:
    t0 = time.perf_counter()
    payload = load_source(source)
    preload_ms = (time.perf_counter() - t0) * 1000.0
    print(f"preload_ms={preload_ms:.3f}")

    times = []
    for iteration in range(iterations):
        path = output_for(prefix, iteration)
        t0 = time.perf_counter()
        write_copy(payload, path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        size = os.stat(path).st_size
        print(f"write iteration={iteration} ms={elapsed_ms:.3f} bytes={size} path={path}")
        times.append(elapsed_ms)
    print(f"write_best_ms={min(times):.3f} write_avg_ms={sum(times) / len(times):.3f}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: seurat_h5py_write_bench.py <bench|verify> ...")
    mode = sys.argv[1]
    if mode == "bench":
        source = Path(sys.argv[2])
        prefix = Path(sys.argv[3])
        iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        bench(source, prefix, iterations)
    elif mode == "verify":
        verify([Path(arg) for arg in sys.argv[2:]])
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
