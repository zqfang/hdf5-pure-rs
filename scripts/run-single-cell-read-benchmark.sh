#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/release/examples/perf_compare"
FILE="${1:-$ROOT/tests/data/real_world/10x_pbmc_1k_v3_filtered_feature_bc_matrix.h5}"
DURATION_SECONDS="${2:-3}"

if [[ ! -f "$FILE" ]]; then
    echo "missing fixture: $FILE" >&2
    echo "populate fixtures with: scripts/download-real-world-fixtures.py" >&2
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    cargo build --release --example perf_compare --manifest-path "$ROOT/Cargo.toml" >/dev/null
fi

echo "benchmark_file=$FILE"
echo "seconds=$DURATION_SECONDS"
echo "rust:"
"$BIN" bench-read-i32 "$FILE" matrix/data "$DURATION_SECONDS"
"$BIN" bench-read-i64 "$FILE" matrix/indices "$DURATION_SECONDS"
"$BIN" bench-read-i64 "$FILE" matrix/indptr "$DURATION_SECONDS"

if command -v python3 >/dev/null 2>&1; then
    echo "h5py:"
    python3 - "$FILE" "$DURATION_SECONDS" <<'PY'
import sys
import time

import h5py

path = sys.argv[1]
seconds = float(sys.argv[2])
datasets = ["matrix/data", "matrix/indices", "matrix/indptr"]

with h5py.File(path, "r") as f:
    print(f"h5py_version={h5py.__version__} hdf5_version={h5py.version.hdf5_version}")
    for name in datasets:
        ds = f[name]
        end = time.perf_counter() + seconds
        n = 0
        total = 0.0
        best = float("inf")
        checksum = 0
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            values = ds[:]
            dt = (time.perf_counter() - t0) * 1000.0
            total += dt
            best = min(best, dt)
            checksum = int(values.sum())
            n += 1
        print(
            f"{name} iterations={n} total_ms={total:.3f} "
            f"avg_ms={total / n:.3f} best_ms={best:.3f} checksum={checksum}"
        )
PY
fi
