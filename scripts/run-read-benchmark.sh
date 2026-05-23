#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/release/examples/perf_compare"
FILE="${1:-$ROOT/target/perf_large_simple.h5}"
LEN="${2:-32000000}"
CHUNK="${3:-1000000}"
DEFLATE="${4:-1}"
DURATION_SECONDS="${5:-5}"

echo "benchmark_file=$FILE"
echo "len=$LEN chunk=$CHUNK deflate=$DEFLATE seconds=$DURATION_SECONDS"

if [[ ! -x "$BIN" ]]; then
    cargo build --release --example perf_compare --manifest-path "$ROOT/Cargo.toml" >/dev/null
fi

if [[ ! -f "$FILE" ]]; then
    "$BIN" write "$FILE" "$LEN" "$CHUNK" "$DEFLATE"
fi

echo "rust:"
"$BIN" bench-read "$FILE" "$DURATION_SECONDS"

if command -v python3 >/dev/null 2>&1; then
    echo "h5py:"
    python3 - "$FILE" "$DURATION_SECONDS" <<'PY'
import sys
import time

import h5py

path = sys.argv[1]
seconds = float(sys.argv[2])

with h5py.File(path, "r") as f:
    ds = f["data"]
    end = time.perf_counter() + seconds
    n = 0
    total = 0.0
    best = float("inf")
    checksum = 0.0
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        x = ds[:]
        dt = (time.perf_counter() - t0) * 1000.0
        total += dt
        best = min(best, dt)
        checksum = float(x.sum())
        n += 1
    print(
        f"benchmark_read iterations={n} total_ms={total:.3f} "
        f"avg_ms={total/n:.3f} best_ms={best:.3f} checksum={checksum:.1f}"
    )
PY
fi
