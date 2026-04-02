#!/usr/bin/env python3
"""
tests/test_client.py — Concurrent HTTP client for the auto-orientation API.

Usage
-----
  # Single request (binary stream)
  python tests/test_client.py --url http://localhost:8000 --file ./tests/sample.jpg

  # Base64 JSON response
  python tests/test_client.py --url http://localhost:8000 --file ./tests/sample.jpg --base64

  # Concurrent load test
  python tests/test_client.py \
      --url http://localhost:8000 \
      --concurrency 10 \
      --total 100 \
      --warmup 2 \
      --output-dir ./output/

  # Custom workers (override server-side)
  WORKERS=8 ./deployment.sh
  python tests/test_client.py --concurrency 8 --total 200

Performance targets (8-core machine, --concurrency 8)
----------------------------------------------------
  Throughput  : ~4–6 images / second
  Latency p50: < 800 ms
  Latency p99: < 2 000 ms
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed — run: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed — run: pip install Pillow")
    sys.exit(1)


# ── Types ──────────────────────────────────────────────────────────────────────

class RequestResult(NamedTuple):
    filename: str
    success: bool
    status_code: int
    latency_ms: float
    size_in_bytes: int
    rotation_deg: float
    processing_time_ms: float
    request_time_ms: float
    exif_correction_deg: float
    onnx_angle_deg: float
    deskew_angle_deg: float
    error: Optional[str] = None


# ── Core HTTP client ──────────────────────────────────────────────────────────

def submit_image(
    url: str,
    image_path: Path,
    return_base64: bool = False,
) -> RequestResult:
    """Send one image to the auto-orient API and collect metrics."""
    t0 = time.perf_counter()

    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f.read())}
            data = {"return_base64": str(return_base64).lower()}

            response = requests.post(
                f"{url}/v1/image/auto-orient",
                files=files,
                data=data,
                timeout=120,
                stream=not return_base64,
            )
    except Exception as exc:
        return RequestResult(
            filename=image_path.name,
            success=False,
            status_code=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            size_in_bytes=0,
            rotation_deg=0.0,
            processing_time_ms=0.0,
            request_time_ms=0.0,
            exif_correction_deg=0.0,
            onnx_angle_deg=0.0,
            deskew_angle_deg=0.0,
            error=str(exc),
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    status_code = response.status_code

    if status_code != 200:
        return RequestResult(
            filename=image_path.name,
            success=False,
            status_code=status_code,
            latency_ms=latency_ms,
            size_in_bytes=0,
            rotation_deg=0.0,
            processing_time_ms=0.0,
            request_time_ms=0.0,
            exif_correction_deg=0.0,
            onnx_angle_deg=0.0,
            deskew_angle_deg=0.0,
            error=response.text[:200],
        )

    try:
        # Extract response headers
        rotation_deg = float(response.headers.get("X-Orientation-Corrected-Deg", "0"))
        proc_ms = float(response.headers.get("X-Processing-Time-Ms", "0"))
        req_ms = float(response.headers.get("X-Request-Time-Ms", "0"))
        exif_deg = float(response.headers.get("X-Exif-Correction-Deg", "0"))
        onnx_deg = float(response.headers.get("X-ONNX-Angle-Deg", "0"))
        deskew_deg = float(response.headers.get("X-Deskew-Angle-Deg", "0"))

        if return_base64:
            payload = response.json()
            result_bytes = base64.b64decode(payload["image_base64"])
            meta = payload.get("metadata", {})
            rotation_deg = meta.get("total_correction_deg", rotation_deg)
        else:
            result_bytes = response.content

        return RequestResult(
            filename=image_path.name,
            success=True,
            status_code=status_code,
            latency_ms=latency_ms,
            size_in_bytes=len(result_bytes),
            rotation_deg=rotation_deg,
            processing_time_ms=proc_ms,
            request_time_ms=req_ms,
            exif_correction_deg=exif_deg,
            onnx_angle_deg=onnx_deg,
            deskew_angle_deg=deskew_deg,
        )

    except Exception as exc:
        return RequestResult(
            filename=image_path.name,
            success=True,
            status_code=status_code,
            latency_ms=latency_ms,
            size_in_bytes=len(response.content),
            rotation_deg=0.0,
            processing_time_ms=0.0,
            request_time_ms=0.0,
            exif_correction_deg=0.0,
            onnx_angle_deg=0.0,
            deskew_angle_deg=0.0,
            error=f"Response parse error: {exc}",
        )


def load_test(
    url: str,
    image_paths: list[Path],
    concurrency: int,
    total_requests: int,
    warmup: int,
    output_dir: Optional[Path],
    return_base64: bool,
    progress_callback=None,
) -> list[RequestResult]:
    """
    Run a concurrent load test across `total_requests` image submissions.

    image_paths is cycled if total_requests > len(image_paths).
    """
    results: list[RequestResult] = []
    results_lock = threading.Lock()
    completed = 0
    completed_lock = threading.Lock()

    def _worker(idx: int) -> RequestResult:
        path = image_paths[idx % len(image_paths)]
        result = submit_image(url, path, return_base64=return_base64)

        if output_dir and result.success:
            out_path = output_dir / f"oriented_{idx:04d}_{path.name}"
            with open(out_path, "wb") as f:
                if return_base64:
                    payload = requests.post(
                        f"{url}/v1/image/auto-orient",
                        files={"file": (path.name, open(path, "rb").read())},
                        data={"return_base64": "true"},
                        timeout=120,
                    ).json()
                    f.write(base64.b64decode(payload["image_base64"]))
                else:
                    resp = requests.post(
                        f"{url}/v1/image/auto-orient",
                        files={"file": (path.name, open(path, "rb").read())},
                        timeout=120,
                    )
                    f.write(resp.content)

        with completed_lock:
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed)

        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_worker, i)
            for i in range(total_requests)
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = RequestResult(
                    filename="?", success=False, status_code=0,
                    latency_ms=0, size_in_bytes=0,
                    rotation_deg=0, processing_time_ms=0, request_time_ms=0,
                    exif_correction_deg=0, onnx_angle_deg=0, deskew_angle_deg=0,
                    error=str(exc),
                )
            with results_lock:
                results.append(result)

    return results


# ── Progress bar ───────────────────────────────────────────────────────────────

def _make_progress_bar(total: int, width: int = 40):
    done = [0]
    lock = threading.Lock()
    t0 = time.perf_counter()

    def _update(n: int):
        with lock:
            done[0] = n
            pct = done[0] / total
            filled = int(width * pct)
            bar = "█" * filled + "░" * (width - filled)
            elapsed = time.perf_counter() - t0
            rate = done[0] / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r[{bar}] {done[0]:>{len(str(total))}}/{total}  "
                f"{pct*100:5.1f}%  {rate:6.2f} req/s   "
            )
            sys.stdout.flush()

    return _update


# ── Pretty report ─────────────────────────────────────────────────────────────

def print_report(results: list[RequestResult], total_requests: int, concurrency: int):
    elapsed = time.time() - _report_t0  # set by caller
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    failures_text = [r for r in results if r.error]
    latencies = [r.latency_ms for r in successes]
    proc_times = [r.processing_time_ms for r in successes if r.processing_time_ms > 0]
    rotations = [r.rotation_deg for r in successes]
    sizes_in = [
        Path(r.filename).stat().st_size
        for r in successes
        if Path(r.filename).exists()
    ]

    throughput = len(successes) / elapsed if elapsed > 0 else 0

    W = "\033[0m"; G = "\033[0;32m"; Y = "\033[1;33m"; R = "\033[0;31m"
    B = "\033[1m"

    def stat_row(label: str, value: str, ok: bool = True):
        c = G if ok else R
        return f"  {c}{label:<32} {B}{value}{W}"

    print()
    print(f"{B}{'─'*70}{W}")
    print(f"  {B}LOAD TEST REPORT — {concurrency} concurrent workers, {total_requests} total requests{W}")
    print(f"{B}{'─'*70}{W}")
    print(stat_row("Total requests",        str(total_requests)))
    print(stat_row("Succeeded",             f"{len(successes)}  ({len(successes)/total_requests*100:.1f}%)",
                    ok=len(successes) > 0))
    print(stat_row("Failed",                 f"{len(failures)}  ({len(failures)/total_requests*100:.1f}%)",
                    ok=len(failures) == 0))

    if latencies:
        print()
        print(f"  {B}Latency (ms){W}")
        for label, val in [
            ("p50",  f"{statistics.median(latencies):.1f}"),
            ("p90",  f"{statistics.quantiles(latencies, n=10)[8]:.1f}"),
            ("p99",  f"{statistics.quantiles(latencies, n=100)[98]:.1f}"),
            ("mean", f"{statistics.mean(latencies):.1f}"),
            ("min",  f"{min(latencies):.1f}"),
            ("max",  f"{max(latencies):.1f}"),
        ]:
            print(stat_row(label, val))

    if proc_times:
        print()
        print(f"  {B}Processing time (ms){W}")
        for label, val in [
            ("p50",  f"{statistics.median(proc_times):.1f}"),
            ("p99",  f"{statistics.quantiles(proc_times, n=100)[98]:.1f}"),
            ("mean", f"{statistics.mean(proc_times):.1f}"),
        ]:
            print(stat_row(label, val))

    print()
    print(f"  {B}Throughput{W}")
    print(stat_row("Effective throughput", f"{throughput:.2f} req/s"))

    if rotations:
        nonzero = [r for r in rotations if abs(r) > 0.5]
        print()
        print(f"  {B}Rotation corrections{W}")
        print(stat_row("Images corrected (>0.5°)", f"{len(nonzero)} / {len(rotations)}"))

    if sizes_in:
        print()
        print(stat_row("Avg input size (KB)",  f"{statistics.mean(sizes_in)/1024:.0f}"))
        out_sizes = [r.size_in_bytes for r in successes]
        print(stat_row("Avg output size (KB)", f"{statistics.mean(out_sizes)/1024:.0f}"))

    if failures_text:
        print()
        print(f"  {R}{B}Error samples (first 5):{W}")
        for r in failures_text[:5]:
            print(f"    {R}• {r.filename}: {r.error[:120]}{W}")

    print()
    print(f"{B}{'─'*70}{W}")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concurrent test client for the auto-orientation API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",      default=os.getenv("API_URL", "http://localhost:8000"),
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--file",     type=Path,
                        help="Single image file to upload")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent requests for load test (default: 4)")
    parser.add_argument("--total",    type=int, default=20,
                        help="Total requests for load test (default: 20)")
    parser.add_argument("--warmup",   type=int, default=0,
                        help="Warm-up requests before measuring (default: 0)")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path,
                        help="Directory to save corrected images")
    parser.add_argument("--base64",   action="store_true",
                        help="Use base64 JSON response mode")
    parser.add_argument("--json",     type=Path,
                        help="JSON file listing image paths (one per line)")

    args = parser.parse_args()

    # Collect image paths
    image_paths: list[Path] = []
    if args.file:
        if not args.file.exists():
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        image_paths = [args.file]
    elif args.json:
        if not args.json.exists():
            print(f"ERROR: JSON not found: {args.json}", file=sys.stderr)
            sys.exit(1)
        with open(args.json) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                image_paths = [Path(p) for p in data]
            else:
                image_paths = [Path(k) for k, v in data.items() if v]
    else:
        # Try to find a sample image
        default_sample = Path(__file__).parent / "sample.jpg"
        if default_sample.exists():
            image_paths = [default_sample]
        else:
            print("ERROR: no --file or --json specified and no sample.jpg found.", file=sys.stderr)
            print("       Place a test image at tests/sample.jpg or use --file <path>.", file=sys.stderr)
            sys.exit(1)

    image_paths = [p for p in image_paths if p.exists()]
    if not image_paths:
        print("ERROR: no valid image files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Testing against: {args.url}")
    print(f"Image files    : {len(image_paths)}  ({', '.join(p.name for p in image_paths[:3])})"
          f"{' ...' if len(image_paths) > 3 else ''}")

    # Health check
    try:
        r = requests.get(f"{args.url}/health", timeout=10)
        health = r.json()
        print(f"Health status  : {health.get('status','?')}  "
              f"(workers confirmed, pid={health.get('pid','?')})")
    except Exception as exc:
        print(f"WARN: health check failed — server may not be running: {exc}", file=sys.stderr)

    # Warm-up
    if args.warmup > 0:
        print(f"Warming up with {args.warmup} request(s) …")
        with ThreadPoolExecutor(max_workers=min(args.warmup, 4)) as pool:
            list(pool.map(
                lambda p: submit_image(args.url, p, args.base64),
                image_paths * ((args.warmup // len(image_paths)) + 1)
            ))[:args.warmup]

    # Run load test
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to: {args.output_dir}")

    print(f"\nStarting load test — concurrency={args.concurrency}, total={args.total}")
    progress = _make_progress_bar(args.total)

    global _report_t0  # noqa: PLW0603
    _report_t0 = time.time()

    results = load_test(
        url=args.url,
        image_paths=image_paths,
        concurrency=args.concurrency,
        total_requests=args.total,
        warmup=0,
        output_dir=args.output_dir,
        return_base64=args.base64,
        progress_callback=progress,
    )

    sys.stdout.write("\n")  # newline after progress bar
    print_report(results, args.total, args.concurrency)


if __name__ == "__main__":
    main()
