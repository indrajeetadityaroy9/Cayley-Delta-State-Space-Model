"""Profiling utilities for KSSM training and inference.

Provides lightweight tools for:
- CUDA timing with synchronization
- Memory tracking (peak/current)
- Per-layer profiling
"""

import functools
import time
from contextlib import contextmanager
from typing import Callable

import torch


@contextmanager
def cuda_timer(name: str = "operation", verbose: bool = True):
    """Context manager for accurate CUDA timing.

    Uses CUDA events for precise GPU timing with synchronization.

    Args:
        name: Name of the operation being timed.
        verbose: If True, print timing results.

    Yields:
        dict: Dictionary that will contain 'elapsed_ms' after context exits.

    Example:
        >>> with cuda_timer("forward pass") as t:
        ...     output = model(input)
        >>> print(f"Took {t['elapsed_ms']:.2f} ms")
    """
    result = {"elapsed_ms": 0.0}

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        try:
            yield result
        finally:
            end_event.record()
            torch.cuda.synchronize()
            result["elapsed_ms"] = start_event.elapsed_time(end_event)

            if verbose:
                print(f"[TIMER] {name}: {result['elapsed_ms']:.3f} ms")
    else:
        # CPU fallback
        start_time = time.perf_counter()
        try:
            yield result
        finally:
            result["elapsed_ms"] = (time.perf_counter() - start_time) * 1000
            if verbose:
                print(f"[TIMER] {name}: {result['elapsed_ms']:.3f} ms")


class MemoryTracker:
    """Track GPU memory usage during operations.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.start()
        >>> # ... do some GPU work ...
        >>> stats = tracker.stop()
        >>> print(f"Peak memory: {stats['peak_mb']:.1f} MB")
    """

    def __init__(self):
        self.start_allocated = 0
        self.start_reserved = 0
        self.tracking = False

    def start(self):
        """Start memory tracking."""
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start_allocated = torch.cuda.memory_allocated()
        self.start_reserved = torch.cuda.memory_reserved()
        self.tracking = True

    def stop(self) -> dict:
        """Stop tracking and return memory statistics.

        Returns:
            dict with keys:
                - current_mb: Current allocated memory in MB
                - peak_mb: Peak allocated memory in MB
                - delta_mb: Change in allocated memory since start
                - reserved_mb: Total reserved memory in MB
        """
        if not torch.cuda.is_available() or not self.tracking:
            return {
                "current_mb": 0,
                "peak_mb": 0,
                "delta_mb": 0,
                "reserved_mb": 0,
            }

        torch.cuda.synchronize()
        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()

        self.tracking = False

        return {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "delta_mb": (current - self.start_allocated) / (1024 * 1024),
            "reserved_mb": reserved / (1024 * 1024),
        }


@contextmanager
def memory_tracker(name: str = "operation", verbose: bool = True):
    """Context manager for memory tracking.

    Args:
        name: Name of the operation being tracked.
        verbose: If True, print memory statistics.

    Yields:
        dict: Dictionary that will contain memory stats after context exits.

    Example:
        >>> with memory_tracker("forward pass") as mem:
        ...     output = model(input)
        >>> print(f"Peak: {mem['peak_mb']:.1f} MB")
    """
    tracker = MemoryTracker()
    result = {}

    tracker.start()
    try:
        yield result
    finally:
        stats = tracker.stop()
        result.update(stats)

        if verbose:
            print(
                f"[MEMORY] {name}: "
                f"current={stats['current_mb']:.1f}MB, "
                f"peak={stats['peak_mb']:.1f}MB, "
                f"delta={stats['delta_mb']:+.1f}MB"
            )


def layer_profiler(name: str = None):
    """Decorator for profiling layer forward/backward passes.

    Args:
        name: Optional name for the layer (defaults to function name).

    Example:
        >>> class MyModule(nn.Module):
        ...     @layer_profiler("MyModule.forward")
        ...     def forward(self, x):
        ...         return self.layer(x)
    """

    def decorator(func: Callable) -> Callable:
        layer_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with cuda_timer(layer_name, verbose=False) as timing:
                result = func(*args, **kwargs)

            # Store timing in a global registry for later analysis
            if not hasattr(wrapper, "_timings"):
                wrapper._timings = []
            wrapper._timings.append(timing["elapsed_ms"])

            return result

        # Add method to get timing statistics
        def get_stats():
            if not hasattr(wrapper, "_timings") or not wrapper._timings:
                return {"count": 0, "mean_ms": 0, "total_ms": 0}
            timings = wrapper._timings
            return {
                "count": len(timings),
                "mean_ms": sum(timings) / len(timings),
                "total_ms": sum(timings),
                "min_ms": min(timings),
                "max_ms": max(timings),
            }

        def reset_stats():
            wrapper._timings = []

        wrapper.get_stats = get_stats
        wrapper.reset_stats = reset_stats

        return wrapper

    return decorator


class ProfilerContext:
    """Comprehensive profiler for training steps.

    Tracks timing and memory for different phases of training.

    Example:
        >>> profiler = ProfilerContext()
        >>> with profiler.phase("forward"):
        ...     output = model(input)
        >>> with profiler.phase("backward"):
        ...     loss.backward()
        >>> profiler.report()
    """

    def __init__(self):
        self.phases = {}
        self.current_phase = None

    @contextmanager
    def phase(self, name: str):
        """Track a specific phase of computation.

        Args:
            name: Name of the phase (e.g., "forward", "backward", "optimizer").
        """
        if name not in self.phases:
            self.phases[name] = {"times_ms": [], "memory_mb": []}

        self.current_phase = name

        tracker = MemoryTracker()
        tracker.start()

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        else:
            start_time = time.perf_counter()

        try:
            yield
        finally:
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

            mem_stats = tracker.stop()

            self.phases[name]["times_ms"].append(elapsed_ms)
            self.phases[name]["memory_mb"].append(mem_stats["peak_mb"])
            self.current_phase = None

    def report(self) -> dict:
        """Generate profiling report.

        Returns:
            dict: Profiling statistics for each phase.
        """
        report = {}
        total_time = 0

        for name, data in self.phases.items():
            if not data["times_ms"]:
                continue

            times = data["times_ms"]
            memory = data["memory_mb"]

            phase_stats = {
                "count": len(times),
                "time_mean_ms": sum(times) / len(times),
                "time_total_ms": sum(times),
                "time_min_ms": min(times),
                "time_max_ms": max(times),
                "memory_peak_mb": max(memory) if memory else 0,
            }
            report[name] = phase_stats
            total_time += sum(times)

        # Add percentages
        if total_time > 0:
            for name in report:
                report[name]["time_pct"] = (
                    report[name]["time_total_ms"] / total_time * 100
                )

        report["_total_time_ms"] = total_time
        return report

    def print_report(self):
        """Print formatted profiling report."""
        report = self.report()
        total = report.pop("_total_time_ms", 0)

        print("\n" + "=" * 60)
        print("PROFILING REPORT")
        print("=" * 60)

        for name, stats in sorted(report.items(), key=lambda x: -x[1]["time_total_ms"]):
            print(
                f"\n{name}:"
                f"\n  Time: {stats['time_mean_ms']:.2f}ms avg "
                f"({stats['time_pct']:.1f}%)"
                f"\n  Calls: {stats['count']}"
                f"\n  Memory peak: {stats['memory_peak_mb']:.1f}MB"
            )

        print(f"\n{'=' * 60}")
        print(f"Total time: {total:.2f}ms")
        print("=" * 60 + "\n")

    def reset(self):
        """Reset all profiling data."""
        self.phases = {}
        self.current_phase = None
