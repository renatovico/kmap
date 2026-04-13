"""C-accelerated circuit graph evaluator.

Replaces NumPy as the compute backend.  The evaluation loop runs in
Python (trivial overhead) but every tensor operation dispatches to
the compiled C shared library ``_circuit_eval.so``.

Usage::

    from kllm.circuit_executor import evaluate_c
    values = evaluate_c(graph, inputs)
    # values is identical to circuit_graph.evaluate(graph, inputs)
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from kllm.circuit_graph import CircuitGraph, Op

# ---------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------

_LIB: ctypes.CDLL | None = None
_CSRC_DIR = Path(__file__).resolve().parent.parent.parent / "csrc"


def _lib_path() -> Path:
    """Return path to the compiled shared library."""
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return _CSRC_DIR / f"_circuit_eval{suffix}"


def _compile_if_needed() -> Path:
    """Compile the C source if the .so is missing or older than the source."""
    src = _CSRC_DIR / "_circuit_eval.c"
    lib = _lib_path()
    if lib.exists() and lib.stat().st_mtime >= src.stat().st_mtime:
        return lib
    # Also check for .so on macOS (we compile with -shared)
    lib_so = _CSRC_DIR / "_circuit_eval.so"
    if lib_so.exists() and lib_so.stat().st_mtime >= src.stat().st_mtime:
        return lib_so

    cmd = [
        "cc", "-O3", "-shared", "-fPIC", "-march=native",
        "-o", str(lib_so), str(src), "-lm",
    ]
    subprocess.check_call(cmd)
    return lib_so


def _get_lib() -> ctypes.CDLL:
    """Load the C library, compiling if necessary."""
    global _LIB
    if _LIB is not None:
        return _LIB

    # Try .so first (works on both Linux and macOS)
    lib_so = _CSRC_DIR / "_circuit_eval.so"
    if lib_so.exists():
        lib_path = lib_so
    else:
        lib_path = _compile_if_needed()

    _LIB = ctypes.CDLL(str(lib_path))
    _setup_signatures(_LIB)
    return _LIB


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Declare ctypes function signatures."""
    F = ctypes.c_float
    FP = ctypes.POINTER(F)
    IP = ctypes.POINTER(ctypes.c_int)
    I = ctypes.c_int

    # Binary ops (all same signature)
    for name in ["ceval_add", "ceval_sub", "ceval_mul", "ceval_div",
                  "ceval_max", "ceval_cmp_le"]:
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [FP, FP, IP, I, FP, IP, I, IP, I]

    # Mux
    lib.ceval_mux.restype = None
    lib.ceval_mux.argtypes = [FP, FP, FP, FP, I]

    # Unary ops
    for name in ["ceval_neg", "ceval_abs", "ceval_square",
                  "ceval_silu", "ceval_exp", "ceval_rsqrt",
                  "ceval_cos", "ceval_sin"]:
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [FP, FP, I]

    # Matmul
    lib.ceval_matmul.restype = None
    lib.ceval_matmul.argtypes = [FP, FP, IP, I, FP, IP, I, IP, I]

    # Reductions
    for name in ["ceval_sum", "ceval_max_reduce", "ceval_mean"]:
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [FP, FP, IP, I, I]

    # Argmax (output is int*)
    lib.ceval_argmax.restype = None
    lib.ceval_argmax.argtypes = [
        ctypes.POINTER(ctypes.c_int), FP, IP, I, I]

    # Transpose
    lib.ceval_transpose.restype = None
    lib.ceval_transpose.argtypes = [FP, FP, IP, I, IP]

    # Repeat
    lib.ceval_repeat.restype = None
    lib.ceval_repeat.argtypes = [FP, FP, IP, I, I, I]

    # Slice
    lib.ceval_slice.restype = None
    lib.ceval_slice.argtypes = [FP, FP, IP, I, IP, IP, IP, IP, I]

    # Copy
    lib.ceval_copy.restype = None
    lib.ceval_copy.argtypes = [FP, FP, I]


# ---------------------------------------------------------------
# Shape utilities (pure Python — no NumPy)
# ---------------------------------------------------------------

def _broadcast_shapes(a_shape: tuple, b_shape: tuple) -> tuple:
    """Compute NumPy-compatible broadcast output shape."""
    ndim = max(len(a_shape), len(b_shape))
    out = []
    for i in range(ndim):
        ai = a_shape[len(a_shape) - 1 - i] if i < len(a_shape) else 1
        bi = b_shape[len(b_shape) - 1 - i] if i < len(b_shape) else 1
        if ai != bi and ai != 1 and bi != 1:
            raise ValueError(f"Incompatible shapes: {a_shape} vs {b_shape}")
        out.append(max(ai, bi))
    return tuple(reversed(out))


def _matmul_shape(a_shape: tuple, b_shape: tuple) -> tuple:
    """Compute output shape of matmul."""
    if len(a_shape) < 2 or len(b_shape) < 2:
        # 1-D cases: not used by our compiler but handle for correctness
        if len(a_shape) == 1 and len(b_shape) == 1:
            return ()
        elif len(a_shape) == 1:
            return b_shape[:-2] + (b_shape[-1],)
        else:
            return a_shape[:-1]

    m = a_shape[-2]
    n = b_shape[-1]

    # Broadcast batch dims
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    if a_batch or b_batch:
        batch = _broadcast_shapes(a_batch or (1,), b_batch or (1,))
        return batch + (m, n)
    return (m, n)


def _reduce_shape(shape: tuple, axis: int, keepdims: bool) -> tuple:
    """Compute output shape of a reduction."""
    if keepdims:
        return shape[:axis] + (1,) + shape[axis + 1:]
    return shape[:axis] + shape[axis + 1:]


def _to_c_float(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    """Get a ctypes float pointer to a contiguous float32 array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _to_c_int_array(values: tuple | list) -> ctypes.Array:
    """Create a ctypes int array from a Python sequence."""
    n = len(values)
    arr = (ctypes.c_int * n)(*values)
    return arr


# ---------------------------------------------------------------
# LUT dispatch
# ---------------------------------------------------------------

_LUT_FN_MAP = {
    "silu": "ceval_silu",
    "exp": "ceval_exp",
    "rsqrt": "ceval_rsqrt",
    "cos": "ceval_cos",
    "sin": "ceval_sin",
}

# ---------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------


def evaluate_c(graph: CircuitGraph,
               inputs: dict[int, np.ndarray] | None = None,
               ) -> dict[int, np.ndarray]:
    """Evaluate a circuit graph using the C tensor ops library.

    Same API and output as ``circuit_graph.evaluate()``, but all
    tensor math dispatches to compiled C code instead of NumPy.

    Parameters
    ----------
    graph : CircuitGraph
        The circuit to evaluate.
    inputs : dict mapping input node ID → NumPy array
        Values for ``INPUT`` nodes.

    Returns
    -------
    dict mapping node ID → NumPy array
        Computed value for every node.
    """
    lib = _get_lib()
    inputs = inputs or {}
    values: dict[int, np.ndarray] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]
        inp = [values[i] for i in node.inputs]

        if node.op == Op.CONST:
            values[nid] = np.ascontiguousarray(node.params["value"])

        elif node.op == Op.INPUT:
            if nid not in inputs:
                raise ValueError(
                    f"Missing input for node {nid} ({node.name!r})")
            values[nid] = np.ascontiguousarray(inputs[nid])

        elif node.op == Op.LUT:
            fn_name = node.params["fn"]
            c_fn_name = _LUT_FN_MAP.get(fn_name)
            if c_fn_name is None:
                raise ValueError(f"Unknown LUT function: {fn_name!r}")
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            out = np.empty_like(x)
            getattr(lib, c_fn_name)(
                _to_c_float(out), _to_c_float(x), x.size)
            values[nid] = out

        elif node.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MAX):
            a = np.ascontiguousarray(inp[0], dtype=np.float32)
            b = np.ascontiguousarray(inp[1], dtype=np.float32)

            # Handle scalar (0-d) tensors
            a_shape = a.shape if a.ndim > 0 else (1,)
            b_shape = b.shape if b.ndim > 0 else (1,)
            out_shape = _broadcast_shapes(a_shape, b_shape)
            out = np.empty(out_shape, dtype=np.float32)

            fn_map = {
                Op.ADD: lib.ceval_add,
                Op.SUB: lib.ceval_sub,
                Op.MUL: lib.ceval_mul,
                Op.DIV: lib.ceval_div,
                Op.MAX: lib.ceval_max,
            }
            fn_map[node.op](
                _to_c_float(out),
                _to_c_float(a.reshape(-1) if a.ndim == 0 else a),
                _to_c_int_array(a_shape), len(a_shape),
                _to_c_float(b.reshape(-1) if b.ndim == 0 else b),
                _to_c_int_array(b_shape), len(b_shape),
                _to_c_int_array(out_shape), len(out_shape),
            )
            # Match NumPy scalar output behavior
            if inp[0].ndim == 0 and inp[1].ndim == 0:
                out = out.reshape(())
            values[nid] = out

        elif node.op == Op.CMP_LE:
            a = np.ascontiguousarray(inp[0], dtype=np.float32)
            b = np.ascontiguousarray(inp[1], dtype=np.float32)
            a_shape = a.shape if a.ndim > 0 else (1,)
            b_shape = b.shape if b.ndim > 0 else (1,)
            out_shape = _broadcast_shapes(a_shape, b_shape)
            out = np.empty(out_shape, dtype=np.float32)
            lib.ceval_cmp_le(
                _to_c_float(out),
                _to_c_float(a.reshape(-1) if a.ndim == 0 else a),
                _to_c_int_array(a_shape), len(a_shape),
                _to_c_float(b.reshape(-1) if b.ndim == 0 else b),
                _to_c_int_array(b_shape), len(b_shape),
                _to_c_int_array(out_shape), len(out_shape),
            )
            # Convert to uint8 to match reference evaluator
            values[nid] = out.astype(np.uint8)

        elif node.op == Op.MUX:
            cond = np.ascontiguousarray(inp[0]).astype(np.float32)
            a = np.ascontiguousarray(inp[1], dtype=np.float32)
            b = np.ascontiguousarray(inp[2], dtype=np.float32)
            out = np.empty_like(a)
            lib.ceval_mux(
                _to_c_float(out), _to_c_float(cond),
                _to_c_float(a), _to_c_float(b), a.size)
            values[nid] = out

        elif node.op == Op.NEG:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            out = np.empty_like(x)
            lib.ceval_neg(_to_c_float(out), _to_c_float(x), x.size)
            values[nid] = out

        elif node.op == Op.ABS:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            out = np.empty_like(x)
            lib.ceval_abs(_to_c_float(out), _to_c_float(x), x.size)
            values[nid] = out

        elif node.op == Op.SQUARE:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            out = np.empty_like(x)
            lib.ceval_square(_to_c_float(out), _to_c_float(x), x.size)
            values[nid] = out

        elif node.op == Op.MATMUL:
            a = np.ascontiguousarray(inp[0], dtype=np.float32)
            b = np.ascontiguousarray(inp[1], dtype=np.float32)
            out_shape = _matmul_shape(a.shape, b.shape)
            out = np.empty(out_shape, dtype=np.float32)
            lib.ceval_matmul(
                _to_c_float(out),
                _to_c_float(a), _to_c_int_array(a.shape), a.ndim,
                _to_c_float(b), _to_c_int_array(b.shape), b.ndim,
                _to_c_int_array(out_shape), len(out_shape),
            )
            values[nid] = out

        elif node.op in (Op.SUM, Op.MAX_REDUCE, Op.MEAN):
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            axis = node.params["axis"]
            keepdims = node.params.get("keepdims", False)
            # Normalise negative axis
            if axis < 0:
                axis = x.ndim + axis
            out_shape = _reduce_shape(x.shape, axis, keepdims)
            # Allocate output with the non-keepdims shape for C
            # (C always outputs reduced shape without keepdims)
            reduced_shape = _reduce_shape(x.shape, axis, False)
            reduced_size = 1
            for s in reduced_shape:
                reduced_size *= s
            if reduced_size == 0:
                reduced_size = 1  # scalar
            out = np.empty(max(reduced_size, 1), dtype=np.float32)

            fn_map = {
                Op.SUM: lib.ceval_sum,
                Op.MAX_REDUCE: lib.ceval_max_reduce,
                Op.MEAN: lib.ceval_mean,
            }
            fn_map[node.op](
                _to_c_float(out),
                _to_c_float(x),
                _to_c_int_array(x.shape), x.ndim,
                axis,
            )
            values[nid] = out.reshape(out_shape) if out_shape else out[0]

        elif node.op == Op.ARGMAX:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            axis = node.params["axis"]
            if axis < 0:
                axis = x.ndim + axis
            reduced_shape = _reduce_shape(x.shape, axis, False)
            reduced_size = 1
            for s in reduced_shape:
                reduced_size *= s
            if reduced_size == 0:
                reduced_size = 1
            out = np.empty(max(reduced_size, 1), dtype=np.int32)
            lib.ceval_argmax(
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                _to_c_float(x),
                _to_c_int_array(x.shape), x.ndim,
                axis,
            )
            # Match NumPy's int64 output
            out = out.astype(np.int64)
            if reduced_shape:
                values[nid] = out.reshape(reduced_shape)
            else:
                values[nid] = out[0]

        elif node.op == Op.RESHAPE:
            values[nid] = np.ascontiguousarray(
                inp[0]).reshape(node.params["shape"])

        elif node.op == Op.TRANSPOSE:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            axes = node.params["axes"]
            out_shape = tuple(x.shape[a] for a in axes)
            out = np.empty(out_shape, dtype=np.float32)
            lib.ceval_transpose(
                _to_c_float(out), _to_c_float(x),
                _to_c_int_array(x.shape), x.ndim,
                _to_c_int_array(axes),
            )
            values[nid] = out

        elif node.op == Op.CONCAT:
            axis = node.params["axis"]
            arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in inp]
            # Normalize negative axis
            if axis < 0:
                axis = arrays[0].ndim + axis
            # Use numpy for concat shape, then C for the copy
            # (concat is memory movement, not compute)
            values[nid] = np.concatenate(arrays, axis=axis)

        elif node.op == Op.REPEAT:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            repeats = node.params["repeats"]
            axis = node.params["axis"]
            if axis < 0:
                axis = x.ndim + axis
            out_shape = list(x.shape)
            out_shape[axis] *= repeats
            out = np.empty(out_shape, dtype=np.float32)
            lib.ceval_repeat(
                _to_c_float(out), _to_c_float(x),
                _to_c_int_array(x.shape), x.ndim,
                repeats, axis,
            )
            values[nid] = out

        elif node.op == Op.SLICE:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            slices = node.params["slices"]
            # Normalise slices to (start, stop, step) per dimension
            if not isinstance(slices, tuple):
                slices = (slices,)

            starts = []
            stops = []
            steps = []
            out_shape_list = []

            for d in range(x.ndim):
                if d < len(slices):
                    s = slices[d]
                    if isinstance(s, slice):
                        start, stop, step = s.indices(x.shape[d])
                    else:
                        # Integer index — not a slice, reduces dim
                        # For now, use numpy
                        values[nid] = x[slices]
                        break
                else:
                    start, stop, step = 0, x.shape[d], 1
                starts.append(start)
                stops.append(stop)
                steps.append(step)
                dim_size = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
                out_shape_list.append(dim_size)
            else:
                out_shape = tuple(out_shape_list)
                out = np.empty(out_shape, dtype=np.float32)
                lib.ceval_slice(
                    _to_c_float(out), _to_c_float(x),
                    _to_c_int_array(x.shape), x.ndim,
                    _to_c_int_array(starts),
                    _to_c_int_array(stops),
                    _to_c_int_array(steps),
                    _to_c_int_array(out_shape), len(out_shape),
                )
                values[nid] = out

        elif node.op == Op.CAST:
            values[nid] = np.asarray(inp[0]).astype(node.params["dtype"])

        elif node.op == Op.EXPAND_DIMS:
            values[nid] = np.expand_dims(
                np.ascontiguousarray(inp[0]), axis=node.params["axis"])

        else:
            raise ValueError(f"Unknown op: {node.op}")

    return values
