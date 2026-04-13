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
    # Link BLAS: Accelerate on macOS, OpenBLAS on Linux
    if platform.system() == "Darwin":
        cmd += ["-framework", "Accelerate"]
    else:
        cmd += ["-lopenblas"]
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
# Instruction tape tags
# ---------------------------------------------------------------
_T_LUT = 0
_T_BINOP = 1
_T_CMP_LE = 2
_T_MUX = 3
_T_NEG = 4
_T_ABS = 5
_T_SQUARE = 6
_T_MATMUL = 7
_T_REDUCE = 8
_T_ARGMAX = 9
_T_RESHAPE = 10
_T_TRANSPOSE = 11
_T_CONCAT = 12
_T_REPEAT = 13
_T_SLICE = 14
_T_CAST = 15
_T_EXPAND_DIMS = 16

_BINOP_FN = {
    Op.ADD: np.add,
    Op.SUB: np.subtract,
    Op.MUL: np.multiply,
    Op.DIV: np.divide,
    Op.MAX: np.maximum,
}

_REDUCE_FN = {
    Op.SUM: np.sum,
    Op.MAX_REDUCE: np.max,
    Op.MEAN: np.mean,
}

# ---------------------------------------------------------------
# ExecutionPlan — pre-compiled instruction tape
# ---------------------------------------------------------------


class ExecutionPlan:
    """Pre-compiled execution plan for a CircuitGraph.

    Walks the graph once at construction time and produces:

    * A flat *values* list with CONST entries pre-seeded.
    * A list of INPUT node IDs for fast runtime seeding.
    * An instruction tape — each entry is a tuple with pre-resolved
      function pointers, pre-extracted params, and integer node IDs.

    ``run(inputs)`` copies the base list, seeds inputs, and executes
    the tape.  No topological sort, no ``if/elif`` op dispatch by
    enum, no ``node.params`` dict lookups at runtime.
    """

    __slots__ = ("_base_values", "_input_nids", "_tape", "_lib",
                 "_num_nodes")

    def __init__(self, graph: CircuitGraph,
                 const_cache: dict[int, np.ndarray] | None = None) -> None:
        lib = _get_lib()
        self._lib = lib
        n = len(graph.nodes)
        self._num_nodes = n

        const_cache = const_cache or {}

        # Pre-seed CONST values into the base array
        base: list[np.ndarray | None] = [None] * n
        for node in graph.nodes:
            if node.op == Op.CONST:
                if node.id in const_cache:
                    base[node.id] = const_cache[node.id]
                else:
                    base[node.id] = np.ascontiguousarray(
                        node.params["value"])
        self._base_values = base

        # INPUT node IDs (for validation at runtime)
        self._input_nids: list[int] = []

        # Build the instruction tape
        tape: list[tuple] = []
        order = graph.topological_order()

        for nid in order:
            node = graph.nodes[nid]

            if node.op == Op.CONST:
                continue  # pre-seeded in base

            if node.op == Op.INPUT:
                self._input_nids.append(nid)
                continue

            ins = tuple(node.inputs)

            if node.op == Op.LUT:
                c_fn = getattr(lib, _LUT_FN_MAP[node.params["fn"]])
                tape.append((_T_LUT, nid, ins[0], c_fn))

            elif node.op in _BINOP_FN:
                tape.append((_T_BINOP, nid, ins[0], ins[1],
                             _BINOP_FN[node.op]))

            elif node.op == Op.CMP_LE:
                tape.append((_T_CMP_LE, nid, ins[0], ins[1]))

            elif node.op == Op.MUX:
                tape.append((_T_MUX, nid, ins[0], ins[1], ins[2]))

            elif node.op == Op.NEG:
                tape.append((_T_NEG, nid, ins[0]))

            elif node.op == Op.ABS:
                tape.append((_T_ABS, nid, ins[0]))

            elif node.op == Op.SQUARE:
                tape.append((_T_SQUARE, nid, ins[0]))

            elif node.op == Op.MATMUL:
                tape.append((_T_MATMUL, nid, ins[0], ins[1]))

            elif node.op in _REDUCE_FN:
                axis = node.params["axis"]
                keepdims = node.params.get("keepdims", False)
                tape.append((_T_REDUCE, nid, ins[0],
                             _REDUCE_FN[node.op], axis, keepdims))

            elif node.op == Op.ARGMAX:
                tape.append((_T_ARGMAX, nid, ins[0],
                             node.params["axis"]))

            elif node.op == Op.RESHAPE:
                tape.append((_T_RESHAPE, nid, ins[0],
                             node.params["shape"]))

            elif node.op == Op.TRANSPOSE:
                tape.append((_T_TRANSPOSE, nid, ins[0],
                             node.params["axes"]))

            elif node.op == Op.CONCAT:
                axis = node.params["axis"]
                tape.append((_T_CONCAT, nid, ins, axis))

            elif node.op == Op.REPEAT:
                tape.append((_T_REPEAT, nid, ins[0],
                             node.params["repeats"],
                             node.params["axis"]))

            elif node.op == Op.SLICE:
                tape.append((_T_SLICE, nid, ins[0],
                             node.params["slices"]))

            elif node.op == Op.CAST:
                tape.append((_T_CAST, nid, ins[0],
                             node.params["dtype"]))

            elif node.op == Op.EXPAND_DIMS:
                tape.append((_T_EXPAND_DIMS, nid, ins[0],
                             node.params["axis"]))

            else:
                raise ValueError(f"Unknown op: {node.op}")

        self._tape = tape

    def run(self, inputs: dict[int, np.ndarray]) -> list[np.ndarray | None]:
        """Execute the plan.  Returns a list indexed by node ID."""
        v = self._base_values.copy()  # shallow — shares numpy arrays

        # Seed INPUT nodes
        for nid in self._input_nids:
            if nid not in inputs:
                raise ValueError(f"Missing input for node {nid}")
            v[nid] = np.ascontiguousarray(inputs[nid])

        # Execute the instruction tape
        for instr in self._tape:
            tag = instr[0]

            if tag == _T_BINOP:
                # (tag, nid, i0, i1, np_fn)
                v[instr[1]] = instr[4](
                    np.asarray(v[instr[2]], dtype=np.float32),
                    np.asarray(v[instr[3]], dtype=np.float32))

            elif tag == _T_MATMUL:
                # (tag, nid, i0, i1)
                v[instr[1]] = np.matmul(
                    np.ascontiguousarray(v[instr[2]], dtype=np.float32),
                    np.ascontiguousarray(v[instr[3]], dtype=np.float32))

            elif tag == _T_LUT:
                # (tag, nid, i0, c_fn)
                x = np.ascontiguousarray(v[instr[2]], dtype=np.float32)
                out = np.empty_like(x)
                instr[3](_to_c_float(out), _to_c_float(x), x.size)
                v[instr[1]] = out

            elif tag == _T_RESHAPE:
                # (tag, nid, i0, shape)
                v[instr[1]] = np.ascontiguousarray(
                    v[instr[2]]).reshape(instr[3])

            elif tag == _T_TRANSPOSE:
                # (tag, nid, i0, axes)
                v[instr[1]] = np.ascontiguousarray(
                    np.transpose(v[instr[2]], axes=instr[3]),
                    dtype=np.float32)

            elif tag == _T_CONCAT:
                # (tag, nid, input_nids, axis)
                axis = instr[3]
                arrays = [np.ascontiguousarray(v[i], dtype=np.float32)
                          for i in instr[2]]
                if axis < 0:
                    axis = arrays[0].ndim + axis
                v[instr[1]] = np.concatenate(arrays, axis=axis)

            elif tag == _T_SLICE:
                # (tag, nid, i0, slices)
                v[instr[1]] = np.ascontiguousarray(
                    np.asarray(v[instr[2]], dtype=np.float32)[instr[3]])

            elif tag == _T_REDUCE:
                # (tag, nid, i0, np_fn, axis, keepdims)
                v[instr[1]] = instr[3](
                    np.asarray(v[instr[2]], dtype=np.float32),
                    axis=instr[4], keepdims=instr[5])

            elif tag == _T_NEG:
                # (tag, nid, i0)
                v[instr[1]] = -np.asarray(v[instr[2]], dtype=np.float32)

            elif tag == _T_SQUARE:
                # (tag, nid, i0)
                x = np.asarray(v[instr[2]], dtype=np.float32)
                v[instr[1]] = x * x

            elif tag == _T_REPEAT:
                # (tag, nid, i0, repeats, axis)
                v[instr[1]] = np.repeat(
                    np.asarray(v[instr[2]], dtype=np.float32),
                    instr[3], axis=instr[4])

            elif tag == _T_EXPAND_DIMS:
                # (tag, nid, i0, axis)
                v[instr[1]] = np.expand_dims(
                    np.ascontiguousarray(v[instr[2]]), axis=instr[3])

            elif tag == _T_MUX:
                # (tag, nid, cond, a, b)
                cond = np.asarray(v[instr[2]], dtype=np.float32)
                v[instr[1]] = np.where(
                    cond != 0.0,
                    np.asarray(v[instr[4]], dtype=np.float32),
                    np.asarray(v[instr[3]], dtype=np.float32))

            elif tag == _T_CMP_LE:
                # (tag, nid, i0, i1)
                v[instr[1]] = (
                    np.asarray(v[instr[2]], dtype=np.float32)
                    <= np.asarray(v[instr[3]], dtype=np.float32)
                ).astype(np.uint8)

            elif tag == _T_ABS:
                # (tag, nid, i0)
                v[instr[1]] = np.abs(
                    np.asarray(v[instr[2]], dtype=np.float32))

            elif tag == _T_ARGMAX:
                # (tag, nid, i0, axis)
                v[instr[1]] = np.argmax(
                    np.asarray(v[instr[2]], dtype=np.float32),
                    axis=instr[3])

            elif tag == _T_CAST:
                # (tag, nid, i0, dtype)
                v[instr[1]] = np.asarray(v[instr[2]]).astype(instr[3])

        return v


# ---------------------------------------------------------------
# Const pre-caching
# ---------------------------------------------------------------


def precompute_consts(graph: CircuitGraph) -> dict[int, np.ndarray]:
    """Pre-evaluate all CONST nodes once and return a reusable cache.

    Pass the result as ``const_cache`` to ``evaluate_c`` to skip
    redundant ``np.ascontiguousarray`` calls on every evaluation.
    """
    cache: dict[int, np.ndarray] = {}
    for node in graph.nodes:
        if node.op == Op.CONST:
            cache[node.id] = np.ascontiguousarray(node.params["value"])
    return cache


# ---------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------


def evaluate_c(graph: CircuitGraph,
               inputs: dict[int, np.ndarray] | None = None,
               const_cache: dict[int, np.ndarray] | None = None,
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
    const_cache : dict mapping CONST node ID → NumPy array, optional
        Pre-evaluated constant values.  When provided, CONST nodes
        found in this dict are reused instantly (no copy/check).
        Call ``precompute_consts(graph)`` to build this once.

    Returns
    -------
    dict mapping node ID → NumPy array
        Computed value for every node.
    """
    lib = _get_lib()
    inputs = inputs or {}
    const_cache = const_cache or {}
    values: dict[int, np.ndarray] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]
        inp = [values[i] for i in node.inputs]

        if node.op == Op.CONST:
            if nid in const_cache:
                values[nid] = const_cache[nid]
            else:
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
            a = np.asarray(inp[0], dtype=np.float32)
            b = np.asarray(inp[1], dtype=np.float32)
            _np_binop = {
                Op.ADD: np.add,
                Op.SUB: np.subtract,
                Op.MUL: np.multiply,
                Op.DIV: np.divide,
                Op.MAX: np.maximum,
            }
            values[nid] = _np_binop[node.op](a, b)

        elif node.op == Op.CMP_LE:
            a = np.asarray(inp[0], dtype=np.float32)
            b = np.asarray(inp[1], dtype=np.float32)
            values[nid] = (a <= b).astype(np.uint8)

        elif node.op == Op.MUX:
            cond = np.asarray(inp[0], dtype=np.float32)
            a = np.asarray(inp[1], dtype=np.float32)
            b = np.asarray(inp[2], dtype=np.float32)
            values[nid] = np.where(cond != 0.0, b, a)

        elif node.op == Op.NEG:
            values[nid] = -np.asarray(inp[0], dtype=np.float32)

        elif node.op == Op.ABS:
            values[nid] = np.abs(np.asarray(inp[0], dtype=np.float32))

        elif node.op == Op.SQUARE:
            x = np.asarray(inp[0], dtype=np.float32)
            values[nid] = x * x

        elif node.op == Op.MATMUL:
            # Use np.matmul — it links to BLAS (Accelerate/MKL/OpenBLAS)
            # with optimized dispatch, avoiding the ctypes overhead of
            # calling our C wrapper per matmul.
            a = np.ascontiguousarray(inp[0], dtype=np.float32)
            b = np.ascontiguousarray(inp[1], dtype=np.float32)
            values[nid] = np.matmul(a, b)

        elif node.op in (Op.SUM, Op.MAX_REDUCE, Op.MEAN):
            x = np.asarray(inp[0], dtype=np.float32)
            axis = node.params["axis"]
            keepdims = node.params.get("keepdims", False)
            _np_reduce = {
                Op.SUM: np.sum,
                Op.MAX_REDUCE: np.max,
                Op.MEAN: np.mean,
            }
            values[nid] = _np_reduce[node.op](
                x, axis=axis, keepdims=keepdims)

        elif node.op == Op.ARGMAX:
            x = np.asarray(inp[0], dtype=np.float32)
            axis = node.params["axis"]
            values[nid] = np.argmax(x, axis=axis)

        elif node.op == Op.RESHAPE:
            values[nid] = np.ascontiguousarray(
                inp[0]).reshape(node.params["shape"])

        elif node.op == Op.TRANSPOSE:
            values[nid] = np.ascontiguousarray(
                np.transpose(inp[0], axes=node.params["axes"]),
                dtype=np.float32,
            )

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
            x = np.asarray(inp[0], dtype=np.float32)
            repeats = node.params["repeats"]
            axis = node.params["axis"]
            values[nid] = np.repeat(x, repeats, axis=axis)

        elif node.op == Op.SLICE:
            x = np.asarray(inp[0], dtype=np.float32)
            slices = node.params["slices"]
            values[nid] = np.ascontiguousarray(x[slices])

        elif node.op == Op.CAST:
            values[nid] = np.asarray(inp[0]).astype(node.params["dtype"])

        elif node.op == Op.EXPAND_DIMS:
            values[nid] = np.expand_dims(
                np.ascontiguousarray(inp[0]), axis=node.params["axis"])

        else:
            raise ValueError(f"Unknown op: {node.op}")

    return values
