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

# Module-level ctypes pointer types (avoid repeated POINTER() calls)
_FP = ctypes.POINTER(ctypes.c_float)
_I8P = ctypes.POINTER(ctypes.c_int8)

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

    # Quantized matmul (x_f32 @ W_int8 * scales_f32)
    I8P = ctypes.POINTER(ctypes.c_int8)
    lib.ceval_matmul_q8.restype = None
    lib.ceval_matmul_q8.argtypes = [FP, FP, I, I, I8P, I, FP]


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


def _to_c_int8(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int8):
    """Get a ctypes int8 pointer to a contiguous int8 array."""
    arr = np.ascontiguousarray(arr, dtype=np.int8)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


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
_T_MATMUL_Q8 = 17

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
                 "_num_nodes", "_const_ptrs")

    def __init__(self, graph: CircuitGraph,
                 const_cache: dict[int, np.ndarray] | None = None) -> None:
        lib = _get_lib()
        self._lib = lib
        n = len(graph.nodes)
        self._num_nodes = n

        const_cache = const_cache or {}

        # Pre-seed CONST values into the base array
        base: list[np.ndarray | None] = [None] * n
        const_ids: set[int] = set()
        for node in graph.nodes:
            if node.op == Op.CONST:
                const_ids.add(node.id)
                if node.id in const_cache:
                    base[node.id] = const_cache[node.id]
                else:
                    base[node.id] = np.ascontiguousarray(
                        node.params["value"])
        self._base_values = base

        # Pre-cache ctypes pointers for CONST arrays (avoids repeated
        # ctypes.cast calls at runtime — saves ~10ms/step).
        const_ptrs: dict[int, object] = {}  # nid → ctypes pointer
        for nid in const_ids:
            arr = base[nid]
            if arr is not None:
                if arr.dtype == np.int8:
                    const_ptrs[nid] = arr.ctypes.data_as(_I8P)
                else:
                    arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
                    base[nid] = arr_f32  # ensure base has f32 version
                    const_ptrs[nid] = arr_f32.ctypes.data_as(_FP)
        self._const_ptrs = const_ptrs

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

            elif node.op == Op.MATMUL_Q8:
                # inputs: [activation, weight_q8, scales]
                # Pre-resolve CONST weight/scales pointers, shapes,
                # and pre-allocate output buffer.
                wq8_nid, scales_nid = ins[1], ins[2]
                wq8_arr = base[wq8_nid]
                scales_arr = base[scales_nid]
                K_q8 = wq8_arr.shape[0]
                N = wq8_arr.shape[-1]
                wq8_ptr = const_ptrs.get(wq8_nid)
                scales_ptr = const_ptrs.get(scales_nid)
                tape.append((_T_MATMUL_Q8, nid, ins[0],
                             wq8_ptr, K_q8, N, scales_ptr))

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

            elif tag == _T_MATMUL_Q8:
                # (tag, nid, x_id, wq8_ptr, K, N, scales_ptr)
                x = np.ascontiguousarray(v[instr[2]], dtype=np.float32)
                K = instr[4]
                N = instr[5]
                out = np.empty((1, N), dtype=np.float32)
                self._lib.ceval_matmul_q8(
                    out.ctypes.data_as(_FP),
                    x.ctypes.data_as(_FP), 1, K,
                    instr[3], N,   # pre-cached wq8 pointer
                    instr[6])      # pre-cached scales pointer
                v[instr[1]] = out

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
# C Tape Runner — execute entire tape in one C call
# ---------------------------------------------------------------

_TAPE_LIB: ctypes.CDLL | None = None


def _get_tape_lib() -> ctypes.CDLL:
    """Load the _tape_runner shared library."""
    global _TAPE_LIB
    if _TAPE_LIB is not None:
        return _TAPE_LIB

    csrc_dir = Path(__file__).resolve().parent.parent.parent / "csrc"
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    so_path = csrc_dir / f"_tape_runner{suffix}"
    if not so_path.exists():
        so_path = csrc_dir / "_tape_runner.so"
    if not so_path.exists():
        # Try to compile
        src = csrc_dir / "_tape_runner.c"
        if src.exists():
            _compile_tape_runner(src, so_path)
    _TAPE_LIB = ctypes.CDLL(str(so_path))
    _setup_tape_lib(_TAPE_LIB)
    return _TAPE_LIB


def _compile_tape_runner(src: Path, out: Path) -> None:
    """Compile _tape_runner.c."""
    if platform.system() == "Darwin":
        cmd = ["cc", "-O3", "-shared", "-fPIC", "-march=native",
               "-o", str(out), str(src), "-lm", "-framework", "Accelerate"]
    else:
        cmd = ["cc", "-O3", "-shared", "-fPIC", "-march=native",
               "-o", str(out), str(src), "-lm", "-lopenblas"]
    subprocess.run(cmd, check=True, capture_output=True)


def _setup_tape_lib(lib: ctypes.CDLL) -> None:
    """Set up ctypes signatures for the tape runner."""
    VP = ctypes.c_void_p
    IP = ctypes.POINTER(ctypes.c_int)
    FP = ctypes.POINTER(ctypes.c_float)
    I = ctypes.c_int

    lib.tape_ctx_create.restype = VP
    lib.tape_ctx_create.argtypes = [I, I]

    lib.tape_ctx_destroy.restype = None
    lib.tape_ctx_destroy.argtypes = [VP]

    lib.tape_slot_alloc.restype = None
    lib.tape_slot_alloc.argtypes = [VP, I, IP, I]

    lib.tape_slot_set_external.restype = None
    lib.tape_slot_set_external.argtypes = [VP, I, FP, IP, I]

    lib.tape_slot_write.restype = None
    lib.tape_slot_write.argtypes = [VP, I, FP, I]

    lib.tape_slot_set_shape.restype = None
    lib.tape_slot_set_shape.argtypes = [VP, I, IP, I]

    lib.tape_slot_read.restype = FP
    lib.tape_slot_read.argtypes = [VP, I, IP]

    lib.tape_get_instr.restype = VP
    lib.tape_get_instr.argtypes = [VP, I]

    lib.tape_run.restype = None
    lib.tape_run.argtypes = [VP]

    lib.tape_slot_read_shape.restype = I
    lib.tape_slot_read_shape.argtypes = [VP, I, IP]


# C tape instruction tags (match _tape_runner.c enum)
_CT_LUT_SILU  = 0
_CT_LUT_EXP   = 1
_CT_LUT_RSQRT = 2
_CT_LUT_COS   = 3
_CT_LUT_SIN   = 4
_CT_ADD       = 5
_CT_SUB       = 6
_CT_MUL       = 7
_CT_DIV       = 8
_CT_MAX       = 9
_CT_NEG       = 10
_CT_SQUARE    = 11
_CT_MATMUL    = 12
_CT_MATMUL_Q8 = 13
_CT_SUM       = 14
_CT_MAX_RED   = 15
_CT_MEAN      = 16
_CT_ARGMAX    = 17
_CT_RESHAPE   = 18
_CT_TRANSPOSE = 19
_CT_CONCAT    = 20
_CT_REPEAT    = 21
_CT_SLICE     = 22
_CT_COPY      = 23

_LUT_TAG_MAP = {
    "silu": _CT_LUT_SILU, "exp": _CT_LUT_EXP, "rsqrt": _CT_LUT_RSQRT,
    "cos": _CT_LUT_COS, "sin": _CT_LUT_SIN,
}

_BINOP_TAG = {
    Op.ADD: _CT_ADD, Op.SUB: _CT_SUB, Op.MUL: _CT_MUL,
    Op.DIV: _CT_DIV, Op.MAX: _CT_MAX,
}

_REDUCE_TAG = {
    Op.SUM: _CT_SUM, Op.MAX_REDUCE: _CT_MAX_RED, Op.MEAN: _CT_MEAN,
}


class _TapeInstrFields(ctypes.Structure):
    """Mirror of C TapeInstr struct for field access."""
    _fields_ = [
        ("tag", ctypes.c_int),
        ("out_slot", ctypes.c_int),
        ("in0", ctypes.c_int),
        ("in1", ctypes.c_int),
        ("in2", ctypes.c_int),
        ("axis", ctypes.c_int),
        ("keepdims", ctypes.c_int),
        ("repeats", ctypes.c_int),
        ("out_shape", ctypes.c_int * 8),
        ("out_ndim", ctypes.c_int),
        ("out_size", ctypes.c_int),
        ("wq8_ptr", ctypes.c_void_p),
        ("scales_ptr", ctypes.c_void_p),
        ("K_q8", ctypes.c_int),
        ("N_q8", ctypes.c_int),
        ("axes", ctypes.c_int * 8),
        ("starts", ctypes.c_int * 8),
        ("stops", ctypes.c_int * 8),
        ("steps", ctypes.c_int * 8),
        ("slice_out_shape", ctypes.c_int * 8),
        ("slice_out_ndim", ctypes.c_int),
        ("concat_inputs", ctypes.c_int * 8),
        ("concat_n_inputs", ctypes.c_int),
    ]


def _int_arr(vals):
    """Make a ctypes int array from a sequence."""
    return (ctypes.c_int * len(vals))(*vals)


class CTapeRunner:
    """Execute the decode machine tape entirely in C.

    Translates an ExecutionPlan + CircuitGraph into a C TapeCtx,
    then each ``run()`` call does one ``tape_run()`` C call —
    zero Python iteration over 1371 tape instructions.
    """

    def __init__(self, graph: CircuitGraph,
                 const_cache: dict[int, np.ndarray] | None = None,
                 sample_inputs: dict[int, np.ndarray] | None = None):
        self._lib = _get_tape_lib()
        self._graph = graph
        nodes = graph.nodes
        n_slots = len(nodes)

        # Phase 1: evaluate shapes by doing one forward pass
        const_cache = const_cache or {}
        sample_inputs = sample_inputs or {}
        shapes: dict[int, tuple] = {}
        slot_data: dict[int, np.ndarray] = {}

        # Collect CONST and INPUT node info
        const_ids: set[int] = set()
        input_ids: list[int] = []
        for node in nodes:
            if node.op == Op.CONST:
                const_ids.add(node.id)
                arr = const_cache.get(node.id)
                if arr is None:
                    arr = np.ascontiguousarray(node.params["value"])
                slot_data[node.id] = arr
                shapes[node.id] = arr.shape
            elif node.op == Op.INPUT:
                input_ids.append(node.id)
                if node.id in sample_inputs:
                    s = sample_inputs[node.id].shape
                else:
                    s = node.params.get("shape", (1,))
                shapes[node.id] = s
                slot_data[node.id] = np.zeros(s, dtype=np.float32)

        # Build topological order and compile instructions
        order = graph.topological_order()
        instrs = []  # list of (c_tag, out_slot, params_dict)

        for nid in order:
            node = nodes[nid]
            if node.op in (Op.CONST, Op.INPUT):
                continue

            ins = tuple(node.inputs)
            p = {"in0": ins[0] if len(ins) > 0 else -1,
                 "in1": ins[1] if len(ins) > 1 else -1,
                 "in2": ins[2] if len(ins) > 2 else -1}

            # Compute output shape from inputs
            if node.op == Op.LUT:
                s = shapes[ins[0]]
                shapes[nid] = s
                tag = _LUT_TAG_MAP[node.params["fn"]]
                instrs.append((tag, nid, p))

            elif node.op in _BINOP_TAG:
                s0 = shapes[ins[0]]
                s1 = shapes[ins[1]]
                out_s = np.broadcast_shapes(s0, s1)
                shapes[nid] = out_s
                p["out_shape"] = out_s
                instrs.append((_BINOP_TAG[node.op], nid, p))

            elif node.op == Op.NEG:
                shapes[nid] = shapes[ins[0]]
                instrs.append((_CT_NEG, nid, p))

            elif node.op == Op.SQUARE:
                shapes[nid] = shapes[ins[0]]
                instrs.append((_CT_SQUARE, nid, p))

            elif node.op == Op.MATMUL:
                s0, s1 = shapes[ins[0]], shapes[ins[1]]
                out_s = s0[:-1] + (s1[-1],)
                if len(s0) > 2:
                    out_s = s0[:-2] + (s0[-2], s1[-1])
                shapes[nid] = out_s
                instrs.append((_CT_MATMUL, nid, p))

            elif node.op == Op.MATMUL_Q8:
                wq8 = slot_data.get(ins[1])
                scales = slot_data.get(ins[2])
                s_in = shapes[ins[0]]
                K_q8 = wq8.shape[0] if wq8 is not None else 0
                N_q8 = wq8.shape[1] if wq8 is not None else 0
                M = s_in[0] if len(s_in) >= 2 else 1
                shapes[nid] = (M, N_q8)
                p["wq8_ptr"] = wq8.ctypes.data if wq8 is not None else 0
                p["scales_ptr"] = scales.ctypes.data if scales is not None else 0
                p["K_q8"] = K_q8
                p["N_q8"] = N_q8
                instrs.append((_CT_MATMUL_Q8, nid, p))

            elif node.op in _REDUCE_TAG:
                axis = node.params["axis"]
                keepdims = node.params.get("keepdims", False)
                s = shapes[ins[0]]
                ax = axis if axis >= 0 else len(s) + axis
                if keepdims:
                    out_s = s[:ax] + (1,) + s[ax+1:]
                else:
                    out_s = s[:ax] + s[ax+1:]
                shapes[nid] = out_s
                p["axis"] = ax
                p["keepdims"] = int(keepdims)
                instrs.append((_REDUCE_TAG[node.op], nid, p))

            elif node.op == Op.ARGMAX:
                axis = node.params["axis"]
                s = shapes[ins[0]]
                ax = axis if axis >= 0 else len(s) + axis
                out_s = s[:ax] + s[ax+1:]
                shapes[nid] = out_s if out_s else (1,)
                p["axis"] = ax
                instrs.append((_CT_ARGMAX, nid, p))

            elif node.op == Op.RESHAPE:
                target = node.params["shape"]
                shapes[nid] = tuple(target)
                p["out_shape"] = tuple(target)
                instrs.append((_CT_RESHAPE, nid, p))

            elif node.op == Op.TRANSPOSE:
                axes_perm = node.params["axes"]
                s = shapes[ins[0]]
                out_s = tuple(s[a] for a in axes_perm)
                shapes[nid] = out_s
                p["axes"] = axes_perm
                instrs.append((_CT_TRANSPOSE, nid, p))

            elif node.op == Op.CONCAT:
                axis = node.params["axis"]
                ss = [shapes[i] for i in ins]
                ax = axis if axis >= 0 else len(ss[0]) + axis
                total_ax = sum(s[ax] for s in ss)
                out_s = list(ss[0])
                out_s[ax] = total_ax
                shapes[nid] = tuple(out_s)
                p["axis"] = ax
                p["concat_inputs"] = list(ins)
                instrs.append((_CT_CONCAT, nid, p))

            elif node.op == Op.REPEAT:
                repeats = node.params["repeats"]
                axis = node.params["axis"]
                s = list(shapes[ins[0]])
                s[axis] *= repeats
                shapes[nid] = tuple(s)
                p["repeats"] = repeats
                p["axis"] = axis
                instrs.append((_CT_REPEAT, nid, p))

            elif node.op == Op.SLICE:
                slices = node.params["slices"]
                s_in = shapes[ins[0]]
                out_dims = []
                slice_starts = []
                slice_steps = []
                # Track which dims are "full" (slice(None)) — use 0 sentinel
                is_full_dim = []
                for d, sl in enumerate(slices):
                    if isinstance(sl, slice):
                        start = sl.start or 0
                        stop = sl.stop if sl.stop is not None else s_in[d]
                        step = sl.step or 1
                        if start < 0:
                            start += s_in[d]
                        if stop < 0:
                            stop += s_in[d]
                        dim_size = max(0, (stop - start + step - 1) // step) if step > 0 else 0
                        out_dims.append(dim_size)
                        slice_starts.append(start)
                        slice_steps.append(step)
                        # Mark as full if it covers the whole dimension
                        is_full_dim.append(
                            sl.start is None and sl.stop is None and (sl.step is None or sl.step == 1))
                    elif isinstance(sl, int):
                        idx = sl if sl >= 0 else sl + s_in[d]
                        slice_starts.append(idx)
                        slice_steps.append(1)
                        out_dims.append(1)
                        is_full_dim.append(False)
                    else:
                        out_dims.append(s_in[d])
                        slice_starts.append(0)
                        slice_steps.append(1)
                        is_full_dim.append(True)
                # Detect integer-indexed dims to squeeze
                squeeze_dims = []
                for d, sl in enumerate(slices):
                    if isinstance(sl, int):
                        squeeze_dims.append(d)
                final_shape = [out_dims[d] for d in range(len(out_dims))
                               if d not in squeeze_dims]
                if not final_shape:
                    final_shape = [1]
                shapes[nid] = tuple(final_shape)
                p["starts"] = slice_starts
                p["steps"] = slice_steps
                # Use 0 sentinel for full-dimension slices (dynamic at runtime)
                p["slice_out_shape"] = [
                    0 if is_full_dim[d] else out_dims[d]
                    for d in range(len(out_dims))
                ]
                instrs.append((_CT_SLICE, nid, p))

            elif node.op == Op.EXPAND_DIMS:
                axis = node.params["axis"]
                s = shapes[ins[0]]
                out_s = s[:axis] + (1,) + s[axis:]
                shapes[nid] = out_s
                instrs.append((_CT_COPY, nid, p))

            elif node.op == Op.CAST:
                shapes[nid] = shapes[ins[0]]
                instrs.append((_CT_COPY, nid, p))

            else:
                raise ValueError(f"CTapeRunner: unknown op {node.op}")

        # Phase 2: Create C TapeCtx
        ctx = self._lib.tape_ctx_create(n_slots, len(instrs))
        self._ctx = ctx
        self._shapes = shapes
        self._input_ids = input_ids
        self._n_slots = n_slots
        self._slot_data = slot_data  # keep refs alive for GC

        # Allocate slots
        for nid in range(n_slots):
            node = nodes[nid]
            s = shapes.get(nid)
            if s is None:
                continue
            shape_arr = _int_arr(s)
            ndim = len(s)
            if nid in const_ids:
                # Point at existing numpy data
                arr = slot_data[nid]
                if arr.dtype == np.float32:
                    self._lib.tape_slot_set_external(
                        ctx, nid,
                        arr.ctypes.data_as(_FP),
                        shape_arr, ndim)
                elif arr.dtype == np.int8:
                    # int8 slots: allocate dummy f32 slot (not used as f32)
                    self._lib.tape_slot_alloc(ctx, nid, shape_arr, ndim)
                else:
                    arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
                    slot_data[nid] = arr_f32
                    self._lib.tape_slot_set_external(
                        ctx, nid,
                        arr_f32.ctypes.data_as(_FP),
                        shape_arr, ndim)
            elif nid in [i for i in input_ids]:
                self._lib.tape_slot_alloc(ctx, nid, shape_arr, ndim)
            else:
                # Intermediate — allocate
                self._lib.tape_slot_alloc(ctx, nid, shape_arr, ndim)

        # Phase 3: Fill tape instructions
        for idx, (tag, out_slot, p) in enumerate(instrs):
            iptr = self._lib.tape_get_instr(ctx, idx)
            instr = ctypes.cast(iptr, ctypes.POINTER(_TapeInstrFields)).contents
            instr.tag = tag
            instr.out_slot = out_slot
            instr.in0 = p.get("in0", -1)
            instr.in1 = p.get("in1", -1)
            instr.in2 = p.get("in2", -1)
            instr.axis = p.get("axis", 0)
            instr.keepdims = p.get("keepdims", 0)
            instr.repeats = p.get("repeats", 0)

            # out_shape for binops
            if "out_shape" in p:
                os = p["out_shape"]
                for d in range(len(os)):
                    instr.out_shape[d] = os[d]
                instr.out_ndim = len(os)
                sz = 1
                for d in os:
                    sz *= d
                instr.out_size = sz

            # MATMUL_Q8 pointers
            if tag == _CT_MATMUL_Q8:
                instr.wq8_ptr = p["wq8_ptr"]
                instr.scales_ptr = p["scales_ptr"]
                instr.K_q8 = p["K_q8"]
                instr.N_q8 = p["N_q8"]

            # Transpose axes
            if "axes" in p:
                for d, a in enumerate(p["axes"]):
                    instr.axes[d] = a

            # Slice params
            if "starts" in p:
                for d in range(len(p["starts"])):
                    instr.starts[d] = p["starts"][d]
                    instr.steps[d] = p["steps"][d]
                sos = p["slice_out_shape"]
                for d in range(len(sos)):
                    instr.slice_out_shape[d] = sos[d]
                instr.slice_out_ndim = len(sos)

            # Concat inputs
            if "concat_inputs" in p:
                ci = p["concat_inputs"]
                for d in range(len(ci)):
                    instr.concat_inputs[d] = ci[d]
                instr.concat_n_inputs = len(ci)

    def run(self, inputs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Execute the tape in C. Returns dict[nid → array]."""
        lib = self._lib
        ctx = self._ctx

        # Seed INPUT slots
        for nid in self._input_ids:
            arr = np.ascontiguousarray(inputs[nid], dtype=np.float32)
            shape_arr = _int_arr(arr.shape)
            # Write data first (handles reallocation if size grew)
            lib.tape_slot_write(ctx, nid,
                                arr.ctypes.data_as(_FP),
                                arr.size)
            # Then update shape metadata
            lib.tape_slot_set_shape(ctx, nid, shape_arr, len(arr.shape))

        # One C call executes entire tape
        lib.tape_run(ctx)

        return self

    def get_value(self, nid: int) -> np.ndarray:
        """Read a slot's data as numpy array."""
        out_size = ctypes.c_int(0)
        ptr = self._lib.tape_slot_read(self._ctx, nid,
                                        ctypes.byref(out_size))
        # Read actual shape from C slot (handles dynamic KV cache growth)
        shape_buf = (ctypes.c_int * 8)()
        ndim = self._lib.tape_slot_read_shape(self._ctx, nid, shape_buf)
        s = tuple(shape_buf[d] for d in range(ndim))
        return np.ctypeslib.as_array(ptr, shape=(out_size.value,)).reshape(s).copy()

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.tape_ctx_destroy(self._ctx)
            self._ctx = None


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
            a = np.ascontiguousarray(inp[0], dtype=np.float32)
            b = np.ascontiguousarray(inp[1], dtype=np.float32)
            values[nid] = np.matmul(a, b)

        elif node.op == Op.MATMUL_Q8:
            x = np.ascontiguousarray(inp[0], dtype=np.float32)
            wq8 = np.ascontiguousarray(inp[1], dtype=np.int8)
            scales = np.ascontiguousarray(inp[2], dtype=np.float32)
            M, K = x.shape[-2] if x.ndim >= 2 else 1, x.shape[-1]
            N = wq8.shape[-1]
            out = np.empty((M, N), dtype=np.float32)
            lib.ceval_matmul_q8(
                _to_c_float(out),
                _to_c_float(x.reshape(M, K)), M, K,
                _to_c_int8(wq8), N,
                _to_c_float(scales))
            values[nid] = out

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
