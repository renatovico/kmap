"""Circuit graph — DAG of gate nodes representing the complete computation.

Every operation the transformer performs (matmul, add, multiply, norm,
softmax, embedding lookup, …) is a node in this graph.  The graph IS
the circuit.  It can be:

1. **Evaluated** by the reference evaluator (NumPy) — golden output.
2. **Executed** by the C gate executor (byte-plane shift+XOR) — Phase 4.
3. **Optimised** (constant folding, dead elimination, gate merge) — Phase 5.
4. **Synthesised** to Verilog/VHDL for FPGA — Phase 7.

The Python IEEE-754 bit-manipulation that was in ``binary_ops.py`` does
NOT belong here — execution is the **executor's** responsibility,
not the graph's.  The graph defines WHAT to compute; the executor
defines HOW.

Node kinds
----------
- ``const``    : fixed tensor (weight, bias, RoPE freq, causal -inf)
- ``input``    : graph input (token IDs, position indices)
- ``lut``      : unary lookup table (SiLU, exp, rsqrt, cos, sin)
- ``add``      : IEEE-754 float32 element-wise addition
- ``sub``      : negate + add (sign-bit XOR then add)
- ``mul``      : IEEE-754 float32 element-wise multiplication
- ``div``      : IEEE-754 float32 element-wise division
- ``neg``      : flip sign bit (bit 31 XOR)
- ``abs``      : clear sign bit (bit 31 AND-mask)
- ``max``      : element-wise max (sign-magnitude compare)
- ``cmp_le``   : element-wise a ≤ b → uint8
- ``mux``      : select(cond, a, b) — bitwise multiplexer
- ``matmul``   : matrix multiply (grid of mul + sum-reduction)
- ``sum``      : reduce-sum along axis
- ``max_reduce``: reduce-max along axis
- ``argmax``   : index of max along axis
- ``mean``     : sum / count
- ``reshape``  : wire routing (zero gates)
- ``transpose``: wire routing (zero gates)
- ``concat``   : wire join (zero gates)
- ``repeat``   : wire fanout (zero gates)
- ``slice``    : wire select (zero gates)
- ``sqrt``     : via rsqrt LUT (or Newton-Raphson circuit)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Op(str, Enum):
    """Node operation type."""
    # Storage / I/O
    CONST       = "const"
    INPUT       = "input"

    # Unary LUT (activation circuits)
    LUT         = "lut"

    # IEEE-754 arithmetic (element-wise)
    ADD         = "add"
    SUB         = "sub"
    MUL         = "mul"
    DIV         = "div"
    NEG         = "neg"
    ABS         = "abs"
    SQUARE      = "square"

    # Comparison / selection
    MAX         = "max"
    CMP_LE      = "cmp_le"
    MUX         = "mux"

    # Linear algebra
    MATMUL      = "matmul"

    # Reductions
    SUM         = "sum"
    MAX_REDUCE  = "max_reduce"
    ARGMAX      = "argmax"
    MEAN        = "mean"

    # Wire routing (zero gates)
    RESHAPE     = "reshape"
    TRANSPOSE   = "transpose"
    CONCAT      = "concat"
    REPEAT      = "repeat"
    SLICE       = "slice"

    # Cast / view
    CAST        = "cast"
    EXPAND_DIMS = "expand_dims"


@dataclass
class Node:
    """Single node in the circuit DAG."""
    id: int
    op: Op
    inputs: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    name: str = ""


class CircuitGraph:
    """Directed acyclic graph of gate nodes.

    Build the graph by adding nodes (``const``, ``add``, ``matmul``, …).
    Each method returns the new node's ID.  Then evaluate with the
    reference evaluator or serialise for the C executor.
    """

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self._next_id: int = 0

    def _add_node(self, op: Op, inputs: list[int],
                  params: dict | None = None,
                  shape: tuple[int, ...] | None = None,
                  dtype: np.dtype | None = None,
                  name: str = "") -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes.append(Node(
            id=nid,
            op=op,
            inputs=inputs,
            params=params or {},
            shape=shape,
            dtype=dtype or np.dtype(np.float32),
            name=name,
        ))
        return nid

    # ---- Storage / I/O ------------------------------------------

    def const(self, value: np.ndarray, name: str = "") -> int:
        """Add a constant node (weight, bias, precomputed value)."""
        v = np.asarray(value)
        return self._add_node(Op.CONST, [], {"value": v},
                              shape=v.shape, dtype=v.dtype, name=name)

    def input(self, shape: tuple[int, ...],
              dtype: np.dtype = np.dtype(np.float32),
              name: str = "") -> int:
        """Add an input node (token IDs, position indices)."""
        return self._add_node(Op.INPUT, [], {"name": name},
                              shape=shape, dtype=dtype, name=name)

    # ---- Unary LUT (activation circuits) ------------------------

    def lut(self, x: int, fn_name: str, name: str = "") -> int:
        """Unary lookup-table gate (SiLU, exp, rsqrt, cos, sin)."""
        return self._add_node(Op.LUT, [x], {"fn": fn_name}, name=name)

    # ---- IEEE-754 arithmetic ------------------------------------

    def add(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.ADD, [a, b], name=name)

    def sub(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.SUB, [a, b], name=name)

    def mul(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.MUL, [a, b], name=name)

    def div(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.DIV, [a, b], name=name)

    def neg(self, x: int, name: str = "") -> int:
        return self._add_node(Op.NEG, [x], name=name)

    def abs(self, x: int, name: str = "") -> int:
        return self._add_node(Op.ABS, [x], name=name)

    def square(self, x: int, name: str = "") -> int:
        return self._add_node(Op.SQUARE, [x], name=name)

    # ---- Comparison / selection ---------------------------------

    def max(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.MAX, [a, b], name=name)

    def cmp_le(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.CMP_LE, [a, b], name=name)

    def mux(self, cond: int, a: int, b: int, name: str = "") -> int:
        """Select: cond==0 → a, cond!=0 → b."""
        return self._add_node(Op.MUX, [cond, a, b], name=name)

    # ---- Linear algebra -----------------------------------------

    def matmul(self, a: int, b: int, name: str = "") -> int:
        return self._add_node(Op.MATMUL, [a, b], name=name)

    # ---- Reductions ---------------------------------------------

    def sum(self, x: int, axis: int = -1,
            keepdims: bool = False, name: str = "") -> int:
        return self._add_node(Op.SUM, [x],
                              {"axis": axis, "keepdims": keepdims}, name=name)

    def max_reduce(self, x: int, axis: int = -1,
                   keepdims: bool = False, name: str = "") -> int:
        return self._add_node(Op.MAX_REDUCE, [x],
                              {"axis": axis, "keepdims": keepdims}, name=name)

    def argmax(self, x: int, axis: int = -1, name: str = "") -> int:
        return self._add_node(Op.ARGMAX, [x], {"axis": axis}, name=name)

    def mean(self, x: int, axis: int = -1,
             keepdims: bool = False, name: str = "") -> int:
        return self._add_node(Op.MEAN, [x],
                              {"axis": axis, "keepdims": keepdims}, name=name)

    # ---- Wire routing (zero gates) ------------------------------

    def reshape(self, x: int, shape: tuple[int, ...],
                name: str = "") -> int:
        return self._add_node(Op.RESHAPE, [x], {"shape": shape}, name=name)

    def transpose(self, x: int, axes: tuple[int, ...],
                  name: str = "") -> int:
        return self._add_node(Op.TRANSPOSE, [x], {"axes": axes}, name=name)

    def concat(self, inputs: list[int], axis: int = 0,
               name: str = "") -> int:
        return self._add_node(Op.CONCAT, inputs, {"axis": axis}, name=name)

    def repeat(self, x: int, repeats: int, axis: int = 0,
               name: str = "") -> int:
        return self._add_node(Op.REPEAT, [x],
                              {"repeats": repeats, "axis": axis}, name=name)

    def slice(self, x: int, slices: tuple, name: str = "") -> int:
        return self._add_node(Op.SLICE, [x], {"slices": slices}, name=name)

    def expand_dims(self, x: int, axis: int | tuple[int, ...],
                    name: str = "") -> int:
        return self._add_node(Op.EXPAND_DIMS, [x],
                              {"axis": axis}, name=name)

    def arange(self, start: int, stop: int, dtype: np.dtype = np.dtype(np.float32),
               name: str = "") -> int:
        v = np.arange(start, stop).astype(dtype)
        return self.const(v, name=name)

    # ---- Cast / view --------------------------------------------

    def cast(self, x: int, dtype: np.dtype, name: str = "") -> int:
        return self._add_node(Op.CAST, [x], {"dtype": dtype},
                              dtype=dtype, name=name)

    # ---- Composite subgraphs ------------------------------------

    def softmax(self, x: int, axis: int = -1, name: str = "") -> int:
        """Softmax decomposed into primitive nodes.

        Steps: max_reduce → sub → exp LUT → sum → div.
        """
        m = self.max_reduce(x, axis=axis, keepdims=True,
                            name=f"{name}/max")
        shifted = self.sub(x, m, name=f"{name}/shift")
        e = self.lut(shifted, "exp", name=f"{name}/exp")
        s = self.sum(e, axis=axis, keepdims=True, name=f"{name}/sum")
        return self.div(e, s, name=f"{name}/div")

    def rms_norm(self, x: int, weight: int, eps_val: float,
                 name: str = "") -> int:
        """RMSNorm: x * rsqrt(mean(x²) + eps) * weight."""
        sq = self.square(x, name=f"{name}/sq")
        var = self.mean(sq, axis=-1, keepdims=True,
                        name=f"{name}/var")
        eps = self.const(np.float32(eps_val), name=f"{name}/eps")
        var_eps = self.add(var, eps, name=f"{name}/var_eps")
        scale = self.lut(var_eps, "rsqrt", name=f"{name}/rsqrt")
        normed = self.mul(x, scale, name=f"{name}/normed")
        return self.mul(normed, weight, name=f"{name}/mul_w")

    # ---- Topology -----------------------------------------------

    def topological_order(self) -> list[int]:
        """Return node IDs in topological order (inputs before outputs)."""
        visited: set[int] = set()
        order: list[int] = []

        def visit(nid: int) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for inp in self.nodes[nid].inputs:
                visit(inp)
            order.append(nid)

        for node in self.nodes:
            visit(node.id)
        return order

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        ops = {}
        for n in self.nodes:
            ops[n.op.value] = ops.get(n.op.value, 0) + 1
        counts = ", ".join(f"{k}={v}" for k, v in sorted(ops.items()))
        return f"CircuitGraph({len(self.nodes)} nodes: {counts})"

    # ---- Gate counting ------------------------------------------

    def gate_count(self) -> dict[str, int]:
        """Count nodes by type.  Wire-routing ops have zero gates."""
        zero_gate_ops = {Op.RESHAPE, Op.TRANSPOSE, Op.CONCAT,
                         Op.REPEAT, Op.SLICE, Op.CONST, Op.INPUT, Op.CAST,
                         Op.EXPAND_DIMS}
        counts: dict[str, int] = {"total": 0, "wire": 0, "gate": 0}
        for n in self.nodes:
            counts[n.op.value] = counts.get(n.op.value, 0) + 1
            if n.op in zero_gate_ops:
                counts["wire"] += 1
            else:
                counts["gate"] += 1
            counts["total"] += 1
        return counts


# ---------------------------------------------------------------
# Reference evaluator — golden output via NumPy
# ---------------------------------------------------------------
# The evaluator walks the graph in topological order and computes
# each node using NumPy.  This is the REFERENCE implementation
# that the C executor must match bit-for-bit.
#
# It uses float arithmetic (NumPy FPU) — that's correct!  The
# graph defines WHAT to compute; the evaluator defines HOW for
# validation.  The C executor will use byte-plane gates instead.
# ---------------------------------------------------------------

# LUT function registry (maps name → NumPy function)
_LUT_REGISTRY: dict[str, callable] = {}


def register_lut(name: str, fn: callable) -> None:
    """Register a NumPy function as a LUT evaluator."""
    _LUT_REGISTRY[name] = fn


def _silu_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _exp_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return np.exp(x.astype(np.float64)).astype(np.float32)


def _rsqrt_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float32)


def _cos_fn(x: np.ndarray) -> np.ndarray:
    return np.cos(x.astype(np.float64)).astype(np.float32)


def _sin_fn(x: np.ndarray) -> np.ndarray:
    return np.sin(x.astype(np.float64)).astype(np.float32)


# Register built-in LUTs
for _name, _fn in [("silu", _silu_fn), ("exp", _exp_fn),
                    ("rsqrt", _rsqrt_fn), ("cos", _cos_fn),
                    ("sin", _sin_fn)]:
    register_lut(_name, _fn)


def evaluate(graph: CircuitGraph,
             inputs: dict[int, np.ndarray] | None = None,
             ) -> dict[int, np.ndarray]:
    """Reference evaluation of a circuit graph via NumPy.

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
    inputs = inputs or {}
    values: dict[int, np.ndarray] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]
        inp = [values[i] for i in node.inputs]

        if node.op == Op.CONST:
            values[nid] = node.params["value"]

        elif node.op == Op.INPUT:
            if nid not in inputs:
                raise ValueError(
                    f"Missing input for node {nid} ({node.name!r})")
            values[nid] = inputs[nid]

        elif node.op == Op.LUT:
            fn_name = node.params["fn"]
            if fn_name not in _LUT_REGISTRY:
                raise ValueError(f"Unknown LUT function: {fn_name!r}")
            values[nid] = _LUT_REGISTRY[fn_name](inp[0])

        elif node.op == Op.ADD:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           + np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.SUB:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           - np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.MUL:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           * np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.DIV:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           / np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.NEG:
            u = np.asarray(inp[0], dtype=np.float32).view(np.uint32)
            values[nid] = (u ^ np.uint32(0x80000000)).view(np.float32)

        elif node.op == Op.ABS:
            u = np.asarray(inp[0], dtype=np.float32).view(np.uint32)
            values[nid] = (u & np.uint32(0x7FFFFFFF)).view(np.float32)

        elif node.op == Op.SQUARE:
            x = np.asarray(inp[0], dtype=np.float32)
            values[nid] = x * x

        elif node.op == Op.MAX:
            values[nid] = np.maximum(
                np.asarray(inp[0], dtype=np.float32),
                np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.CMP_LE:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           <= np.asarray(inp[1], dtype=np.float32)
                           ).astype(np.uint8)

        elif node.op == Op.MUX:
            cond = np.asarray(inp[0]).astype(bool)
            values[nid] = np.where(
                cond,
                np.asarray(inp[2], dtype=np.float32),
                np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.MATMUL:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           @ np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.SUM:
            values[nid] = np.asarray(inp[0], dtype=np.float32).sum(
                axis=node.params["axis"],
                keepdims=node.params.get("keepdims", False))

        elif node.op == Op.MAX_REDUCE:
            values[nid] = np.asarray(inp[0], dtype=np.float32).max(
                axis=node.params["axis"],
                keepdims=node.params.get("keepdims", False))

        elif node.op == Op.ARGMAX:
            values[nid] = np.asarray(inp[0], dtype=np.float32).argmax(
                axis=node.params["axis"])

        elif node.op == Op.MEAN:
            values[nid] = np.asarray(inp[0], dtype=np.float32).mean(
                axis=node.params["axis"],
                keepdims=node.params.get("keepdims", False))

        elif node.op == Op.RESHAPE:
            values[nid] = np.asarray(inp[0]).reshape(node.params["shape"])

        elif node.op == Op.TRANSPOSE:
            values[nid] = np.asarray(inp[0]).transpose(node.params["axes"])

        elif node.op == Op.CONCAT:
            values[nid] = np.concatenate(inp, axis=node.params["axis"])

        elif node.op == Op.REPEAT:
            values[nid] = np.repeat(inp[0],
                                    node.params["repeats"],
                                    axis=node.params["axis"])

        elif node.op == Op.SLICE:
            values[nid] = inp[0][node.params["slices"]]

        elif node.op == Op.CAST:
            values[nid] = np.asarray(inp[0]).astype(node.params["dtype"])

        elif node.op == Op.EXPAND_DIMS:
            values[nid] = np.expand_dims(inp[0], axis=node.params["axis"])

        else:
            raise ValueError(f"Unknown op: {node.op}")

    return values
