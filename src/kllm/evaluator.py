"""Reference evaluator — golden output via NumPy.

The evaluator walks the graph in topological order and computes
each node using NumPy.  This is the REFERENCE implementation
that the C executor must match bit-for-bit.

It uses float arithmetic (NumPy FPU) — that's correct!  The
graph defines WHAT to compute; the evaluator defines HOW for
validation.  The C executor will use byte-plane gates instead.
"""

from __future__ import annotations

import numpy as np

from kllm.circuit_graph import CircuitGraph, Op


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

        elif node.op == Op.MATMUL_Q8:
            # inp[0]=activation(f32), inp[1]=weight(int8), inp[2]=scales(f32)
            x = np.asarray(inp[0], dtype=np.float32)
            w_q8 = np.asarray(inp[1])
            scales = np.asarray(inp[2], dtype=np.float32)
            values[nid] = (x @ w_q8.astype(np.float32)) * scales

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
