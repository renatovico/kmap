"""Graph-level DAG optimizations for CircuitGraph.

These are **compile-time** optimizations on the circuit DAG itself.
Weights and model structure are fully known at compile time, so most
nodes in a compiled model graph have all-constant inputs and can be
folded into a single constant.

Passes
------
1. **Constant folding**  — evaluate nodes whose inputs are all constants
   and replace each chain with a single CONST node.
2. **Dead node elimination** — remove nodes not reachable from the
   output set.
3. **Identity elimination** — remove trivial nodes: add(x,0), mul(x,1),
   reshape to same shape, neg(neg(x)), etc.

All passes preserve the invariant:  output values of the optimized
graph == output values of the original graph (bit-exact).

Usage::

    from kllm.graph.graph_optimizer import optimize_graph
    opt_graph, new_ids = optimize_graph(graph, output_ids=[logits_id])
    # new_ids maps old output IDs to new IDs in the optimized graph
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from kllm.graph.circuit_graph import CircuitGraph, Node, Op
from kllm.graph.evaluator import evaluate


# ---------------------------------------------------------------
# Pass 1: Constant folding
# ---------------------------------------------------------------

def constant_fold(graph: CircuitGraph) -> CircuitGraph:
    """Fold nodes with all-constant inputs into CONST nodes.

    Walks the graph in topological order.  When a non-CONST,
    non-INPUT node has all inputs that are CONST (or transitively
    foldable), we evaluate it with the reference evaluator and
    replace it with a CONST node.

    Operates on a copy — the original graph is unchanged.
    """
    g = _copy_graph(graph)

    # First pass: identify which nodes are foldable
    is_const: set[int] = set()
    foldable: list[int] = []

    for nid in g.topological_order():
        node = g.nodes[nid]
        if node.op == Op.CONST:
            is_const.add(nid)
        elif node.op == Op.INPUT:
            continue
        elif all(inp in is_const for inp in node.inputs):
            is_const.add(nid)
            foldable.append(nid)

    if not foldable:
        return g

    # Evaluate only the constant subgraph.
    # Build a sub-graph containing only const + foldable nodes,
    # then evaluate it (no INPUT nodes → evaluate() won't fail).
    needed = set(foldable)
    # Also need all CONST ancestors
    def _add_ancestors(nid: int) -> None:
        for inp in g.nodes[nid].inputs:
            if inp not in needed:
                needed.add(inp)
                _add_ancestors(inp)
    for nid in foldable:
        _add_ancestors(nid)

    sub = CircuitGraph()
    id_map: dict[int, int] = {}
    for node in g.nodes:
        if node.id not in needed:
            continue
        new_inputs = [id_map[i] for i in node.inputs]
        new_id = sub._add_node(
            node.op, new_inputs,
            params=dict(node.params),
            shape=node.shape,
            dtype=node.dtype,
            name=node.name,
        )
        id_map[node.id] = new_id

    values = evaluate(sub)

    # Apply folding to the copy
    for nid in foldable:
        node = g.nodes[nid]
        val = values[id_map[nid]]
        if isinstance(val, np.ndarray):
            node.params = {"value": val.copy()}
        else:
            node.params = {"value": np.asarray(val)}
        node.op = Op.CONST
        node.inputs = []

    return g


# ---------------------------------------------------------------
# Pass 2: Dead node elimination
# ---------------------------------------------------------------

def dead_node_elimination(
    graph: CircuitGraph,
    keep_ids: list[int],
) -> tuple[CircuitGraph, dict[int, int]]:
    """Remove nodes not reachable backward from *keep_ids*.

    Builds a fresh graph containing only the reachable nodes,
    renumbered contiguously from 0.

    Returns (new_graph, id_map) where id_map[old_id] = new_id.
    """
    # Find all reachable nodes via backward traversal
    reachable: set[int] = set()

    def _mark(nid: int) -> None:
        if nid in reachable:
            return
        reachable.add(nid)
        for inp in graph.nodes[nid].inputs:
            _mark(inp)

    for nid in keep_ids:
        _mark(nid)

    # Build new graph with renumbered nodes
    new_graph = CircuitGraph()
    id_map: dict[int, int] = {}

    for node in graph.nodes:
        if node.id not in reachable:
            continue
        new_inputs = [id_map[i] for i in node.inputs]
        new_id = new_graph._add_node(
            node.op, new_inputs,
            params=dict(node.params),  # shallow copy
            shape=node.shape,
            dtype=node.dtype,
            name=node.name,
        )
        id_map[node.id] = new_id

    return new_graph, id_map


# ---------------------------------------------------------------
# Pass 3: Identity / trivial elimination
# ---------------------------------------------------------------

def identity_elimination(graph: CircuitGraph) -> CircuitGraph:
    """Remove trivial / identity operations.

    - add(x, 0) → x
    - sub(x, 0) → x
    - mul(x, 1) → x
    - div(x, 1) → x
    - neg(neg(x)) → x
    - reshape(x, same_shape) when shape already matches

    Operates on a copy.  Returns the modified graph with identity
    nodes replaced by their input (using an indirection map).
    """
    g = _copy_graph(graph)

    # Build a redirect map: node_id → replacement node_id
    redirect: dict[int, int] = {}

    def _resolve(nid: int) -> int:
        """Follow the redirect chain to the final node."""
        while nid in redirect:
            nid = redirect[nid]
        return nid

    def _is_zero_const(nid: int) -> bool:
        node = g.nodes[nid]
        if node.op != Op.CONST:
            return False
        v = node.params.get("value")
        if v is None:
            return False
        return np.all(np.asarray(v) == 0)

    def _is_one_const(nid: int) -> bool:
        node = g.nodes[nid]
        if node.op != Op.CONST:
            return False
        v = node.params.get("value")
        if v is None:
            return False
        return np.all(np.asarray(v) == 1)

    for nid in g.topological_order():
        node = g.nodes[nid]
        # Resolve inputs through redirects
        node.inputs = [_resolve(i) for i in node.inputs]

        if node.op == Op.ADD and len(node.inputs) == 2:
            if _is_zero_const(node.inputs[1]):
                redirect[nid] = node.inputs[0]
            elif _is_zero_const(node.inputs[0]):
                redirect[nid] = node.inputs[1]

        elif node.op == Op.SUB and len(node.inputs) == 2:
            if _is_zero_const(node.inputs[1]):
                redirect[nid] = node.inputs[0]

        elif node.op == Op.MUL and len(node.inputs) == 2:
            if _is_one_const(node.inputs[1]):
                redirect[nid] = node.inputs[0]
            elif _is_one_const(node.inputs[0]):
                redirect[nid] = node.inputs[1]

        elif node.op == Op.DIV and len(node.inputs) == 2:
            if _is_one_const(node.inputs[1]):
                redirect[nid] = node.inputs[0]

        elif node.op == Op.NEG and len(node.inputs) == 1:
            inp_node = g.nodes[node.inputs[0]]
            if inp_node.op == Op.NEG:
                redirect[nid] = inp_node.inputs[0]

    # Apply remaining redirects to all node inputs
    for node in g.nodes:
        node.inputs = [_resolve(i) for i in node.inputs]

    return g


# ---------------------------------------------------------------
# Combined optimizer
# ---------------------------------------------------------------

def optimize_graph(
    graph: CircuitGraph,
    output_ids: list[int],
) -> tuple[CircuitGraph, dict[int, int]]:
    """Run all graph optimization passes.

    Parameters
    ----------
    graph : CircuitGraph
        The unoptimized circuit graph.
    output_ids : list[int]
        Node IDs whose values must be preserved (e.g. logits).

    Returns
    -------
    (optimized_graph, id_map)
        id_map maps each old output_id to its new id in the optimized graph.
    """
    # 1. Identity elimination (before folding: fewer nodes to evaluate)
    g = identity_elimination(graph)

    # 2. Constant folding (the big one: evaluates entire const subgraphs)
    g = constant_fold(g)

    # 3. Dead node elimination (remove unreachable nodes, renumber)
    g, id_map = dead_node_elimination(g, output_ids)

    return g, id_map


def optimization_stats(
    original: CircuitGraph,
    optimized: CircuitGraph,
) -> dict:
    """Compute optimization statistics."""
    orig_counts = original.gate_count()
    opt_counts = optimized.gate_count()

    return {
        "original_nodes": orig_counts["total"],
        "optimized_nodes": opt_counts["total"],
        "original_gates": orig_counts["gate"],
        "optimized_gates": opt_counts["gate"],
        "nodes_removed": orig_counts["total"] - opt_counts["total"],
        "gate_reduction_pct": (
            100.0 * (1.0 - opt_counts["gate"] / max(orig_counts["gate"], 1))
        ),
        "original_breakdown": orig_counts,
        "optimized_breakdown": opt_counts,
    }


# ---------------------------------------------------------------
# Utility
# ---------------------------------------------------------------

def _copy_graph(graph: CircuitGraph) -> CircuitGraph:
    """Deep-copy a graph (nodes + params, shared const data)."""
    g = CircuitGraph()
    for node in graph.nodes:
        new_params = dict(node.params)
        # Share numpy array data (copy-on-write in constant_fold)
        g.nodes.append(Node(
            id=node.id,
            op=node.op,
            inputs=list(node.inputs),
            params=new_params,
            shape=node.shape,
            dtype=node.dtype,
            name=node.name,
        ))
    g._next_id = graph._next_id
    return g
