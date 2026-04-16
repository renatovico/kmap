"""Circuit graph — DAG of gate nodes representing the complete computation.

Every operation the transformer performs (matmul, add, multiply, norm,
softmax, embedding lookup, …) is a node in this graph.  The graph IS
the circuit.  It can be:

1. **Evaluated** by the reference evaluator (NumPy) — golden output.
2. **Executed** by the C gate executor (byte-plane shift+XOR) — Phase 4.
3. **Optimised** (constant folding, dead elimination, gate merge) — Phase 5.
4. **Synthesised** to Verilog/VHDL for FPGA — Phase 7.

IEEE-754 bit-manipulation does NOT belong here — execution is the
**executor's** responsibility, not the graph's.  The graph defines
WHAT to compute; the executor defines HOW.

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
- ``bpe_encode``: ROM-based BPE tokenizer (UTF-8 bytes → token IDs)
- ``bpe_decode``: ROM-based BPE detokenizer (token IDs → UTF-8 bytes)
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
    MATMUL_Q8   = "matmul_q8"    # quantized: x_f32 @ W_int8 * scales

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

    # Tokenizer (ROM-backed FSM)
    BPE_ENCODE  = "bpe_encode"   # UTF-8 bytes → token IDs
    BPE_DECODE  = "bpe_decode"   # token IDs → UTF-8 bytes


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

    def matmul_q8(self, x: int, weight_q8: int, scales: int,
                  name: str = "") -> int:
        """Quantized matmul: x_f32 @ W_int8 * scales_f32.

        Inputs: x (float32), weight_q8 (int8 CONST), scales (float32 CONST).
        The executor reads 4x less weight memory.
        """
        return self._add_node(Op.MATMUL_Q8, [x, weight_q8, scales],
                              name=name)

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

    # ---- Tokenizer (ROM-backed FSM) -----------------------------

    def bpe_encode(
        self,
        byte_input: int,
        byte_length: int,
        vocab_hash_keys: int,
        vocab_hash_vals: int,
        vocab_hash_lens: int,
        merge_a: int,
        merge_b: int,
        merge_result: int,
        special_ids: int,
        bos_token_id: int = 1,
        max_tokens: int = 2048,
        name: str = "",
    ) -> tuple[int, int]:
        """BPE encode: UTF-8 bytes → token IDs.

        Inputs:
          byte_input     — INPUT uint8[max_bytes], raw UTF-8
          byte_length    — INPUT int32 scalar, actual byte count
          vocab_hash_*   — CONST ROMs for piece→token_id lookup
          merge_a/b/result — CONST merge-priority ROM
          special_ids    — CONST int32 array of special token IDs

        Returns (token_ids_node, num_tokens_node):
          token_ids  — int32[max_tokens], padded with 0
          num_tokens — int32 scalar, actual count
        """
        token_ids = self._add_node(
            Op.BPE_ENCODE,
            [byte_input, byte_length, vocab_hash_keys, vocab_hash_vals,
             vocab_hash_lens, merge_a, merge_b, merge_result, special_ids],
            {"bos_token_id": bos_token_id, "max_tokens": max_tokens,
             "output_index": 0},
            shape=(max_tokens,),
            dtype=np.dtype(np.int32),
            name=f"{name}/token_ids" if name else "bpe_enc/token_ids",
        )
        num_tokens = self._add_node(
            Op.BPE_ENCODE,
            [byte_input, byte_length, vocab_hash_keys, vocab_hash_vals,
             vocab_hash_lens, merge_a, merge_b, merge_result, special_ids],
            {"bos_token_id": bos_token_id, "max_tokens": max_tokens,
             "output_index": 1, "paired_node": token_ids},
            shape=(1,),
            dtype=np.dtype(np.int32),
            name=f"{name}/num_tokens" if name else "bpe_enc/num_tokens",
        )
        return token_ids, num_tokens

    def bpe_decode(
        self,
        token_ids: int,
        num_tokens: int,
        id_to_bytes: int,
        id_to_offsets: int,
        special_ids: int,
        max_bytes: int = 8192,
        name: str = "",
    ) -> tuple[int, int]:
        """BPE decode: token IDs → UTF-8 bytes.

        Inputs:
          token_ids     — int32[max_tokens]
          num_tokens    — int32 scalar
          id_to_bytes   — CONST uint8 ROM (concatenated token strings)
          id_to_offsets — CONST int32[vocab_size, 2] (offset, length)
          special_ids   — CONST int32 array of special token IDs

        Returns (byte_output_node, byte_length_node):
          byte_output — uint8[max_bytes], padded with 0
          byte_length — int32 scalar, actual count
        """
        byte_output = self._add_node(
            Op.BPE_DECODE,
            [token_ids, num_tokens, id_to_bytes, id_to_offsets, special_ids],
            {"max_bytes": max_bytes, "output_index": 0},
            shape=(max_bytes,),
            dtype=np.dtype(np.uint8),
            name=f"{name}/byte_output" if name else "bpe_dec/byte_output",
        )
        byte_length = self._add_node(
            Op.BPE_DECODE,
            [token_ids, num_tokens, id_to_bytes, id_to_offsets, special_ids],
            {"max_bytes": max_bytes, "output_index": 1,
             "paired_node": byte_output},
            shape=(1,),
            dtype=np.dtype(np.int32),
            name=f"{name}/byte_length" if name else "bpe_dec/byte_length",
        )
        return byte_output, byte_length

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
        # BPE ops are FSM-driven ROM lookups — they have gate logic
        counts: dict[str, int] = {"total": 0, "wire": 0, "gate": 0}
        for n in self.nodes:
            counts[n.op.value] = counts.get(n.op.value, 0) + 1
            if n.op in zero_gate_ops:
                counts["wire"] += 1
            else:
                counts["gate"] += 1
            counts["total"] += 1
        return counts

    # ---- Evaluation (convenience re-export) ----------------------

    def evaluate(self, inputs: dict[int, np.ndarray] | None = None):
        """Evaluate using the reference evaluator (see evaluator.py)."""
        from kllm.evaluator import evaluate as _evaluate
        return _evaluate(self, inputs)

    # ---- Serialization ------------------------------------------

    def serialize(self, path: str) -> None:
        """Serialize the graph to a directory of flat files.

        Format (designed for C consumption):
        - ``nodes.bin``: packed node descriptors
        - ``topo.bin``: topological order (uint32 array)
        - ``const_NNN.bin``: raw tensor data for each const node
        """
        import os
        import struct
        os.makedirs(path, exist_ok=True)
        order = self.topological_order()

        # Op enum → integer mapping
        op_map = {op: i for i, op in enumerate(Op)}

        with open(os.path.join(path, "nodes.bin"), "wb") as f:
            # Header: num_nodes (uint32)
            f.write(struct.pack("<I", len(self.nodes)))
            for node in self.nodes:
                # node_id, op, num_inputs
                f.write(struct.pack("<III",
                                    node.id, op_map[node.op],
                                    len(node.inputs)))
                # input IDs
                for inp in node.inputs:
                    f.write(struct.pack("<I", inp))
                # Params: encode as key=value pairs, terminated by \0
                param_bytes = _encode_params(node.params)
                f.write(struct.pack("<I", len(param_bytes)))
                f.write(param_bytes)

        # Node shape/dtype/name metadata (separate file for compat)
        import json as _json
        node_meta_list = []
        for node in self.nodes:
            node_meta_list.append({
                "id": node.id,
                "shape": list(node.shape) if node.shape else None,
                "dtype": str(node.dtype),
                "name": node.name,
            })
        with open(os.path.join(path, "node_meta.json"), "w") as mf:
            _json.dump(node_meta_list, mf)

        # Topological order
        order_arr = np.array(order, dtype=np.uint32)
        order_arr.tofile(os.path.join(path, "topo.bin"))

        # Constant tensor data
        for node in self.nodes:
            if node.op == Op.CONST and "value" in node.params:
                val = np.ascontiguousarray(node.params["value"])
                val.tofile(os.path.join(path, f"const_{node.id}.bin"))
                # Also save shape/dtype metadata
                meta = {
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                }
                import json
                with open(os.path.join(path, f"const_{node.id}.json"),
                          "w") as mf:
                    json.dump(meta, mf)

    @classmethod
    def deserialize(cls, path: str) -> "CircuitGraph":
        """Load a graph from a serialized directory."""
        import os
        import struct
        import json

        g = cls()
        op_list = list(Op)

        # Load node shape/dtype metadata if available (newer format)
        meta_by_id: dict[int, dict] = {}
        meta_path = os.path.join(path, "node_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                for entry in json.load(mf):
                    meta_by_id[entry["id"]] = entry

        with open(os.path.join(path, "nodes.bin"), "rb") as f:
            num_nodes = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_nodes):
                nid, op_idx, num_inputs = struct.unpack("<III", f.read(12))
                inputs = [struct.unpack("<I", f.read(4))[0]
                          for _ in range(num_inputs)]
                param_len = struct.unpack("<I", f.read(4))[0]
                param_bytes = f.read(param_len)
                params = _decode_params(param_bytes)

                op = op_list[op_idx]

                # Restore shape/dtype/name from metadata
                nm = meta_by_id.get(nid, {})
                node_shape = tuple(nm["shape"]) if nm.get("shape") else None
                node_dtype = np.dtype(nm.get("dtype", "float32"))
                node_name = nm.get("name", "")

                # Load const data
                if op == Op.CONST:
                    const_path = os.path.join(path, f"const_{nid}.bin")
                    const_meta_path = os.path.join(
                        path, f"const_{nid}.json")
                    if os.path.exists(const_path):
                        with open(const_meta_path) as cmf:
                            cmeta = json.load(cmf)
                        dtype = np.dtype(cmeta["dtype"])
                        shape = tuple(cmeta["shape"])
                        data = np.fromfile(const_path, dtype=dtype)
                        params["value"] = data.reshape(shape)
                        node_shape = shape
                        node_dtype = dtype

                node = Node(
                    id=nid, op=op, inputs=inputs, params=params,
                    shape=node_shape, dtype=node_dtype, name=node_name,
                )
                g.nodes.append(node)
                g._next_id = max(g._next_id, nid + 1)

        return g


def _encode_params(params: dict) -> bytes:
    """Encode params dict to bytes for serialization."""
    import json
    # Filter out numpy arrays (handled separately for const nodes)
    filtered = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, np.dtype):
            filtered[k] = str(v)
        elif isinstance(v, (np.integer, np.floating)):
            filtered[k] = v.item()
        elif isinstance(v, slice):
            filtered[k] = {"__slice__": [v.start, v.stop, v.step]}
        elif isinstance(v, tuple) and any(isinstance(i, slice) for i in v):
            encoded = []
            for item in v:
                if isinstance(item, slice):
                    encoded.append({"__slice__": [item.start, item.stop,
                                                  item.step]})
                else:
                    encoded.append(item)
            filtered[k] = {"__tuple_slices__": encoded}
        else:
            filtered[k] = v
    return json.dumps(filtered).encode("utf-8")


def _decode_params(data: bytes) -> dict:
    """Decode params from bytes."""
    import json
    if not data:
        return {}
    params = json.loads(data.decode("utf-8"))
    # Reconstruct special types
    for k, v in list(params.items()):
        if isinstance(v, dict) and "__slice__" in v:
            s = v["__slice__"]
            params[k] = slice(s[0], s[1], s[2])
        elif isinstance(v, dict) and "__tuple_slices__" in v:
            items = []
            for item in v["__tuple_slices__"]:
                if isinstance(item, dict) and "__slice__" in item:
                    s = item["__slice__"]
                    items.append(slice(s[0], s[1], s[2]))
                else:
                    items.append(item)
            params[k] = tuple(items)
        elif k == "dtype":
            params[k] = np.dtype(v)
        elif k == "shape" and isinstance(v, list):
            params[k] = tuple(v)
        elif k == "axes" and isinstance(v, list):
            params[k] = tuple(v)
    return params
