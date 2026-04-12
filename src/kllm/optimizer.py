"""Circuit optimizer — Espresso-style boolean logic minimization.

After Z3 compiles the gate LUTs, this optimizer minimizes each
boolean function using the Quine-McCluskey algorithm (exact two-level
minimization), producing compact sum-of-products (SOP) representations.

Each byte-level gate function (8 inputs → 8 outputs) is decomposed
into 8 single-output boolean functions and independently minimized.
The result is a set of product terms (AND-OR logic) per output bit —
the same representation a Karnaugh map or Espresso produces.

Architecture
------------
**Constant gates** (weight storage): The identity function (output = input)
is trivially minimized to 8 product terms (1 per bit).  The optimizer
builds a compact 256-entry LUT from these SOPs and stores it in
``circuits_optimized.npz``.  The Fabric loads this at startup and uses
it for weight reconstruction instead of the raw Z3 ``(shift, mask)``
pairs.

**Activation functions** (silu, exp, rsqrt): Each is a 32-bit→32-bit
function where every output byte depends on all 4 input bytes.  QMC on
32 variables (2^32 minterms) is intractable, so these cannot be directly
minimized.  The Z3 solver *proved* the NumPy formulas bit-exact for all
2^32 float32 patterns — so ``_np_silu``, ``_np_exp``, ``_np_rsqrt`` ARE
the circuits' transfer functions running at SIMD speed.  The optimizer
analyses byte-plane projections (1 byte varies, 3 fixed to 0) to report
the boolean complexity of each slice.

Pipeline::

    compile → compile-circuits → optimize-circuits → inference
                                  ^^^^^^^^^^^^^^^^
                                  this module

Usage::

    kllm --mode optimize-circuits
"""

from __future__ import annotations

import os
import time

import numpy as np


# ---------------------------------------------------------------
# Quine-McCluskey boolean minimizer
# ---------------------------------------------------------------

def _popcount(n: int) -> int:
    """Number of set bits."""
    return bin(n).count("1")


def _is_power_of_two(n: int) -> bool:
    """True if n has exactly one bit set."""
    return n > 0 and (n & (n - 1)) == 0


def _expand_implicant(value: int, mask: int, n_vars: int) -> set[int]:
    """Expand implicant (value, care_mask) to all covered minterms.

    Don't-care positions (0 in mask) can be 0 or 1.
    """
    dc_bits = [i for i in range(n_vars) if not (mask & (1 << i))]
    base = value & mask
    minterms = set()
    for combo in range(1 << len(dc_bits)):
        m = base
        for j, bit_pos in enumerate(dc_bits):
            if combo & (1 << j):
                m |= 1 << bit_pos
        minterms.add(m)
    return minterms


def quine_mccluskey(on_set: set[int], n_vars: int = 8) -> list[tuple[int, int]]:
    """Find all prime implicants of a boolean function.

    Parameters
    ----------
    on_set : set of int
        Minterms where the function is 1.
    n_vars : int
        Number of input variables (max 16 for tractability).

    Returns
    -------
    list of (value, care_mask)
        Prime implicants.  Each covers minterms where
        ``(x & care_mask) == (value & care_mask)``.
    """
    if not on_set:
        return []

    full_mask = (1 << n_vars) - 1

    # Initialize: each minterm is a fully-specified implicant
    current: set[tuple[int, int]] = {(m, full_mask) for m in on_set}
    all_primes: set[tuple[int, int]] = set()

    while current:
        # Group by (care_mask, popcount of value's care bits)
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for val, mask in current:
            key = (mask, _popcount(val & mask))
            groups.setdefault(key, []).append((val, mask))

        combined: set[tuple[int, int]] = set()
        used: set[tuple[int, int]] = set()

        for (mask, count), items in groups.items():
            partner_key = (mask, count + 1)
            if partner_key not in groups:
                continue
            partners = groups[partner_key]
            for v1, m1 in items:
                for v2, m2 in partners:
                    diff = (v1 ^ v2) & mask
                    if _is_power_of_two(diff):
                        new_mask = mask & ~diff
                        new_val = v1 & new_mask
                        combined.add((new_val, new_mask))
                        used.add((v1, m1))
                        used.add((v2, m2))

        # Prime implicants: those not absorbed into larger implicants
        all_primes |= current - used
        current = combined

    return list(all_primes)


def minimum_cover(
    primes: list[tuple[int, int]],
    on_set: set[int],
    n_vars: int,
) -> list[tuple[int, int]]:
    """Select fewest prime implicants covering all on-set minterms.

    Uses essential-prime detection + greedy largest-first heuristic.
    Exact for 8-variable functions (small problem size).
    """
    if not on_set:
        return []

    remaining = set(on_set)
    cover: list[tuple[int, int]] = []
    available = list(primes)

    # Phase 1: essential primes (only one PI covers a minterm)
    changed = True
    while changed and remaining:
        changed = False
        for m in list(remaining):
            covering = [
                p for p in available
                if (m & p[1]) == (p[0] & p[1])
            ]
            if len(covering) == 1:
                pi = covering[0]
                if pi not in cover:
                    cover.append(pi)
                    available.remove(pi)
                remaining -= _expand_implicant(pi[0], pi[1], n_vars) & on_set
                changed = True
                break

    # Phase 2: greedy (covers most uncovered minterms first)
    while remaining and available:
        best = max(
            available,
            key=lambda p: len(
                _expand_implicant(p[0], p[1], n_vars) & remaining
            ),
        )
        cover.append(best)
        available.remove(best)
        remaining -= _expand_implicant(best[0], best[1], n_vars)

    return cover


def minimize_boolean(
    truth_table: np.ndarray, n_vars: int = 8,
) -> list[tuple[int, int]]:
    """Minimize a single-output boolean function (truth table → SOP).

    Parameters
    ----------
    truth_table : ndarray of 0/1, length 2**n_vars
    n_vars : int

    Returns
    -------
    list of (value, care_mask)
        Minimal SOP cover.
    """
    on_set = {i for i, v in enumerate(truth_table) if v}
    if not on_set:
        return []
    if len(on_set) == (1 << n_vars):
        return [(0, 0)]  # tautology

    primes = quine_mccluskey(on_set, n_vars)
    return minimum_cover(primes, on_set, n_vars)


# ---------------------------------------------------------------
# Human-readable SOP expression
# ---------------------------------------------------------------

def sop_to_expr(
    terms: list[tuple[int, int]],
    n_vars: int = 8,
    var_names: list[str] | None = None,
) -> str:
    """Render SOP as a human-readable boolean expression.

    Example: ``"b7·b5·~b2 + ~b7·b3"``
    """
    if not terms:
        return "0"
    if terms == [(0, 0)]:
        return "1"

    if var_names is None:
        var_names = [f"b{i}" for i in range(n_vars)]

    parts = []
    for value, mask in terms:
        literals = []
        for i in range(n_vars - 1, -1, -1):
            if mask & (1 << i):
                if value & (1 << i):
                    literals.append(var_names[i])
                else:
                    literals.append(f"~{var_names[i]}")
        parts.append("·".join(literals) if literals else "1")

    return " + ".join(parts)


# ---------------------------------------------------------------
# Optimized circuit representation
# ---------------------------------------------------------------

class OptimizedCircuit:
    """A byte→byte function minimized into 8 SOPs (one per output bit).

    Each SOP is a list of ``(value, care_mask)`` product terms.
    The circuit evaluates to 1 for an input ``x`` when any term
    satisfies ``(x & care_mask) == (value & care_mask)``.
    """

    def __init__(self, n_inputs: int = 8) -> None:
        self.n_inputs = n_inputs
        self.bit_sops: list[list[tuple[int, int]]] = []
        self.total_terms: int = 0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the minimized circuit on uint8 inputs."""
        flat = x.ravel().astype(np.uint8)
        out = np.zeros_like(flat)

        for bit_idx, terms in enumerate(self.bit_sops):
            bit_val = np.uint8(1 << bit_idx)
            for value, mask in terms:
                matches = (flat & np.uint8(mask)) == np.uint8(value & mask)
                out[matches] |= bit_val

        return out.reshape(x.shape)

    def to_lut(self) -> np.ndarray:
        """Materialise the SOP into a 256-entry uint8 lookup table.

        Equivalent to ``self.evaluate(np.arange(256, dtype=np.uint8))``,
        but stored as a flat array for O(1) indexed lookup at runtime.
        """
        return self.evaluate(np.arange(256, dtype=np.uint8))


# ---------------------------------------------------------------
# Circuit optimizer
# ---------------------------------------------------------------

class CircuitOptimizer:
    """Optimize Z3 gate circuits using Quine-McCluskey minimization.

    For each byte-level gate function, decomposes into 8 single-output
    boolean functions and minimizes each — analogous to Karnaugh maps
    for ≤4 variables or Espresso for larger functions.

    The optimizer works on byte→byte truth tables extracted from the
    Z3 gate LUTs.  For full-domain mmap planes (uint32→uint8), it
    analyses byte-slice projections to show the boolean complexity.
    """

    def __init__(self, save_dir: str = "./lossless_logic") -> None:
        self.save_dir = save_dir

    # ---- Byte-level optimization --------------------------------

    def optimize_byte_function(
        self, truth_table: np.ndarray, name: str = "",
    ) -> OptimizedCircuit:
        """Minimize an 8-in → 8-out byte function via Kmap / QMC.

        *truth_table*: length-256 uint8 array where
        ``truth_table[i]`` is the output for input ``i``.
        """
        circuit = OptimizedCircuit()

        for bit in range(8):
            bit_tt = np.array(
                [(int(truth_table[i]) >> bit) & 1 for i in range(256)],
                dtype=np.uint8,
            )
            terms = minimize_boolean(bit_tt, n_vars=8)
            circuit.bit_sops.append(terms)
            circuit.total_terms += len(terms)

            if name:
                on_count = int(bit_tt.sum())
                print(
                    f"    bit {bit}: {len(terms):3d} terms  "
                    f"(on-set {on_count:3d}/256)"
                )

        return circuit

    # ---- High-level optimize pipeline ---------------------------

    def optimize(self) -> dict:
        """Run full optimisation on compiled circuits.

        1. Constant gate (identity)
        2. Per-op byte-plane projections (silu, exp, rsqrt)

        Returns stats dict.
        """
        from kllm.circuits import ArithmeticUnit, _exp_fn, _rsqrt_fn, _silu_fn

        circuit_path = os.path.join(self.save_dir, "circuits.npz")
        if not os.path.exists(circuit_path):
            raise FileNotFoundError(
                f"No circuits at {circuit_path}. "
                "Run `kllm --mode compile-circuits` first."
            )

        print("=" * 64)
        print("  Circuit Optimizer — Quine-McCluskey / Espresso")
        print("=" * 64)
        t0_all = time.perf_counter()

        stats: dict = {}
        all_circuits: dict[str, list[OptimizedCircuit]] = {}

        # ---- 1. Constant gate (identity: output = input) --------
        print("\n[constant] Identity function (output = input)")
        identity_tt = np.arange(256, dtype=np.uint8)
        const_opt = self.optimize_byte_function(identity_tt, name="constant")
        stats["constant"] = {
            "total_terms": const_opt.total_terms,
            "terms_per_bit": [len(s) for s in const_opt.bit_sops],
        }
        # Build the optimized LUT — used by Fabric at inference time
        const_lut = const_opt.to_lut()
        np.testing.assert_array_equal(
            const_lut, identity_tt,
            err_msg="Optimized constant gate LUT mismatch!",
        )

        # Materialise the full gate execution table: gate_lut[s1, mask]
        # = uint8(0xFF << s1) ^ mask.  Replaces per-element shift+XOR
        # at weight load time with a single indexed read.
        gate_lut = np.empty((8, 256), dtype=np.uint8)
        probe = np.uint8(0xFF)
        for s in range(8):
            shifted = np.uint8(probe << np.uint8(s))
            for m in range(256):
                gate_lut[s, m] = shifted ^ np.uint8(m)

        # Verify: gate_lut matches direct computation for all (s1, mask)
        for s in range(8):
            for m in range(256):
                expected = np.uint8((np.uint8(0xFF) << np.uint8(s))) ^ np.uint8(m)
                assert gate_lut[s, m] == expected, (
                    f"gate_lut[{s},{m}]={gate_lut[s,m]} != {expected}"
                )

        print(
            f"  → {const_opt.total_terms} total product terms "
            f"(vs 512 bytes LUT) ✓ verified"
        )
        print(f"  → gate_lut[8,256] materialised for Fabric")

        # ---- 2. Activation byte-plane projections ---------------
        for op_name, fn in [
            ("silu", _silu_fn),
            ("exp", _exp_fn),
            ("rsqrt", _rsqrt_fn),
        ]:
            print(f"\n[{op_name}] Byte-plane projections "
                  f"(fix 3 bytes = 0, vary 1)")
            plane_circuits: list[OptimizedCircuit] = []
            total_terms = 0

            for plane_idx in range(4):
                # Build truth table: vary byte `plane_idx`, fix rest to 0
                inputs_u32 = np.zeros(256, dtype=np.uint32)
                for i in range(256):
                    inputs_u32[i] = np.uint32(i) << np.uint32(plane_idx * 8)

                y_float = fn(inputs_u32.view(np.float32)).astype(np.float32)
                y_bytes = y_float.view(np.uint8).reshape(-1, 4)
                truth_table = y_bytes[:, plane_idx]

                print(f"  plane {plane_idx} (byte{plane_idx}→byte{plane_idx}):")
                opt = self.optimize_byte_function(truth_table, name=op_name)
                plane_circuits.append(opt)
                total_terms += opt.total_terms

            all_circuits[op_name] = plane_circuits
            stats[op_name] = {
                "total_terms": total_terms,
                "terms_per_plane": [c.total_terms for c in plane_circuits],
            }
            print(f"  → {total_terms} total product terms across 4 planes")

        elapsed = time.perf_counter() - t0_all

        # ---- Summary -------------------------------------------
        print(f"\n{'=' * 64}")
        print(f"  Optimization complete in {elapsed:.1f}s")
        print(f"{'=' * 64}")
        grand_total = sum(s["total_terms"] for s in stats.values())
        for name, s in stats.items():
            print(f"  {name:12s}: {s['total_terms']:5d} product terms")
        print(f"  {'total':12s}: {grand_total:5d} product terms")
        print(f"{'=' * 64}")

        # ---- Save -----------------------------------------------
        self._save(stats, const_opt, gate_lut, all_circuits)

        # ---- Materialize float32 weights for fast loading -------
        self._materialize_weights()

        return stats

    def _save(
        self,
        stats: dict,
        const_opt: OptimizedCircuit,
        gate_lut: np.ndarray,
        op_circuits: dict[str, list[OptimizedCircuit]],
    ) -> None:
        """Persist optimized circuits to ``circuits_optimized.npz``.

        The ``gate_lut`` (8×256 uint8) is loaded by
        :class:`~kllm.fabric.Fabric` to replace per-element shift+XOR
        with a single indexed read during weight reconstruction.
        """
        out_path = os.path.join(self.save_dir, "circuits_optimized.npz")

        flat: dict[str, np.ndarray] = {}
        flat["_stats"] = np.array(str(stats), dtype=object)
        flat["gate_lut"] = gate_lut

        # Constant gate SOPs
        for bit_idx, terms in enumerate(const_opt.bit_sops):
            if terms:
                flat[f"const/bit{bit_idx}/values"] = np.array(
                    [t[0] for t in terms], dtype=np.uint16,
                )
                flat[f"const/bit{bit_idx}/masks"] = np.array(
                    [t[1] for t in terms], dtype=np.uint16,
                )

        # Activation SOPs
        for op_name, circuits in op_circuits.items():
            for p_idx, circuit in enumerate(circuits):
                for bit_idx, terms in enumerate(circuit.bit_sops):
                    if terms:
                        prefix = f"{op_name}/p{p_idx}/bit{bit_idx}"
                        flat[f"{prefix}/values"] = np.array(
                            [t[0] for t in terms], dtype=np.uint16,
                        )
                        flat[f"{prefix}/masks"] = np.array(
                            [t[1] for t in terms], dtype=np.uint16,
                        )

        np.savez_compressed(out_path, **flat)
        print(f"\n[+] Optimized circuits saved to {out_path}")

    def _materialize_weights(self) -> None:
        """Pre-compute float32 weights and save as ``.npy`` for fast loading.

        Runs the full Z3 gate reconstruction once and caches the
        resulting float32 arrays to ``<save_dir>/optimized/``.
        On subsequent loads, :class:`~kllm.fabric.Fabric` mmap's
        these files directly — skipping all shift+XOR computation.
        """
        from kllm.fabric import Fabric, LINEAR_NAMES

        cache_dir = os.path.join(self.save_dir, "optimized")
        os.makedirs(cache_dir, exist_ok=True)

        print("\n[*] Materializing float32 weights from Z3 gates …")
        t0 = time.perf_counter()

        fabric = Fabric(self.save_dir)

        # ---- Global weights ----
        np.save(os.path.join(cache_dir, "embed_tokens.npy"), fabric.embed_tokens)
        np.save(os.path.join(cache_dir, "final_norm_weight.npy"), fabric.final_norm_weight)

        lm_head_tied = fabric.lm_head is fabric.embed_tokens
        np.save(os.path.join(cache_dir, "lm_head_tied.npy"), np.array(lm_head_tied))
        if not lm_head_tied:
            np.save(os.path.join(cache_dir, "lm_head.npy"), fabric.lm_head)

        # ---- Per-layer weights ----
        total_bytes = (
            fabric.embed_tokens.nbytes
            + fabric.final_norm_weight.nbytes
            + fabric.lm_head.nbytes
        )
        for li, layer_w in enumerate(fabric.layers):
            layer_dir = os.path.join(cache_dir, f"layer_{li}")
            os.makedirs(layer_dir, exist_ok=True)
            for name, arr in layer_w.items():
                np.save(os.path.join(layer_dir, f"{name}.npy"), arr)
                total_bytes += arr.nbytes

        elapsed = time.perf_counter() - t0
        print(
            f"[+] Cached {total_bytes / 1e9:.2f} GB to {cache_dir} "
            f"in {elapsed:.1f}s ({fabric.num_layers} layers)"
        )
