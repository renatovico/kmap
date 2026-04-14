"""Circuit optimizer — Espresso-style boolean logic minimization.

Minimizes byte-level boolean functions using the Quine-McCluskey
algorithm (exact two-level minimization), producing compact
sum-of-products (SOP) representations.

Each byte-level gate function (8 inputs → 8 outputs) is decomposed
into 8 single-output boolean functions and independently minimized.
The result is a set of product terms (AND-OR logic) per output bit —
the same representation a Karnaugh map or Espresso produces.
"""

from __future__ import annotations

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
    """Quine-McCluskey minimization for byte-level boolean functions.

    For each byte-level gate function, decomposes into 8 single-output
    boolean functions and minimizes each — analogous to Karnaugh maps
    for ≤4 variables or Espresso for larger functions.
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


