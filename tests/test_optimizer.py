"""Tests for the Quine-McCluskey circuit optimizer."""

import numpy as np
import pytest

from kllm.optimizer import (
    OptimizedCircuit,
    minimize_boolean,
    quine_mccluskey,
    sop_to_expr,
)


# ---------------------------------------------------------------
# Quine-McCluskey core
# ---------------------------------------------------------------

class TestQuineMcCluskey:
    def test_empty_on_set(self):
        assert quine_mccluskey(set(), n_vars=4) == []

    def test_single_minterm(self):
        primes = quine_mccluskey({5}, n_vars=4)
        # Single minterm 0101 → one fully-specified PI
        assert len(primes) == 1
        val, mask = primes[0]
        assert val == 5
        assert mask == 0b1111

    def test_two_adjacent_minterms(self):
        """Minterms 4 (0100) and 5 (0101) combine into 010-."""
        primes = quine_mccluskey({4, 5}, n_vars=4)
        assert len(primes) == 1
        val, mask = primes[0]
        # bit 0 is don't-care, bits 1-3 fixed
        assert mask == 0b1110
        assert (val & mask) == 0b0100

    def test_identity_bit0(self):
        """f(x) = x's bit 0 → 128 minterms collapse to 1 PI."""
        on_set = {i for i in range(256) if i & 1}
        primes = quine_mccluskey(on_set, n_vars=8)
        # Should reduce to a single PI: bit 0 = 1, rest don't-care
        cover = minimize_boolean(
            np.array([i & 1 for i in range(256)], dtype=np.uint8),
            n_vars=8,
        )
        assert len(cover) == 1
        val, mask = cover[0]
        assert mask == 0b00000001  # only bit 0 matters
        assert val & mask == 1     # bit 0 must be 1

    def test_tautology(self):
        """All 256 minterms → tautology (single term, empty mask)."""
        tt = np.ones(256, dtype=np.uint8)
        cover = minimize_boolean(tt, n_vars=8)
        assert len(cover) == 1
        assert cover == [(0, 0)]

    def test_constant_zero(self):
        tt = np.zeros(256, dtype=np.uint8)
        cover = minimize_boolean(tt, n_vars=8)
        assert cover == []


# ---------------------------------------------------------------
# SOP expression rendering
# ---------------------------------------------------------------

class TestSopExpr:
    def test_zero(self):
        assert sop_to_expr([]) == "0"

    def test_tautology(self):
        assert sop_to_expr([(0, 0)]) == "1"

    def test_single_literal(self):
        # bit 2 = 1, rest don't-care → "b2"
        expr = sop_to_expr([(4, 4)], n_vars=8)
        assert "b2" in expr
        assert "~" not in expr

    def test_negated_literal(self):
        # bit 3 = 0, rest don't-care → "~b3"
        expr = sop_to_expr([(0, 8)], n_vars=8)
        assert "~b3" in expr


# ---------------------------------------------------------------
# Identity function (constant gate) optimisation
# ---------------------------------------------------------------

class TestIdentityOptimization:
    def test_identity_minimizes_to_8_terms(self):
        """Identity byte→byte should be 8 terms: one per bit."""
        identity_tt = np.arange(256, dtype=np.uint8)
        circuit = OptimizedCircuit()
        for bit in range(8):
            bit_tt = np.array(
                [(int(identity_tt[i]) >> bit) & 1 for i in range(256)],
                dtype=np.uint8,
            )
            terms = minimize_boolean(bit_tt, n_vars=8)
            circuit.bit_sops.append(terms)
            circuit.total_terms += len(terms)

        assert circuit.total_terms == 8

    def test_identity_evaluates_correctly(self):
        """Minimized identity circuit reproduces all 256 values."""
        identity_tt = np.arange(256, dtype=np.uint8)
        circuit = OptimizedCircuit()
        for bit in range(8):
            bit_tt = np.array(
                [(int(identity_tt[i]) >> bit) & 1 for i in range(256)],
                dtype=np.uint8,
            )
            terms = minimize_boolean(bit_tt, n_vars=8)
            circuit.bit_sops.append(terms)
            circuit.total_terms += len(terms)

        x = np.arange(256, dtype=np.uint8)
        result = circuit.evaluate(x)
        np.testing.assert_array_equal(result, x)


# ---------------------------------------------------------------
# Arbitrary byte function
# ---------------------------------------------------------------

class TestByteFunctionOptimization:
    def test_xor_pattern(self):
        """XOR with constant: f(x) = x ^ 0xAA."""
        truth_table = np.arange(256, dtype=np.uint8) ^ np.uint8(0xAA)
        circuit = OptimizedCircuit()
        for bit in range(8):
            bit_tt = np.array(
                [(int(truth_table[i]) >> bit) & 1 for i in range(256)],
                dtype=np.uint8,
            )
            terms = minimize_boolean(bit_tt, n_vars=8)
            circuit.bit_sops.append(terms)
            circuit.total_terms += len(terms)

        # Should still be 8 terms (each bit is just inverted or not)
        assert circuit.total_terms == 8

        x = np.arange(256, dtype=np.uint8)
        result = circuit.evaluate(x)
        np.testing.assert_array_equal(result, x ^ np.uint8(0xAA))

    def test_random_function_roundtrip(self):
        """Random byte function: optimize → evaluate matches truth table."""
        rng = np.random.default_rng(42)
        truth_table = rng.integers(0, 256, size=256).astype(np.uint8)

        circuit = OptimizedCircuit()
        for bit in range(8):
            bit_tt = np.array(
                [(int(truth_table[i]) >> bit) & 1 for i in range(256)],
                dtype=np.uint8,
            )
            terms = minimize_boolean(bit_tt, n_vars=8)
            circuit.bit_sops.append(terms)
            circuit.total_terms += len(terms)

        x = np.arange(256, dtype=np.uint8)
        result = circuit.evaluate(x)
        np.testing.assert_array_equal(result, truth_table)
