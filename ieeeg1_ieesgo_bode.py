"""Open-loop check: when linear IEEEG1 and IEESGO are the same model.

Input is speed error (omega_ref - omega). Output is mechanical power Pm.
Rate limits, deadband, and power limits are omitted, so this is the linear
transfer function only.

IEESGO draws the droop gain inside a controller lag and then a lead-lag.
IEEEG1 draws the gain on a lead-lag and then a valve lag. Those products are
the same rational function. Swapping T1 and T3 does not change it.

The turbine maps exactly only for a tandem-compound unit with the fourth
IEEEG1 stage removed. Bode plots confirm that identity. They do not recover
a unique parameter card in both directions.

Illustrative time constants, not a vendor data card. See README.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

s = sp.symbols("s")

# Illustrative tandem-compound machine, 5% droop on one power base.
K_DROOP = sp.Integer(20)
T_LEAD = sp.Rational(1, 2)
T_SLOW = sp.Integer(2)
T_FAST = sp.Rational(1, 10)
T_CHEST = sp.Rational(3, 10)
T_REHEAT = sp.Integer(8)
T_CROSS = sp.Rational(1, 2)
FRAC_HP = sp.Rational(3, 10)
FRAC_IP = sp.Rational(2, 5)
FRAC_LP = sp.Rational(3, 10)
K7_FOUR = sp.Rational(3, 20)
T7_FOUR = sp.Integer(1)

W_MIN = 1e-2
W_MAX = 1e2
N_FREQ = 600


def transfer_ieesgo(K1, T1, T2, T3, T4, T5, T6, K2, K3):
    """IEESGO: K1/(1+sT1) * (1+sT2)/(1+sT3) * chest * three steam fractions."""
    governor = K1 * (1 + s * T2) / ((1 + s * T1) * (1 + s * T3))
    steam = (
        (1 - K2)
        + K2 * (1 - K3) / (1 + s * T5)
        + (K2 * K3) / ((1 + s * T5) * (1 + s * T6))
    )
    return governor * steam / (1 + s * T4)


def transfer_ieeeg1(K, T1, T2, T3, T4, T5, T6, T7, K1, K3, K5, K7):
    """IEEEG1 tandem: K(1+sT2)/(1+sT1) * 1/(1+sT3) * chest * up to four stages.

    K2, K4, K6, and K8 are the LP-shaft fractions. They are zero here because
    the output is one mechanical power, not a cross-compound pair.
    A zero K7 drops the fourth stage. Leaving T7 in the chain would add a
    state that Pm does not observe.
    """
    governor = K * (1 + s * T2) / ((1 + s * T1) * (1 + s * T3))
    steam = (
        K1
        + K3 / (1 + s * T5)
        + K5 / ((1 + s * T5) * (1 + s * T6))
        + K7 / ((1 + s * T5) * (1 + s * T6) * (1 + s * T7))
    )
    return governor * steam / (1 + s * T4)


def sgo_fractions_to_g1(K2, K3):
    """Chest, reheater, and crossover fractions on an IEEEG1 card."""
    return {
        "K1": 1 - K2,
        "K3": K2 * (1 - K3),
        "K5": K2 * K3,
        "K7": 0,
    }


def g1_fractions_to_sgo(K1, K3, K5, K7):
    """Inverse fraction map. Exact only when the fourth stage is absent."""
    if sp.simplify(K7) != 0:
        return None
    tail = sp.simplify(K3 + K5)
    k3_sgo = 0 if tail == 0 else sp.simplify(K5 / tail)
    return {"K2": sp.simplify(1 - K1), "K3": k3_sgo}


def canceled(expr):
    return sp.cancel(sp.together(expr))


def minimal_order(expr):
    """Degree of the canceled denominator. A canceled zero-gain stage is not a state."""
    num, den = sp.fraction(sp.together(canceled(expr)))
    if sp.expand(num) == 0:
        return 0
    return int(sp.degree(sp.Poly(sp.expand(den), s)))


def controllable_canonical(expr):
    """Strictly proper controllable canonical form of a canceled SISO transfer function.

    G(s) = (b1 s^(n-1) + ... + bn) / (s^n + a1 s^(n-1) + ... + an)
    """
    num_e, den_e = sp.fraction(sp.together(canceled(expr)))
    num = sp.Poly(sp.expand(num_e), s)
    den = sp.Poly(sp.expand(den_e), s)
    leading = den.LC()
    den = sp.Poly(sp.expand(den.as_expr() / leading), s)
    num = sp.Poly(sp.expand(num.as_expr() / leading), s)
    order = int(den.degree())
    if int(num.degree()) >= order:
        raise ValueError("expected a strictly proper transfer function")

    a_coeffs = den.all_coeffs()[1:]
    high_to_low = [sp.Integer(0)] * order
    if num.as_expr() != 0:
        coeffs = num.all_coeffs()
        degree = int(num.degree())
        for index, coeff in enumerate(coeffs):
            power = degree - index
            high_to_low[order - 1 - power] = sp.simplify(coeff)

    state = sp.zeros(order)
    for row in range(order - 1):
        state[row, row + 1] = 1
    for column, coeff in enumerate(reversed(a_coeffs)):
        state[order - 1, column] = -sp.simplify(coeff)

    input_vec = sp.zeros(order, 1)
    input_vec[order - 1, 0] = 1
    output_row = sp.Matrix([[c for c in reversed(high_to_low)]])
    direct = sp.zeros(1)
    return state, input_vec, output_row, direct


def ss_response(state, input_vec, output_row, direct, s_value):
    """C(sI-A)^(-1)B + D at one complex frequency, as a Python complex."""
    amat = np.array(state, dtype=complex)
    bvec = np.array(input_vec, dtype=complex)
    crow = np.array(output_row, dtype=complex)
    dterm = complex(direct[0])
    order = amat.shape[0]
    resolvent = s_value * np.eye(order) - amat
    return complex((crow @ np.linalg.solve(resolvent, bvec) + dterm)[0, 0])


def realization_residual(expr, realization, samples):
    state, input_vec, output_row, direct = realization
    response = sp.lambdify(s, expr, "numpy")
    errors = []
    for sample in samples:
        from_ss = ss_response(state, input_vec, output_row, direct, sample)
        from_tf = complex(response(sample))
        errors.append(abs(from_ss - from_tf))
    return max(errors)


def check_canonical_form():
    sample = (s + 2) / (s**2 + 4 * s + 3)
    realization = controllable_canonical(sample)
    residual = realization_residual(sample, realization, (1 + 1j, 2j, -4))
    if residual > 1e-9:
        raise RuntimeError(f"canonical form self-check residual {residual}")


def frequency_response(expr, omega):
    response = sp.lambdify(s, expr, "numpy")
    values = np.asarray(response(1j * omega), dtype=complex)
    if values.ndim == 0:
        values = np.full(omega.shape, values)
    return values


def bode_curves(values):
    magnitude_db = 20 * np.log10(np.abs(values))
    phase_deg = np.degrees(np.unwrap(np.angle(values)))
    return magnitude_db, phase_deg


def max_pair_error(reference, other):
    """Largest gain and phase mismatch. Phase uses the complex ratio, not unwrap."""
    ratio = other / reference
    mag_db = np.max(np.abs(20 * np.log10(np.abs(ratio))))
    phase_deg = np.max(np.abs(np.degrees(np.angle(ratio))))
    return float(mag_db), float(phase_deg)


def build_cases():
    """Symbolic identity first, then the illustrative numeric cards."""
    k1, t1, t2, t3, t4, t5, t6, k2, k3 = sp.symbols(
        "K1 T1 T2 T3 T4 T5 T6 K2 K3", positive=True
    )
    mapped = sgo_fractions_to_g1(k2, k3)
    general_sgo = transfer_ieesgo(k1, t1, t2, t3, t4, t5, t6, k2, k3)
    general_g1 = transfer_ieeeg1(
        k1, t1, t2, t3, t4, t5, t6, 1, mapped["K1"], mapped["K3"], mapped["K5"], 0
    )
    general_swap = transfer_ieeeg1(
        k1, t3, t2, t1, t4, t5, t6, 1, mapped["K1"], mapped["K3"], mapped["K5"], 0
    )
    difference_mapped = sp.simplify(canceled(general_g1 - general_sgo))
    difference_swap = sp.simplify(canceled(general_swap - general_g1))
    reverse = g1_fractions_to_sgo(mapped["K1"], mapped["K3"], mapped["K5"], 0)
    reverse_k2 = sp.simplify(reverse["K2"] - k2)
    reverse_k3 = sp.simplify(reverse["K3"] - k3)

    sgo = transfer_ieesgo(
        K_DROOP, T_SLOW, T_LEAD, T_FAST, T_CHEST, T_REHEAT, T_CROSS,
        FRAC_IP + FRAC_LP, FRAC_LP / (FRAC_IP + FRAC_LP),
    )
    reduced = transfer_ieeeg1(
        K_DROOP, T_SLOW, T_LEAD, T_FAST, T_CHEST, T_REHEAT, T_CROSS, 1,
        FRAC_HP, FRAC_IP, FRAC_LP, 0,
    )
    swapped = transfer_ieeeg1(
        K_DROOP, T_FAST, T_LEAD, T_SLOW, T_CHEST, T_REHEAT, T_CROSS, 1,
        FRAC_HP, FRAC_IP, FRAC_LP, 0,
    )
    scale = 1 - K7_FOUR
    four_stage = transfer_ieeeg1(
        K_DROOP, T_SLOW, T_LEAD, T_FAST, T_CHEST, T_REHEAT, T_CROSS, T7_FOUR,
        FRAC_HP * scale, FRAC_IP * scale, FRAC_LP * scale, K7_FOUR,
    )
    return {
        "difference_mapped": difference_mapped,
        "difference_swap": difference_swap,
        "reverse_k2": reverse_k2,
        "reverse_k3": reverse_k3,
        "mapped_symbols": mapped,
        "sgo": sgo,
        "reduced": reduced,
        "swapped": swapped,
        "four_stage": four_stage,
    }


def print_identity(cases):
    print("Fraction map, IEESGO to tandem IEEEG1 with the fourth stage off")
    print("  IEEEG1.K  = IEESGO.K1")
    print("  IEEEG1.K1 = 1 - IEESGO.K2")
    print("  IEEEG1.K3 = IEESGO.K2 * (1 - IEESGO.K3)")
    print("  IEEEG1.K5 = IEESGO.K2 * IEESGO.K3")
    print("  IEEEG1.K7 = 0")
    print("  T4, T5, T6 copy across")
    print("  {T1, T3} is an unordered pair: either assignment has the same transfer function")
    print()
    print("Reverse map residual (0 means the fractions come back)")
    print(f"  K2: {cases['reverse_k2']}")
    print(f"  K3: {cases['reverse_k3']}")
    print()
    print("Simplified transfer-function differences")
    print(f"  mapped IEEEG1 - IEESGO = {cases['difference_mapped']}")
    print(f"  T1/T3 swapped - named IEEEG1 = {cases['difference_swap']}")
    four_gap = canceled(cases["four_stage"] - cases["sgo"])
    print(f"  four-stage difference at s=0: {sp.simplify(four_gap.subs(s, 0))}")
    print(f"  four-stage difference at s=1: {sp.simplify(four_gap.subs(s, 1))}")
    print()
    if cases["difference_mapped"] != 0 or cases["difference_swap"] != 0:
        raise RuntimeError("expected exact cancellation for the mapped and swapped cards")
    if cases["reverse_k2"] != 0 or cases["reverse_k3"] != 0:
        raise RuntimeError("reverse fraction map did not recover IEESGO")
    if sp.simplify(four_gap.subs(s, 1)) == 0:
        raise RuntimeError("four-stage card should not match IEESGO")


def print_state_space(cases):
    """Block count is the chain of lags. Minimal order drops a canceled pair.

    T2 and T6 are both 1/2 s in the illustrative card, so the governor lead
    cancels the crossover pole. The block diagram still has that lag. The
    state-space kept here is the minimal one, after cancellation.
    """
    block_counts = {
        "IEESGO": 5,
        "IEEEG1 reduced": 5,
        "IEEEG1 T1/T3 swapped": 5,
        "IEEEG1 four-stage": 6,
    }
    realizations = {
        "IEESGO": controllable_canonical(cases["sgo"]),
        "IEEEG1 reduced": controllable_canonical(cases["reduced"]),
        "IEEEG1 T1/T3 swapped": controllable_canonical(cases["swapped"]),
        "IEEEG1 four-stage": controllable_canonical(cases["four_stage"]),
    }
    samples = (1 + 0.3j, 2j, 10j)
    print("Order")
    print("  Uncanceled lags: IEESGO 5, reduced IEEEG1 5, four-stage IEEEG1 6")
    print("  T2 = T6 = 1/2 s, so the lead cancels the crossover and the minimal order is one less")
    for name, expr in (
        ("IEESGO", cases["sgo"]),
        ("IEEEG1 reduced", cases["reduced"]),
        ("IEEEG1 T1/T3 swapped", cases["swapped"]),
        ("IEEEG1 four-stage", cases["four_stage"]),
    ):
        order = minimal_order(expr)
        residual = realization_residual(expr, realizations[name], samples)
        print(f"  {name}: minimal order {order}, state-space residual {residual:.3e}")
        if order != block_counts[name] - 1:
            raise RuntimeError(f"{name} should lose exactly the T2/T6 pair")
        if residual > 1e-6:
            raise RuntimeError(f"state-space does not match {name}")

    for part in range(3):
        gap = sp.simplify(realizations["IEESGO"][part] - realizations["IEEEG1 reduced"][part])
        if gap != sp.zeros(*realizations["IEESGO"][part].shape):
            raise RuntimeError("equivalent cards should share one controllable realization")

    print()
    print("Controllable canonical A for IEESGO (reduced and swapped cards match it)")
    print(np.array2string(np.array(realizations["IEESGO"][0], dtype=float), precision=4, suppress_small=True))
    print("C")
    print(np.array2string(np.array(realizations["IEESGO"][2], dtype=float), precision=4, suppress_small=True))
    print()
    print("Controllable canonical A for four-stage IEEEG1")
    print(np.array2string(np.array(realizations["IEEEG1 four-stage"][0], dtype=float), precision=4, suppress_small=True))
    print("C")
    print(np.array2string(np.array(realizations["IEEEG1 four-stage"][2], dtype=float), precision=4, suppress_small=True))
    print()


def plot_bode(cases, destination):
    omega = np.logspace(np.log10(W_MIN), np.log10(W_MAX), N_FREQ)
    responses = {
        name: frequency_response(cases[key], omega)
        for name, key in (
            ("IEESGO", "sgo"),
            ("reduced", "reduced"),
            ("swapped", "swapped"),
            ("four", "four_stage"),
        )
    }
    curves = {name: bode_curves(values) for name, values in responses.items()}

    match_reduced = max_pair_error(responses["IEESGO"], responses["reduced"])
    match_swapped = max_pair_error(responses["IEESGO"], responses["swapped"])
    mismatch = max_pair_error(responses["IEESGO"], responses["four"])
    error_db = curves["four"][0] - curves["IEESGO"][0]
    worst = int(np.argmax(np.abs(error_db)))

    print("Bode check, Pm / (omega_ref - omega)")
    dc_gain = frequency_response(cases["sgo"], np.array([0.0]))[0].real
    print(f"  DC gain IEESGO: {dc_gain:.6g}")
    print(f"  max |dB| and |deg| , reduced vs IEESGO: {match_reduced[0]:.3e}, {match_reduced[1]:.3e}")
    print(f"  max |dB| and |deg| , swapped vs IEESGO: {match_swapped[0]:.3e}, {match_swapped[1]:.3e}")
    print(f"  max |dB| and |deg| , four-stage vs IEESGO: {mismatch[0]:.4g}, {mismatch[1]:.4g}")
    print(f"  four-stage magnitude error peaks at {error_db[worst]:+.4g} dB, omega = {omega[worst]:.4g} rad/s")
    print()

    if match_reduced[0] > 1e-6 or match_swapped[0] > 1e-6:
        raise RuntimeError("equivalent Bode curves should overlie")

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 10.2), sharex=True)
    magnitude = axes[0]
    phase = axes[1]
    error = axes[2]

    magnitude.semilogx(omega, curves["IEESGO"][0], color="black", lw=2.2, label="IEESGO", zorder=2)
    phase.semilogx(omega, curves["IEESGO"][1], color="black", lw=2.2, label="IEESGO", zorder=2)
    marker_sets = (
        (slice(0, None, 24), "o", "C0", "IEEEG1 reduced"),
        (slice(12, None, 24), "x", "C1", "IEEEG1, T1 and T3 swapped"),
    )
    for index, (picker, marker, color, label) in enumerate(marker_sets):
        magnitude.semilogx(
            omega[picker], curves["reduced" if index == 0 else "swapped"][0][picker],
            linestyle="none", marker=marker, color=color, ms=5, label=label, zorder=3,
        )
        phase.semilogx(
            omega[picker], curves["reduced" if index == 0 else "swapped"][1][picker],
            linestyle="none", marker=marker, color=color, ms=5, label=label, zorder=3,
        )

    magnitude.semilogx(omega, curves["four"][0], color="C3", lw=1.6, label="IEEEG1 four-stage", zorder=1)
    phase.semilogx(omega, curves["four"][1], color="C3", lw=1.6, label="IEEEG1 four-stage", zorder=1)
    error.semilogx(omega, error_db, color="C3", lw=1.6, label="four-stage minus IEESGO")
    error.axhline(0, color="black", lw=0.8, ls="--")

    magnitude.set_ylabel("magnitude (dB)")
    phase.set_ylabel("phase (deg)")
    error.set_ylabel("magnitude error (dB)")
    error.set_xlabel(r"frequency $\omega$ (rad/s)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.35)
        axis.legend(loc="best", fontsize=8)
    magnitude.set_title(r"Open-loop $P_m/(\omega_{ref}-\omega)$. Markers on the black curve are the same linear model.")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"Wrote {destination}")

    return {
        "match_reduced_db": match_reduced[0],
        "match_reduced_deg": match_reduced[1],
        "match_swapped_db": match_swapped[0],
        "match_swapped_deg": match_swapped[1],
        "mismatch_db": mismatch[0],
        "mismatch_deg": mismatch[1],
        "error_peak_db": float(error_db[worst]),
        "error_peak_omega": float(omega[worst]),
    }


def main():
    check_canonical_form()
    cases = build_cases()
    print_identity(cases)
    print_state_space(cases)
    figure = Path(__file__).with_name("ieeeg1_ieesgo_bode.png")
    plot_bode(cases, figure)


if __name__ == "__main__":
    main()
