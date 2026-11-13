# IEEEG1 and IEESGO: follow-up

Linear check of when the steam turbine-governor models IEEEG1 and IEESGO are the same transfer function. The conversion is an algebraic map. The Bode plot only confirms it.

Input is speed error \(\omega_{ref}-\omega\). Output is mechanical power \(P_m\). Valve rate limits, deadband, and power limits are left out.

## Run

```text
python ieeeg1_ieesgo_bode.py
```

The script prints the canceled differences, the minimal state-space, and the Bode errors. It writes `ieeeg1_ieesgo_bode.png` in this folder.

SymPy builds and cancels the transfer functions and builds the controllable canonical state-space. The frequency response is `G(jω)` through `lambdify`. Needs SymPy, NumPy, and Matplotlib.

## Same symbols, different blocks

Do not copy a parameter card from one model onto the other.

| Symbol | IEESGO | IEEEG1 |
| --- | --- | --- |
| Gain | \(K_1 = 1/R\), inside the controller lag | \(K = 1/R\), on the lead-lag |
| \(T_1\) | Controller lag | Lead-lag lag |
| \(T_2\) | Lead | Lead |
| \(T_3\) | Lead-lag lag | Valve lag |
| \(K_1\) | Droop gain \(1/R\) | Chest / HP fraction |
| \(K_2, K_3\) | Nested steam fractions | LP-shaft fractions |

The governor products are the same rational function:

\[
K\frac{1+sT_2}{(1+sT_1)(1+sT_3)}
\]

Swapping \(T_1\) and \(T_3\) does not change it. A Bode plot cannot say which pole is the controller and which is the valve.

## Exact map, tandem compound

This is exact only when IEEEG1 has one shaft and the fourth steam stage is off: \(K_7=0\), and \(K_2=K_4=K_6=K_8=0\), with \(K_1+K_3+K_5=1\).

IEESGO to IEEEG1:

- \(K = K_1^{\mathrm{SGO}}\)
- \(K_1 = 1-K_2^{\mathrm{SGO}}\) (power before the reheater)
- \(K_3 = K_2^{\mathrm{SGO}}(1-K_3^{\mathrm{SGO}})\) (after the reheater, before the crossover)
- \(K_5 = K_2^{\mathrm{SGO}} K_3^{\mathrm{SGO}}\) (after the crossover)
- \(K_7 = 0\)
- \(T_4, T_5, T_6\) copy across
- \(\{T_1, T_3\}\) may be assigned either way

IEEEG1 back to IEESGO, when \(K_7=0\) and \(K_3+K_5 \neq 0\):

- \(K_2 = 1-K_1\)
- \(K_3 = K_5/(K_3+K_5)\)

SymPy reduces both `(mapped IEEEG1 − IEESGO)` and `(T1/T3 swapped − named IEEEG1)` to 0. The reverse fraction residuals are 0.

Steam paths, written so the products match:

\[
\frac{1}{1+sT_4}\left[(1-K_2)+\frac{K_2(1-K_3)}{1+sT_5}+\frac{K_2 K_3}{(1+sT_5)(1+sT_6)}\right]
\]

\[
\frac{1}{1+sT_4}\left[K_1+\frac{K_3}{1+sT_5}+\frac{K_5}{(1+sT_5)(1+sT_6)}+\frac{K_7}{(1+sT_5)(1+sT_6)(1+sT_7)}\right]
\]

## Illustrative card

Not a plant data card. Five percent droop on a common power base, so the DC gain is 20.

- Lead \(T_2 = 0.5\,\mathrm{s}\), poles at \(2.0\,\mathrm{s}\) and \(0.1\,\mathrm{s}\)
- Chest \(0.3\,\mathrm{s}\), reheater \(8\,\mathrm{s}\), crossover \(0.5\,\mathrm{s}\)
- Power split 0.3 / 0.4 / 0.3, so IEESGO uses \(K_2 = 0.7\), \(K_3 = 3/7\)

The four-stage counterexample keeps the same governor and moves 0.15 of the power behind \(T_7 = 1\,\mathrm{s}\). The other fractions are scaled by 0.85 and still sum to 1: \(K_1 = 0.255\), \(K_3 = 0.34\), \(K_5 = 0.255\), \(K_7 = 0.15\).

## What this run showed

- Equivalent Bode error, reduced and swapped cards versus IEESGO: about \(10^{-15}\,\mathrm{dB}\) and \(10^{-15}\,\mathrm{deg}\). The markers in the figure sit on the IEESGO curve. Three different cards, one linear model.
- Four-stage card versus IEESGO: DC values still match (difference 0 at \(s=0\)). At \(s=1\) the transfer functions differ by \(-445/1287\).
- Largest magnitude error is \(-1.714\,\mathrm{dB}\) at \(0.933\,\mathrm{rad/s}\). Largest phase error is about \(6.5\,\mathrm{deg}\).
- Above a few rad/s the magnitude error settles near \(20\log_{10}(0.85) \approx -1.41\,\mathrm{dB}\). The chest fraction fell from 0.3 to 0.255, and that fraction sets the high-frequency gain.
- Uncanceled lag count: 5 for IEESGO and the reduced IEEEG1, 6 for the four-stage model.
- Minimal order is one less, 4 and 5. On this card \(T_2 = T_6 = 0.5\,\mathrm{s}\), so the governor lead cancels the crossover pole. The state-space in the script is that minimal realization. Its frequency response matches \(G(j\omega)\) to about \(10^{-15}\). The reduced card and the swapped-pole card share the IEESGO matrices.

## What a Bode match does not give you

- A unique IEEEG1 card from an IEESGO card. \(T_7\), \(K_7\), the LP-shaft fractions, \(U_o\), \(U_c\), deadband, and the \(T_1\) versus \(T_3\) assignment stay free.
- An exact IEESGO card from a four-stage IEEEG1. The extra pole changes the frequency response. A fit over a band would be an approximation, and it would not be unique.
- Valve rate limits. Those are the main large-signal difference and they never enter this linear curve. Identical Bode plots can still diverge on a large frequency step.
- An HP versus LP split from total \(P_m\) alone.

## Next, not done here

1. Pick a frequency band and reduce the four-stage model to IEESGO. Report the dB and degree error of that approximation, and show that more than one reduced card can land inside the same error.
2. Time-domain step with IEEEG1 rate limits \(U_o\) and \(U_c\). Use two cards that share this linear Bode plot and show that \(P_m(t)\) separates once the valve rate saturates.
3. Close the loop with \(1/(2Hs)\) only after an inertia \(H\) is chosen. The same linear \(P_m\) / speed-error transfer function gives the same speed response. The rate limits still separate the time response.
4. If a real data card is used, drop any zero time constant before building a state. A zero \(T\) is an algebraic block. Also treat a zero steam fraction as unobservable: do not keep its lag as a state.

## Files

- `ieeeg1_ieesgo_bode.py` — identities, state-space, Bode figure
- `ieeeg1_ieesgo_bode.png` — magnitude, phase, and the four-stage magnitude error
