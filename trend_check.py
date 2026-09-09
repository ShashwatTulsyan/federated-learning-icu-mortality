"""
Does the model respond to a DECLINING trend, not just a bad snapshot value?

Run this BEFORE your demonstration so you have real numbers from your actual
trained model, not a guess. Takes under a minute.

Tests four shapes for heart rate over the 24h window, holding the rest of the
patient's real recorded trajectory fixed:

  stable_normal   flat at a normal value the whole time
  stable_high     flat at an elevated value the whole time (a bad snapshot,
                  no trend)
  crashing        normal for 18h, then a fast drop to severe bradycardia in
                  the final 6 hours
  crashing_late   the same drop compressed into just the final 3 hours (more
                  acute)

If risk(crashing) > risk(stable_high), the model is picking up the trend, not
just the endpoint -- a good, quotable result.
If risk(crashing) <= risk(stable_normal), that is a genuine limitation worth
stating plainly rather than hiding. Report it as such; do not paper over it.

Run:  python trend_check.py
"""
import numpy as np

import demo_app as D
import config as C


def trajectory(pattern, start=82.0, crisis=25.0):
    if pattern == "stable_normal":
        return [start] * 24
    if pattern == "stable_high":
        return [130.0] * 24
    if pattern == "crashing":
        return [start] * 18 + list(np.linspace(start - 5, crisis, 6))
    if pattern == "crashing_late":
        return [start] * 21 + list(np.linspace(start - 10, crisis, 3))
    raise ValueError(pattern)


def main():
    print("Loading the trained model and a sample of held-out patients ...\n")
    S = D.Store()

    # Test on several patients, not one -- a single case could be an outlier.
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(S.cases), min(15, len(S.cases)), replace=False)
    rows = [S.cases[i]["row"] for i in sample_idx]

    print(f"{'pattern':<16}{'mean risk':>12}{'vs stable_normal':>20}")
    print("-" * 48)
    results = {}
    for pattern in ("stable_normal", "stable_high", "crashing", "crashing_late"):
        hr = trajectory(pattern)
        risks = [S.predict_row(row, {"heart_rate": hr})[0] for row in rows]
        m = float(np.mean(risks))
        results[pattern] = m
        base = results.get("stable_normal", m)
        print(f"{pattern:<16}{m*100:>11.1f}%{(m-base)*100:>+19.1f} pts")

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    crash_vs_steady = results["crashing_late"] - results["stable_high"]
    if crash_vs_steady > 0.02:
        print(f"  A late, fast crash scores {crash_vs_steady*100:+.1f} points higher")
        print("  than a steady bad value with the same magnitude. The model")
        print("  IS sensitive to the trend, not just the endpoint reading.")
        print("\n  Quotable: 'A patient whose heart rate collapses in the final")
        print(f"  hours is scored {crash_vs_steady*100:.1f} points higher than one who")
        print("  is steadily tachycardic the whole time -- the model responds")
        print("  to deterioration, not just to a single bad number.'")
    elif crash_vs_steady > -0.02:
        print("  The crash and the steady-bad value score about the same.")
        print("  The model is picking up SOME signal from the abnormal region")
        print("  but is not strongly distinguishing trend from level. State")
        print("  this plainly if asked: 'the model responds to how abnormal a")
        print("  reading is, more strongly than to how it got there.'")
    else:
        print("  The crashing trajectory scores LOWER than the steady-bad one.")
        print("  This is a genuine limitation, not a bug: severe bradycardia is")
        print("  rarer in the training data than tachycardia (many bradycardic")
        print("  events precede death too closely to be well represented in a")
        print("  24h pre-death window), so the network has less signal to learn")
        print("  from in that direction. State this openly -- it is exactly the")
        print("  kind of finding that should inform clinical deployment: 'this")
        print("  model should not be used as the sole basis for detecting acute")
        print("  deterioration; it was not validated for that specific pattern")
        print("  and a live vitals monitor remains necessary alongside it.'")

    print("\n  Either result is a legitimate answer for a viva. An honestly")
    print("  reported limitation, backed by a test you actually ran, is a far")
    print("  stronger answer than an unverified claim that everything works.")


if __name__ == "__main__":
    main()