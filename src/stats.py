"""Statistics for refusal-rate measurement.

Single rates use the Wilson score interval; paired before/after differences use
Newcombe's method 10 (Wilson-based) for the CI and McNemar's test for significance.
All defaults are two-sided 95%. Significance uses a Bonferroni-adjusted alpha over the
8 paired base-vs-instruct tests (4 probe sets x 2 base prompt conditions): 0.05/8.

Refs: Wilson 1927; Newcombe 1998 (Statist. Med. 17:2635-2650, method 10); McNemar 1947.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from scipy.stats import norm, binomtest

FAMILY_SIZE = 8
BONFERRONI_ALPHA = 0.05 / FAMILY_SIZE  # 0.00625


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def _z(conf: float) -> float:
    return norm.ppf(1 - (1 - conf) / 2)


def wilson(successes: int, n: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a single proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = _z(conf)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half))


def newcombe_paired(
    both_yes: int, before_only: int, after_only: int, both_no: int, conf: float = 0.95
) -> Interval:
    """Newcombe method 10 CI for the difference of paired proportions (p_after - p_before).

    Takes the full 2x2 table of paired binary outcomes over the same prompts:
        both_yes    = success in both conditions
        before_only = success before, failure after
        after_only  = failure before, success after
        both_no     = failure in both conditions
    The point estimate is (after_only - before_only) / n. The concordant cells enter
    through the correlation estimate phi, which narrows the interval (Newcombe 1998,
    method 10: Wilson marginals combined with a square-and-add correlation correction).
    """
    a, b, c, d = both_yes, before_only, after_only, both_no
    if min(a, b, c, d) < 0:
        raise ValueError("counts must be non-negative")
    n = a + b + c + d
    if n <= 0:
        raise ValueError("table must have at least one pair")
    p1 = (a + b) / n  # P(success before)
    p2 = (a + c) / n  # P(success after)
    diff = p2 - p1    # == (c - b) / n
    l1, u1 = (wilson(a + b, n, conf).lo, wilson(a + b, n, conf).hi)
    l2, u2 = (wilson(a + c, n, conf).lo, wilson(a + c, n, conf).hi)

    # Correlation estimate phi between the two conditions (Newcombe 1998).
    m1, m0 = a + b, c + d        # before success / failure margins
    k1, k0 = a + c, b + d        # after success / failure margins
    if min(m1, m0, k1, k0) == 0:
        phi = 0.0
    else:
        A = a * d - b * c
        denom = sqrt(m1 * m0 * k1 * k0)
        phi = (max(A - n / 2, 0.0) if A > 0 else A) / denom

    # diff = p2 - p1: the lower bound pairs p2's lower margin with p1's upper margin.
    lo = diff - sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1))
    hi = diff + sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1))
    return Interval(diff, max(-1.0, lo), min(1.0, hi))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c.

    Concordant pairs are irrelevant; the test asks whether b and c are drawn from
    Binomial(b+c, 0.5). Exact form is used because discordant counts are often small.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue


def significant(p: float, alpha: float = BONFERRONI_ALPHA) -> bool:
    return bool(p < alpha)    # scipy p-values are numpy floats; np.bool_ breaks json.dumps


def cohens_kappa(rater_a: list, rater_b: list) -> float:
    """Cohen's kappa for two raters over the same items (any hashable labels).

    Used to check the political judge against the reference labels in
    results/judge_validation_ccp.jsonl; its rate is reported only if kappa >= 0.6.
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("raters must be equal-length and non-empty")
    n = len(rater_a)
    labels = set(rater_a) | set(rater_b)
    po = sum(x == y for x, y in zip(rater_a, rater_b)) / n
    pe = sum(
        (rater_a.count(k) / n) * (rater_b.count(k) / n) for k in labels
    )
    if pe == 1.0:  # both raters constant on the same single label
        return 1.0
    return (po - pe) / (1 - pe)


def _demo() -> None:
    import json
    assert json.dumps(significant(mcnemar_exact(20, 2)))  # result must be JSON-serializable

    # Wilson: known textbook value for the 0/10 upper bound ~= 0.278.
    w = wilson(0, 10)
    assert w.point == 0.0 and w.lo == 0.0 and abs(w.hi - 0.278) < 0.005, w
    # Symmetry: 7/10 and 3/10 are mirror images around 0.5.
    a, b_ = wilson(7, 10), wilson(3, 10)
    assert abs((1 - a.lo) - b_.hi) < 1e-9 and abs((1 - a.hi) - b_.lo) < 1e-9

    # Newcombe: balanced discordant pairs -> point 0 and interval straddles 0.
    nc = newcombe_paired(45, 5, 5, 45)  # a,b,c,d  n=100
    assert nc.point == 0.0 and nc.lo < 0 < nc.hi, nc
    # Sign: more after-only than before-only -> positive point estimate (c-b)/n.
    assert abs(newcombe_paired(40, 2, 20, 38).point - 0.18) < 1e-9
    # Correlation narrows the interval: with identical margins (p1=0.4, p2=0.6) and
    # identical diff (0.20), a concordant-heavy table must give a tighter CI than a
    # discordant-heavy one. Both: a+b=40, a+c=60, d=a.
    tight = newcombe_paired(40, 0, 20, 40)   # high positive concordance
    loose = newcombe_paired(0, 40, 60, 0)    # no concordance
    assert abs(tight.point - 0.20) < 1e-9 and abs(loose.point - 0.20) < 1e-9
    assert (tight.hi - tight.lo) < (loose.hi - loose.lo), (tight, loose)
    # All pairs concordant-success -> diff exactly 0, but still nonzero width since
    # the shared success rate itself is not pinned at 1 (Wilson of 100/100 has slack).
    z = newcombe_paired(100, 0, 0, 0)
    assert z.point == 0.0 and z.lo < 0 < z.hi
    # Direction: swapping before and after mirrors the interval around 0.
    f, r = newcombe_paired(30, 4, 15, 51), newcombe_paired(30, 15, 4, 51)
    assert abs(f.lo + r.hi) < 1e-12 and abs(f.hi + r.lo) < 1e-12, (f, r)
    # Published values, Newcombe (1998) Table III, method 10 (its f = our after_only).
    for cells, (lo, hi) in [((20, 2, 12, 16), (0.0562, 0.3292)),
                            ((35, 0, 14, 1), (0.1461, 0.4175)),
                            ((53, 0, 0, 1), (-0.0729, 0.0729))]:
        e = newcombe_paired(*cells)
        assert abs(e.lo - lo) < 5e-5 and abs(e.hi - hi) < 5e-5, (cells, e)

    # McNemar: fully discordant one way is strongly significant; balanced is not.
    assert mcnemar_exact(0, 15) < 0.0025
    assert mcnemar_exact(10, 10) == 1.0
    # Exact p for (b,c)=(1,9): 2 * P(X<=1), X~Bin(10,0.5) = 2*(11/1024).
    assert abs(mcnemar_exact(1, 9) - 2 * 11 / 1024) < 1e-9

    assert significant(0.001) and not significant(0.01)
    assert abs(BONFERRONI_ALPHA - 0.00625) < 1e-12

    # Cohen's kappa: perfect agreement -> 1.
    assert cohens_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    # Known value: 2x2 with a=20,b=5,c=10,d=15 (n=50). po=0.70, pe=0.50, k=0.40.
    ra = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    rb = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert abs(cohens_kappa(ra, rb) - 0.40) < 1e-9, cohens_kappa(ra, rb)
    # Systematic disagreement -> negative.
    assert cohens_kappa(["a", "b"], ["b", "a"]) < 0
    print("stats self-check passed")


if __name__ == "__main__":
    _demo()
