"""
FIE2026 공식 9구간 梯度(gradient) 채점기 + 置信도→구간 매핑.

9개 叙实강도 구간 (정렬 순):
  0 강反 (FALSE, conf in (.875,1])
  1 较强反(FALSE, (.75,.875])
  2 较弱反(FALSE, (.625,.75])
  3 弱反 (FALSE, (.5,.625])
  4 非   (UNCERTAIN, 0.5)
  5 弱正 (TRUE, (.5,.625])
  6 较弱正(TRUE, (.625,.75])
  7 较强正(TRUE, (.75,.875])
  8 강正 (TRUE, (.875,1])

채점: 같은 구간 1점 / 인접 σ=0.6827 / 비인접 0점.
"""
from __future__ import annotations
import re

SIGMA = 0.6827
INTERVAL_NAMES = ["강反", "较强反", "较弱反", "弱反", "非", "弱正", "较弱正", "较强正", "강正"]

# 경계: 좌개우폐 (lo, hi]  — README 규정
_BANDS = [(0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0)]  # idx 0..3 within a polarity


def _conf_to_quartile(conf: float) -> int:
    """confidence 값을 0..3 사분위 인덱스로. (.5,.625]->0 ... (.875,1]->3"""
    c = float(conf)
    if c <= 0.5:
        return 0
    if c <= 0.625:
        return 0
    if c <= 0.75:
        return 1
    if c <= 0.875:
        return 2
    return 3


def label_conf_to_idx(label: str, conf) -> int:
    """(factivity, confidence) -> 0..8 구간 인덱스. confidence는 float 또는 밴드문자열."""
    label = (label or "").strip().upper()
    if label in ("U", "UNCERTAIN", "NEI", "NEUTRAL"):
        return 4
    # 밴드 문자열이면 파싱
    q = band_string_to_quartile(conf) if isinstance(conf, str) and "(" in conf else _conf_to_quartile(conf)
    if label in ("F", "FALSE", "CONTRADICTION"):
        # 강反(0)이 가장 높은 conf(q=3) ... 弱反(3)이 가장 낮은 conf(q=0)
        return 3 - q
    if label in ("T", "TRUE", "ENTAILMENT"):
        return 5 + q
    raise ValueError(f"unknown label: {label!r}")


def band_string_to_quartile(s: str) -> int:
    """'(0.875, 1]' -> 3, '(0.5, 0.625]' -> 0, '0.5' -> 0"""
    s = s.strip()
    if s in ("0.5", "0.50"):
        return 0
    m = re.findall(r"[0-9.]+", s)
    if not m:
        raise ValueError(f"bad band: {s!r}")
    hi = float(m[-1])
    # hi 기준 매핑
    if hi <= 0.625:
        return 0
    if hi <= 0.75:
        return 1
    if hi <= 0.875:
        return 2
    return 3


def gold_to_idx(item: dict) -> int:
    """gold 항목 -> 구간 인덱스. confidence가 float(0401) 또는 밴드문자열(0502) 모두 지원."""
    return label_conf_to_idx(item["factivity"], item["confidence"])


def score_pair(pred_idx: int, gold_idx: int) -> float:
    d = abs(pred_idx - gold_idx)
    if d == 0:
        return 1.0
    if d == 1:
        return SIGMA
    return 0.0


def s_to_label_conf(S: float, eps: float = 0.06):
    """연속 veridicality S∈[-1,1] -> (label, confidence). |S|<=eps면 U."""
    if abs(S) <= eps:
        return "UNCERTAIN", 0.5
    label = "TRUE" if S > 0 else "FALSE"
    conf = min(max(abs(S), 0.5001), 1.0)
    return label, round(conf, 2)


def evaluate(preds, golds):
    """preds: [(label, conf)] 또는 [idx]; golds: gold dict 리스트.
    반환: dict(총점, 정규화점수, 라벨정확도, 구간정확도, n)"""
    total = 0.0
    label_ok = 0
    band_ok = 0
    n = len(golds)
    for p, g in zip(preds, golds):
        gi = gold_to_idx(g)
        pi = p if isinstance(p, int) else label_conf_to_idx(p[0], p[1])
        s = score_pair(pi, gi)
        total += s
        if s == 1.0:
            band_ok += 1
        # 라벨 정확도(밴드 무시)
        gl = g["factivity"].strip().upper()
        pl = ("UNCERTAIN" if pi == 4 else "FALSE" if pi < 4 else "TRUE")
        if pl == gl or (gl in ("U",) and pl == "UNCERTAIN"):
            label_ok += 1
    return {
        "n": n,
        "total": round(total, 3),
        "norm": round(total / n, 4),          # 0~1 (최대 1)
        "label_acc": round(label_ok / n, 4),
        "band_acc": round(band_ok / n, 4),
    }
