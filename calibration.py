# -*- coding: utf-8 -*-
"""置信도 보정: 라벨/사분위 → 보정 사분위 매핑을 학습(샘플집), 적용."""
from collections import Counter, defaultdict
import scorer as SC

# 사분위 대표 confidence (구간 중앙, 경계 회피)
QUARTILE_CONF = {0: 0.56, 1: 0.69, 2: 0.81, 3: 0.95}


def gold_quartile(item):
    idx = SC.gold_to_idx(item)
    if idx == 4:
        return None  # U
    return idx if idx < 4 else idx - 5  # FALSE: 0..3(강반..약반) / TRUE: 0..3(약정..강정) — 내부 사분위로 환산
    # 주의: FALSE idx0(강반)=q3, 그러므로 아래서 보정은 라벨별로 따로 학습


def fit_label_modal(train):
    """라벨별 gold 밴드의 최빈 사분위(0..3 by confidence height) 학습."""
    by = defaultdict(Counter)
    for it in train:
        lab = it["factivity"].strip().upper()
        if lab == "UNCERTAIN":
            continue
        q = SC.band_string_to_quartile(it["confidence"]) if isinstance(it["confidence"], str) and "(" in str(it["confidence"]) else SC._conf_to_quartile(it["confidence"])
        by[lab][q] += 1
    modal = {}
    for lab, c in by.items():
        modal[lab] = c.most_common(1)[0][0]
    return modal  # {'TRUE':3, 'FALSE':3, ...} (사분위 by conf height)


def apply_label_modal(pred_labels, modal):
    """예측 라벨 리스트 -> (label, conf) 보정. U는 0.5."""
    out = []
    for lab in pred_labels:
        lab = lab.strip().upper()
        if lab == "UNCERTAIN":
            out.append(("UNCERTAIN", 0.5))
        else:
            q = modal.get(lab, 3)
            out.append((lab, QUARTILE_CONF[q]))
    return out


def fit_isotonic_bins(train_preds, train_golds):
    """예측(label, conf) → gold 밴드 재매핑 테이블 학습: (label, pred_quartile)->gold 최빈 사분위."""
    table = defaultdict(Counter)
    for (pl, pc), g in zip(train_preds, train_golds):
        pl = pl.strip().upper()
        if pl == "UNCERTAIN":
            continue
        pq = SC._conf_to_quartile(pc)
        gl = g["factivity"].strip().upper()
        if gl != pl:
            continue  # 라벨 일치 항목으로만 conf 보정 학습
        gq = SC.band_string_to_quartile(g["confidence"]) if isinstance(g["confidence"], str) and "(" in str(g["confidence"]) else SC._conf_to_quartile(g["confidence"])
        table[(pl, pq)][gq] += 1
    mapping = {k: c.most_common(1)[0][0] for k, c in table.items()}
    return mapping


def apply_isotonic(preds, mapping):
    out = []
    for pl, pc in preds:
        pl = pl.strip().upper()
        if pl == "UNCERTAIN":
            out.append(("UNCERTAIN", 0.5)); continue
        pq = SC._conf_to_quartile(pc)
        gq = mapping.get((pl, pq), pq)
        out.append((pl, QUARTILE_CONF[gq]))
    return out
