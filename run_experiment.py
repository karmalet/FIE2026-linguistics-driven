# -*- coding: utf-8 -*-
"""
FIE2026 실험 하니스 (오프라인 실행 가능 부분).
- 9구간 梯度 채점기 검증
- 置信도 보정 ablation (오라클 라벨) — 논문 ★C5 핵심 입증
- 기호 엔진 end-to-end (raw vs 보정)
- 교차셋 일반화(한 셋으로 보정 학습 → 다른 셋 평가)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import scorer as SC
import calibration as CAL
from backends import SymbolicBackend

DATA = os.path.join(os.path.dirname(__file__), "..", "sample_sets")
FILES = {"S1_0401(69)": "sample_20260401.json", "S2_0502(497)": "sample_20260502.json"}


def load(f):
    return json.load(open(os.path.join(DATA, f), encoding="utf-8"))


def oracle_labels(golds):
    return [g["factivity"].strip().upper() for g in golds]


def fmt(d):
    return f"norm={d['norm']:.4f}  label_acc={d['label_acc']:.3f}  band_acc={d['band_acc']:.3f}  (Σ={d['total']}/{d['n']})"


def run():
    sets = {k: load(v) for k, v in FILES.items()}

    print("=" * 78)
    print("E1. 채점기 검증 (gold→gold, norm=1.0 이어야 함)")
    print("=" * 78)
    for name, g in sets.items():
        preds = [(x["factivity"], x["confidence"]) for x in g]
        print(f"  {name:14s} {fmt(SC.evaluate(preds, g))}")

    print("\n" + "=" * 78)
    print("E2. 置信도 보정 ablation — 오라클 라벨 고정, confidence 정책만 변경")
    print("    (라벨이 완벽해도 confidence 정책에 따라 梯度 점수가 갈림 = FIE2026 핵심)")
    print("=" * 78)
    for name, g in sets.items():
        ol = oracle_labels(g)
        # 정책들
        pol = {}
        pol["fix .99 (작년식 과확신)"] = [("UNCERTAIN", 0.5) if l == "UNCERTAIN" else (l, 0.99) for l in ol]
        pol["fix .80"] = [("UNCERTAIN", 0.5) if l == "UNCERTAIN" else (l, 0.80) for l in ol]
        pol["fix .56 (약확신)"] = [("UNCERTAIN", 0.5) if l == "UNCERTAIN" else (l, 0.56) for l in ol]
        # 교차셋 학습 보정 (다른 셋으로 modal 학습)
        other = [v for k, v in sets.items() if k != name][0]
        modal = CAL.fit_label_modal(other)
        pol[f"보정(modal, 他셋학습)"] = CAL.apply_label_modal(ol, modal)
        # 자기셋 학습(상한 참고)
        modal_self = CAL.fit_label_modal(g)
        pol["보정(modal, 自셋상한)"] = CAL.apply_label_modal(ol, modal_self)
        print(f"\n  [{name}]  학습 modal(他셋)={modal}")
        for pname, preds in pol.items():
            print(f"    {pname:24s} {fmt(SC.evaluate(preds, g))}")

    print("\n" + "=" * 78)
    print("E3. 단순 베이스라인")
    print("=" * 78)
    for name, g in sets.items():
        always_true = [("TRUE", 0.95)] * len(g)         # 다수클래스(强正)
        print(f"  {name:14s} always TRUE@.95   {fmt(SC.evaluate(always_true, g))}")

    print("\n" + "=" * 78)
    print("E4. 기호 엔진 end-to-end (差点儿 조건부초기화 + 자연논리) — raw vs 보정")
    print("=" * 78)
    be = SymbolicBackend()
    sym_pred = {}
    for name, g in sets.items():
        preds, labels = [], []
        for x in g:
            lab, conf, _ = be.predict(x["text"], x["hypothesis"])
            preds.append((lab, conf)); labels.append(lab)
        sym_pred[name] = (preds, labels)
        print(f"\n  [{name}] 기호엔진 raw      {fmt(SC.evaluate(preds, g))}")
        # 교차셋 isotonic 보정
        other_name = [k for k in sets if k != name][0]
        op, _ = sym_pred.get(other_name, (None, None)) if other_name in sym_pred else (None, None)
        if op is not None:
            mp = CAL.fit_isotonic_bins(op, sets[other_name])
            cal = CAL.apply_isotonic(preds, mp)
            print(f"            기호엔진+보정   {fmt(SC.evaluate(cal, g))}")

    # 2차 패스: 첫 셋도 둘째 셋 학습으로 보정
    print("\n  [교차보정 완성 패스]")
    names = list(sets.keys())
    p0, _ = sym_pred[names[0]]; p1, _ = sym_pred[names[1]]
    mp10 = CAL.fit_isotonic_bins(p1, sets[names[1]])
    print(f"    {names[0]:14s} 기호+보정(他셋) {fmt(SC.evaluate(CAL.apply_isotonic(p0, mp10), sets[names[0]]))}")

    print("\n" + "=" * 78)
    print("E5. 假装类 진단 사례 (선행연구A·B 误判 유형) — 정성 검증")
    print("=" * 78)
    diag = [
        ("小张假装在系鞋带。", "小张在系鞋带。", "TRUE"),
        ("小张假装害怕。", "小张害怕。", "FALSE"),
        ("他错误地认为地球是平的。", "地球是平的。", "FALSE"),
        ("老张并没有注意到她今天穿了一件红色的连衣裙。", "她今天穿了一件红色的连衣裙。", "TRUE"),
        ("小张瞧见茶水里没有花瓣。", "茶水里有花瓣。", "FALSE"),
        ("他认为那家新开的餐厅定价过高。", "新开的餐厅定价过高。", "UNCERTAIN"),
    ]
    for text, hyp, gold in diag:
        lab, conf, r = be.predict(text, hyp)
        ok = "✓" if lab == gold else "✗"
        print(f"  {ok} gold={gold:9s} pred={lab:9s}@{conf}  «{text}»  →pred[{r['predicate']}/{r['category']}]")


if __name__ == "__main__":
    run()
