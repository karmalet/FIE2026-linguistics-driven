# -*- coding: utf-8 -*-
"""
신경-기호 등급 사실성 추론 — 기호(symbolic) 코어 엔진.
5단계 중 [2]함의서명 ~ [4]등급 veridicality를 어휘+규칙으로 근사 구현.
핵심: 袁毓林 '差点儿' 확률적 意合語法 기반 조건부 초기화(假装/哀叹/怀疑) + 자연논리 극성 합성.
출력 S∈[-1,1] (부호=진릿값 방향, |S|=확신).
"""
from __future__ import annotations
import re
import lexicon as L

# 긴 표지 우선(没有가 没으로 이중 카운트되지 않도록 alternation 순서 중요)
_NEG_RE = re.compile(r"没有|并非|无法|不曾|未曾|没|不|未|无|非")


def _has(text: str, markers) -> bool:
    return any(m in text for m in markers)


def _neg_parity(s: str) -> bool:
    """문자열에 (홀수개) 부정이 있어 명제가 '부정형'인지 근사."""
    # '无'는 '无法/无人' 등에서만 부정으로; 단독 명사성 오탐 줄이려 '无'는 제외 카운트
    toks = [t for t in _NEG_RE.findall(s) if t != "无"]
    return (len(toks) % 2) == 1


def _vp_is_state(complement: str, hypothesis: str) -> bool:
    core = complement or hypothesis
    return _has(core, L.STATE_WORDS) and not _has(core, ["系", "摔", "解", "做", "走", "拿", "写", "打", "踢", "跑", "跳"])


def conditional_init(pred: str, cat: str, complement: str, hypothesis: str, full: str):
    """크로스도메인/漂移 谓词의 조건부 확률 초기화. 반환 (S0, note)."""
    # ① 假装类 双重叙实
    if cat == "double":
        # 知域·多声성(평가/반어/뉴스표제 신호) → U
        if _has(full, L.MULTIVOICE) or _has(full, L.EVAL_REVERSE) or _has(full, ["奋斗", "努力"]) and _has(full, L.STRENGTHEN):
            return 0.0, "假装-知域/多声성→U"
        if _vp_is_state(complement, hypothesis):
            return -0.8, "假装+상태→거짓"
        return +0.7, "假装+동작→동작은 발생(참)"
    # ② 哀叹/感叹류
    if pred in ("哀叹", "感叹", "慨叹", "叹息"):
        if _has(complement, L.SUBJECTIVE_FUTURE):
            return 0.0, "哀叹+주관/미래→U"
        return +0.9, "哀叹+객관/과거→전제 유지"
    # ③ 怀疑류
    if pred in ("怀疑",):
        if ("不再怀疑" in full) or ("谁也不" in full) or ("无人怀疑" in full) or ("毫不怀疑" in full):
            return +0.85, "怀疑+반기대/부정누적→강한 사실"
        return 0.0, "怀疑→중립(U)"
    # 记得/不记得: 과거사건 전제 → factive-like
    if pred in ("记得", "不记得", "感觉", "感到", "感觉到"):
        return +0.8, "漂移→과거/지각 전제(약한 사실)"
    # 担心/害怕/埋怨/抱怨: 사실 전제 경향
    if pred in ("担心", "担忧", "害怕", "埋怨", "抱怨", "恐怕", "怀念"):
        return +0.75, "정서반응→기저 사실 전제"
    return 0.0, "漂移 기본 중립"


def infer(text: str, hypothesis: str):
    """반환: dict(S, label, conf, predicate, category, trace)"""
    pred, cat, pos = L.find_predicate(text, hypothesis)
    trace = []
    if pred is None:
        # [unknown] 谓词 미식별 → 보수적 U (실전 LLM 백엔드가 처리)
        return {"S": 0.0, "label": "UNCERTAIN", "conf": 0.5, "predicate": None,
                "category": "unknown", "trace": ["谓词 미식별→U"]}
    trace.append(f"[1] 谓词='{pred}' ({cat})")

    before = text[:pos]                      # 谓词 앞(주어·평가부사·谓词부정 영역)
    complement = text[pos + len(pred):]      # 보문 영역
    pred_neg = any(before.rstrip().endswith(n) or before[-3:].find(n) != -1 for n in L.NEG)
    pred_neg = _has(before[max(0, len(before) - 4):], L.NEG)  # 谓词 직전 부정
    eval_rev = _has(before, L.EVAL_REVERSE)

    # [2]~[4] 함의 서명별 기본 극성 S_base (부호=보문-as-stated 진릿값 방향)
    if cat == "factive":
        S_base = 0.95; trace.append("[2] 正叙实(+/+): 보문 참, 谓词부정 불변")
    elif cat == "antifactive":
        S_base = -0.92; trace.append("[2] 反叙实: 보문 거짓")
    elif cat in ("double", "drift"):
        S_base, note = conditional_init(pred, cat, complement, hypothesis, text)
        trace.append(f"[3-init] 조건부 초기화(差点儿): {note} → S0={S_base:+.2f}")
    elif cat == "neutral":
        S_base = 0.0; trace.append("[2] 中性(o/o): 기본 U")
    else:
        S_base = 0.0

    # [3] 극성 합성
    if eval_rev and cat in ("neutral", "double", "drift", "factive"):
        # 평가부사 '错误地' → 보문 극성 반전(中性→反叙实)
        S_base = -0.99 if cat == "neutral" else -abs(S_base) if S_base != 0 else -0.9
        trace.append("[3] 평가부사(错误地 등)→극성 반전→反叙实화")

    # 施为동사 부정 → 차단(U)
    if pred in L.PERFORMATIVE and pred_neg:
        trace.append("[3] 施为동사+부정→차단(U)")
        return {"S": 0.0, "label": "UNCERTAIN", "conf": 0.5, "predicate": pred,
                "category": cat, "trace": trace}

    # 谓词 부정 처리: 正叙实은 불변, 그 외(中性/反叙实)는 약화/반전
    if pred_neg:
        if cat == "factive":
            trace.append("[3] 谓词부정+正叙实→전제 보존(불변)")
        elif cat == "neutral":
            trace.append("[3] 谓词부정+中性→여전히 U")
        else:
            S_base = -S_base if S_base != 0 else S_base
            trace.append("[3] 谓词부정→극성 반전(진성 부정)")

    # 내포절 부정 ↔ 가설 부정 패리티 비교 (능동적으로 보문-as-stated vs hypothesis)
    comp_neg = _neg_parity(complement) ^ (False)
    hyp_neg = _neg_parity(hypothesis)
    # 谓词부정을 보문 패리티에서 제외하기 위해 complement만 사용(이미 before/after 분리)
    mismatch = (comp_neg != hyp_neg)
    if mismatch and abs(S_base) > 0.06:
        S_base = -S_base
        trace.append("[3] 내포절 부정↔가설 패리티 불일치→진릿값 반전")

    # 多声성·被动·情态 약화
    if _has(text, L.MULTIVOICE):
        S_base *= 0.78; trace.append("[3] 多声성/被动→화자책임 분리(약화)")
    if _has(text, L.MODAL):
        S_base *= 0.85; trace.append("[3] 情态/불확실→약화")
    if _has(text, L.STRENGTHEN) and abs(S_base) > 0.06:
        S_base = (1 if S_base > 0 else -1) * min(1.0, abs(S_base) + 0.05)
        trace.append("[3] 강화부사→확신 상향")

    # [4] 등급 veridicality 확정
    S = max(-1.0, min(1.0, S_base))
    if abs(S) <= 0.06:
        label, conf = "UNCERTAIN", 0.5
    else:
        label = "TRUE" if S > 0 else "FALSE"
        conf = round(min(max(abs(S), 0.51), 1.0), 2)
    trace.append("[4] S=%+.2f -> %s@%s" % (S, label, conf))
    return {"S": S, "label": label, "conf": conf, "predicate": pred, "category": cat, "trace": trace}
