# -*- coding: utf-8 -*-
"""
로컬 LLM 백엔드 (ASUS GX10 / NVIDIA GB10, 128GB 통합메모리).
Ollama(기본 탑재) 또는 vLLM의 OpenAI 호환 엔드포인트를 사용.
기존 backends.SYSTEM_PROMPT(5단계 HCoT + 袁毓林 差点儿 조건부초기화)를 그대로 재사용.

핵심 기능
- 추론모델 thinking 태그(<think>…</think>, analysis 채널) 자동 제거 후 JSON 파싱
- 자기일관성(N회) → 라벨 빈도로 置信도 추정(§3.5)
- 다중모델 앙상블 → 모델 간 불일치를 置信도 신호로(HUST #51식)
- 기호엔진(engine.infer) 교차검증: 假装 双重叙实·내포절 부정 등 고정밀 사례 veto(neuro-symbolic)
"""
from __future__ import annotations
import os, re, json, time
from collections import Counter

from backends import SYSTEM_PROMPT, USER_TEMPLATE   # 5단계 프롬프트 재사용
import engine as SYM                                  # 기호 코어(교차검증용)

_JSON_RE = re.compile(r"\{[^{}]*?factivity[^{}]*?\}", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>|<\|channel\|>analysis.*?<\|message\|>", re.S)


def _parse(content: str):
    """LLM 출력에서 (label, conf) 추출. thinking 제거 후 마지막 JSON 채택."""
    body = _THINK_RE.sub("", content or "")
    m = _JSON_RE.findall(body) or _JSON_RE.findall(content or "")
    if not m:
        return None
    try:
        obj = json.loads(m[-1])
        lab = str(obj["factivity"]).strip().upper()
        conf = float(obj.get("confidence", 0.5))
        if lab not in ("TRUE", "FALSE", "UNCERTAIN"):
            return None
        if lab == "UNCERTAIN":
            conf = 0.5
        else:
            conf = min(max(conf, 0.5001), 1.0)
        return lab, round(conf, 2)
    except Exception:
        return None


class OllamaBackend:
    """Ollama OpenAI 호환 API. GX10에 기본 탑재. `ollama serve` 후 사용.
    예) OllamaBackend('deepseek-r1:32b', n_consistency=5)"""
    def __init__(self, model, base_url="http://localhost:11434/v1", api_key="ollama",
                 temperature=0.6, n_consistency=1, reasoning=None, num_ctx=8192, timeout=600):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.temperature = temperature
        self.n = max(1, n_consistency)
        self.reasoning = reasoning            # gpt-oss: 'low'|'medium'|'high'
        self.num_ctx = num_ctx

    def _system(self):
        s = SYSTEM_PROMPT
        if self.reasoning:                    # gpt-oss reasoning effort
            s = f"Reasoning: {self.reasoning}\n" + s
        return s

    def _once(self, text, hypothesis, temp):
        msgs = [{"role": "system", "content": self._system()},
                {"role": "user", "content": USER_TEMPLATE.format(text=text, hypothesis=hypothesis)}]
        kw = dict(model=self.model, messages=msgs, temperature=temp)
        # Ollama 확장: 컨텍스트 길이
        kw["extra_body"] = {"options": {"num_ctx": self.num_ctx}}
        r = self.client.chat.completions.create(**kw)
        return r.choices[0].message.content

    def predict(self, text, hypothesis):
        votes, confs, raws = [], [], []
        for i in range(self.n):
            temp = 0.0 if self.n == 1 else self.temperature
            try:
                c = self._once(text, hypothesis, temp)
            except Exception as e:
                raws.append(f"ERR:{e}"); continue
            raws.append(c)
            p = _parse(c)
            if p:
                votes.append(p[0]); confs.append(p[1])
        if not votes:
            return "UNCERTAIN", 0.5, {"raw": raws, "note": "parse_fail→U"}
        if self.n == 1:
            return votes[0], confs[0], {"raw": raws[0]}
        # 자기일관성: 다수 라벨 + 빈도→置信도
        c = Counter(votes); top, k = c.most_common(1)[0]
        p = k / len(votes)
        if top == "UNCERTAIN" or 0.45 <= p <= 0.55:
            return "UNCERTAIN", 0.5, {"votes": votes}
        # 빈도 p와 모델이 답한 평균 conf의 보수적 결합(과확신 억제)
        avg = sum(cf for v, cf in zip(votes, confs) if v == top) / k
        conf = round(min(max(min(p, avg), 0.51), 1.0), 2)
        return top, conf, {"votes": votes, "p": round(p, 2)}


class EnsembleBackend:
    """다중 로컬 모델 앙상블. 불일치를 置信도 신호로(HUST #51)."""
    def __init__(self, backends, symbolic_veto=True):
        self.backends = backends              # [OllamaBackend, ...]
        self.symbolic_veto = symbolic_veto

    def predict(self, text, hypothesis):
        results = [be.predict(text, hypothesis) for be in self.backends]
        labels = [r[0] for r in results]
        c = Counter(labels); top, k = c.most_common(1)[0]
        agree = k / len(labels)
        # 기호 veto: 假装 双重叙实/내포절 부정 등 분석적으로 확실한 경우 기호 결과 우선
        if self.symbolic_veto:
            s = SYM.infer(text, hypothesis)
            if s["category"] in ("double",) or "패리티 불일치" in " ".join(s["trace"]) or "평가부사" in " ".join(s["trace"]):
                return s["label"], s["conf"], {"src": "symbolic_veto", "llm": labels, "trace": s["trace"][-2:]}
        if top == "UNCERTAIN" or agree <= 0.5:
            return "UNCERTAIN", 0.5, {"llm": labels, "agree": round(agree, 2)}
        # 동의율→置信도, 모델별 conf 평균과 보수 결합
        avg = sum(r[1] for r in results if r[0] == top) / k
        conf = round(min(max(min(agree, avg), 0.51), 1.0), 2)
        return top, conf, {"llm": labels, "agree": round(agree, 2)}


# GX10(128GB)에서 권장 모델 (Ollama 태그) — 4비트 기준 단일 장비 적재 가능
RECOMMENDED = {
    "deepseek-r1:32b": "★중국어 추론 1순위(FIE2025 FT 챔피언 백본). 中文 native, reasoning",
    "qwen3:32b":       "★하이브리드 thinking, 최신 中文 최상급",
    "qwq:32b":         "순수 추론모델, 中文 양호",
    "glm4:32b":        "Zhipu, 中文 언어품질·어용 뉘앙스 강함",
    "gpt-oss:120b":    "OpenAI 오픈웨이트(~65GB 4bit), 강한 추론·앙상블 다양성(英 중심)",
    "gpt-oss:20b":     "경량(16GB), 빠른 프로토타이핑",
    "gemma3:27b":      "Google, 다국어·앙상블 멤버",
}
