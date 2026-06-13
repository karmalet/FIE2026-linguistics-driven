# -*- coding: utf-8 -*-
"""추론 백엔드. SymbolicBackend(오프라인 실행 가능) / LLMBackend(키 필요, 즉시 실행 가능 코드)."""
from __future__ import annotations
import engine as E


class SymbolicBackend:
    """어휘+규칙 기반 기호 코어. LLM 없이 실행."""
    name = "symbolic"

    def predict(self, text, hypothesis):
        r = E.infer(text, hypothesis)
        return r["label"], r["conf"], r


# ----------------------------------------------------------------------------
# 5단계 신경-기호 프롬프트(LLM 백엔드용). 키를 꽂으면 즉시 동작.
SYSTEM_PROMPT = """你是严谨的汉语叙实性推理专家。仅依据语言内部的分析性知识（谓词对补足语小句的预设/蕴含、否定与语境算子的极性合成），按五步法推理，最后只输出 JSON。
真值与置信度规则：factivity∈{TRUE,FALSE,UNCERTAIN}；TRUE/FALSE 时 confidence∈(0.50,1.00]（两位小数，越纯粹的分析性叙实/反叙实且无干扰越接近1.00，有情态/多声/构式等削弱因素则降低，如“大概”约0.70）；UNCERTAIN 时 confidence=0.50。
五步：[1]识别关键谓词 [2]判定蕴含签名(正叙实/反叙实/双重叙实(假装类)/中性/漂移) [3]极性合成(小句内部否定→谓词否定→评价副词(如‘错误地’反转)→被动/多声(脱离断言)→情态(削弱)；区分谓词否定与小句内部否定) [4]计算真实性分数 S∈[-1,1] [5]映射输出。
跨域/漂移谓词按条件函数初始化(差点儿式)：假装默认-0.8(动作→+0.7/状态→-0.8/知域多声→U)；哀叹默认+0.8(客观过去→+0.9/主观将来→U)；怀疑默认0.0(反预期否定累积→+0.85)。"""

USER_TEMPLATE = """请判断（先给五步推理，最后一行只输出 JSON）：
text: {text}
hypothesis: {hypothesis}
最后一行: {{"factivity": "...", "confidence": 0.00}}"""


class LLMBackend:
    """OpenAI/Gemini/DeepSeek 호환. 환경변수 키 필요. 네트워크 가능한 환경(사용자 PC)에서 실행."""
    name = "llm"

    def __init__(self, model="gpt-5", provider="openai", temperature=0.2, n_consistency=1):
        self.model, self.provider, self.temperature, self.n = model, provider, temperature, n_consistency

    def _call_once(self, text, hypothesis):
        import os, json, re
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text, hypothesis=hypothesis)}]
        if self.provider in ("openai", "deepseek"):
            from openai import OpenAI
            base = "https://api.deepseek.com" if self.provider == "deepseek" else None
            key = os.environ["DEEPSEEK_API_KEY"] if self.provider == "deepseek" else os.environ["OPENAI_API_KEY"]
            client = OpenAI(api_key=key, base_url=base)
            resp = client.chat.completions.create(model=self.model, messages=msgs, temperature=self.temperature)
            content = resp.choices[0].message.content
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            gm = genai.GenerativeModel(self.model, system_instruction=SYSTEM_PROMPT)
            content = gm.generate_content(USER_TEMPLATE.format(text=text, hypothesis=hypothesis)).text
        else:
            raise ValueError(self.provider)
        m = re.findall(r"\{[^{}]*factivity[^{}]*\}", content)
        obj = json.loads(m[-1]) if m else {"factivity": "UNCERTAIN", "confidence": 0.5}
        return obj["factivity"], float(obj["confidence"]), content

    def predict(self, text, hypothesis):
        """자기일관성 N회 → 라벨 빈도로 confidence 보정(§3.5)."""
        from collections import Counter
        if self.n <= 1:
            lab, conf, raw = self._call_once(text, hypothesis)
            return lab, conf, {"raw": raw}
        votes, confs = [], []
        for _ in range(self.n):
            lab, conf, _ = self._call_once(text, hypothesis)
            votes.append(lab.strip().upper()); confs.append(conf)
        c = Counter(votes); top, k = c.most_common(1)[0]
        p = k / self.n
        if top == "UNCERTAIN" or 0.45 <= p <= 0.55:
            return "UNCERTAIN", 0.5, {"votes": votes}
        return top, round(min(max(p, 0.51), 1.0), 2), {"votes": votes}
