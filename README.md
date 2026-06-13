# FIE2026 — Neuro-symbolic Graded-Veridicality 중국어 사실성 추론

> **CCL2026 Task1 (第二届中文叙实性推理评测, FIE2026)** 참가용 코드.
> 5단계 신경-기호(Neuro-symbolic) 파이프라인 + 袁毓林 '差点儿' 조건부 초기화 + 9구간 置信도 보정.
> 클라우드 API(GPT/Gemini)와 **로컬 LLM(ASUS GX10 / Ollama: deepseek-r1, qwen3, gpt-oss …)** 모두 지원.

---

## 1. 이 코드가 푸는 문제

`text`(주蕴含句)를 근거로 `hypothesis`(被蕴含句)의 진릿값을 판정한다.
출력은 **라벨 + 확신도** 두 가지:

```json
{ "factivity": "TRUE | FALSE | UNCERTAIN", "confidence": 0.50~1.00 }
```

FIE2026은 작년과 달리 ① 谓词 정보 미제공, ② 3-shot 상한, ③ **확신도를 9개 구간으로 梯度 채점**한다.
특히 ③ 때문에 **라벨이 맞아도 확신도 구간이 어긋나면 점수가 깎인다**(같은 구간 1점 · 인접 0.6827 · 비인접 0점).

## 2. 핵심 아이디어 (3축)

| 축 | 내용 |
|----|------|
| **등급 veridicality** | 참/거짓/불확정 3분류 대신, 단일 점수 `S∈[-1,+1]`로 보고 9구간으로 양자화 |
| **差点儿 조건부 초기화** | 假装·哀叹·怀疑 같은 漂移 동사를 고정 가중치가 아니라 *문맥 조건부 함수*로 초기화 |
| **확신도 보정(calibration)** | 자기일관성·샘플집 학습으로 확신도를 9구간에 정렬 (← 올해 점수의 핵심) |

자세한 설계는 상위 폴더의 `FIE2026_논문초안_정박사.md`, `FIE2026_HCoT설계서_정박사.md` 참조.

---

## 3. 디렉토리 구조

```
fie2026/
├─ scorer.py          # 공식 9구간 梯度 채점기 + 확신도→구간 매핑
├─ lexicon.py         # 叙实성 어휘집(正/反/双重/中性/漂移) + 연산자 표지
├─ engine.py          # 기호 엔진: 差点儿 조건부초기화 + 자연논리 극성 합성 → S
├─ calibration.py     # 확신도 보정(라벨별 modal / isotonic 재매핑)
├─ backends.py        # SymbolicBackend(오프라인) + LLMBackend(클라우드: OpenAI/Gemini/DeepSeek)
├─ local_backends.py  # OllamaBackend / EnsembleBackend (로컬 LLM, GX10)
├─ run_experiment.py  # [검증용] 모델 없이 채점기·규칙·보정 실험 (E1~E5)
├─ run_local.py       # [실전용] 로컬 LLM(deepseek 등)으로 추론·채점·제출파일 생성
├─ results_log.txt    # 최근 오프라인 실험 결과
└─ fie2026_run_experiment_vs_run_local_flow.svg  # 실행 흐름도
```

데이터(상위 폴더 `sample_sets/`):
- `sample_20260401.json` (69건, `confidence`는 실수 float)
- `sample_20260502.json` (497건, `confidence`는 구간 문자열 예: `"(0.875, 1]"`)
두 인코딩 모두 `scorer.py`가 자동 처리한다.

---

## 4. 설치

```bash
# 공통
python3 -m pip install openai          # LLM 백엔드용(오프라인 실험은 불필요)

# 로컬 LLM (ASUS GX10 / NVIDIA GB10, DGX OS)
ollama serve &                         # GX10 기본 탑재
ollama pull deepseek-r1:32b            # 권장 모델은 README §8
```

> 오프라인 실험(`run_experiment.py`)은 **외부 의존성 없이** 표준 파이썬만으로 실행된다.

---

## 5. 빠른 시작

### (A) 모델 없이 검증 — `run_experiment.py`
채점기·규칙엔진·보정이 제대로 도는지 확인. **인터넷·GPU 불필요.**
```bash
python3 run_experiment.py
```

### (B) 로컬 LLM 실전 실행 — `run_local.py` (GX10, deepseek)
```bash
# 단일 모델 + 자기일관성 5회
python3 run_local.py --data ../sample_sets/sample_20260502.json --model deepseek-r1:32b --n 5

# 앙상블(모델 불일치→확신도) + 기호 veto + 샘플집 보정 → 제출파일
python3 run_local.py --data <test.json> \
    --ensemble deepseek-r1:32b,qwen3:32b,gpt-oss:120b \
    --calibrate-from ../sample_sets/sample_20260502.json --out submission.json

python3 run_local.py --list-models       # GX10 권장 모델 목록
```

---

## 6. 두 실행 파일의 차이 (★자주 헷갈림)

| | `run_experiment.py` | `run_local.py` |
|---|---|---|
| 목적 | 채점기·규칙·보정 **검증** | **실전** 추론·제출파일 생성 |
| LLM 호출 | **없음** (기호 규칙만) | **있음** (deepseek 등) |
| 네트워크/GPU | 불필요 | 필요(Ollama 서버) |
| 비유 | 채점표가 공정한지 모의점검 | 선수가 실제로 시합 |

흐름도: `fie2026_run_experiment_vs_run_local_flow.svg`

### `run_experiment.py` 함수 호출 순서 (`run()` 하나가 차례로 진행)
1. **E1 채점기 검증** — `SC.evaluate(정답, 정답)` = 1.0 확인
2. **E2 확신도 보정 실험(핵심)** — `oracle_labels()` → `CAL.fit_label_modal()` → `SC.evaluate()`. 라벨이 완벽해도 확신도 정책에 따라 점수가 0.09~0.90으로 갈림을 보임
3. **E3 베이스라인** — "무조건 TRUE@0.95" 점수
4. **E4 기호엔진** — `SymbolicBackend.predict()` → `engine.infer()` → `lexicon.find_predicate()` / `conditional_init()` → `SC.evaluate()`
5. **E5 진단 사례** — 까다로운 예문 6개를 `engine.infer()`로 풀어 정답 대조

### `run_local.py` 함수 호출 순서 (deepseek 경로)
1. `main()` → `build()` → `OllamaBackend("deepseek-r1:32b")` 준비(Ollama 서버 연결)
2. `json.load()`로 문항 반복
3. `backend.predict()` → 내부 `_once()`가 deepseek에 5단계 프롬프트 전송·수신 (`--n`만큼 반복)
4. `_parse()` — `<think>…</think>` 제거 후 `{factivity, confidence}` 추출, 다수결+빈도로 확신도
5. `CAL.apply_label_modal()` 확신도 보정
6. `submission.json` 저장 → 정답 있으면 `SC.evaluate()` 채점

---

## 7. 자주 묻는 질문 (FAQ)

### Q. `_once()`는 안 보이고 `_call_once`만 보인다. 뭐가 다른가?
**둘 다 있고, 파일이 다르다.**
- `backends.py`(클라우드용 GPT·Gemini) → `_call_once`
- `local_backends.py`(로컬 Ollama용, **GX10의 deepseek가 쓰는 파일**) → `_once`

이름만 다른 쌍둥이로, 하는 일은 같다: **"모델에게 딱 한 번 물어 답 하나를 받아오는"** 최소 단위 함수.
`predict()`(여러 번 묻고 종합하는 큰 함수)가 이 작은 함수를 필요한 횟수만큼 반복 호출한다.

### Q. 왜 같은 문제를 5번 물어보나? (`--n 5`)
**자기일관성(self-consistency).** LLM은 계산기가 아니라 확률적으로 답하므로, 온도(temperature)가 0보다 크면 같은 질문에도 답이 매번 조금씩 달라질 수 있다.
- `--n 1` → `temp=0.0`, 한 번만(결정적). 빠른 실행용.
- `--n 5` → `temp=0.6`, 다섯 번 물어 답 5개를 모음.

5번 묻는 이유는 두 가지다.
1. **정확도·안정성**: 한 번 답은 우연히 틀릴 수 있으나, 다수가 같으면 더 믿을 만하다. `Counter(votes).most_common(1)`로 **다수결**(예: 5개 중 TRUE 4개 → TRUE).
2. **확신도를 공짜로 획득**: 몇 개가 일치했는지 비율 `p = k/5`가 곧 확신도가 된다. 5개 다 같으면 1.0(매우 확신), 3개면 0.6(애매), 반반이면(`0.45≤p≤0.55`) **UNCERTAIN** 처리.

비유하면 **전문가 5명에게 따로 같은 문제를 물어보고, 몇 명이 동의했는지로 "얼마나 확실한가"를 매기는 것**. 이 확신도가 FIE2026의 9구간 채점에 직접 들어가므로 단순 안정화를 넘어 **점수와 직결**된다.
또한 `conf = min(p, avg)`로 처리해, 모델이 혼자 "99% 확신"이라 우겨도 실제 일치 비율을 넘지 못하게 **과확신을 억제**한다. (논문 §3.5의 방법)

---

## 8. GX10(128GB)에서 돌아가는 권장 모델 — 中文 사실성 추론

| Ollama 태그 | 메모 |
|-------------|------|
| `deepseek-r1:32b` | ★1순위. FIE2025 미세조정 챔피언(CAS#49) 백본. 中文 native·추론특화. ~20GB(4bit) |
| `qwen3:32b`       | ★최신 中文 최상급, thinking/non-thinking 하이브리드 |
| `qwq:32b`         | 순수 추론모델 |
| `glm4:32b`        | Zhipu. 中文 언어품질·어용 뉘앙스 강함(假装·多声성 유리) |
| `gpt-oss:120b`    | OpenAI 오픈웨이트 ~65GB(4bit). 강한 추론·앙상블 다양성 |
| `gpt-oss:20b`     | 16GB, 빠른 프로토타이핑 |
| `gemma3:27b`      | Google, 다국어 앙상블 멤버 |

권장 앙상블: `deepseek-r1:32b + qwen3:32b + gpt-oss:120b` → 모델 불일치를 확신도로, 假装·부정 사례는 기호 veto.
671B급(DeepSeek-V3/R1 풀)은 단일 128GB 불가 → 32B distill 또는 GX10 2대 200GbE 클러스터.

---

## 9. 채점 방식 (9구간 梯度)

`scorer.py`가 (라벨, 확신도)를 9개 叙实강도 구간으로 매핑:
`강反·较强反·较弱反·弱反 · 非 · 弱正·较弱正·较强正·강正`
점수: 같은 구간 **1.0** / 인접 구간 **σ≈0.6827** / 그 외 **0.0**. 최종 점수는 전 문항 합.

---

## 10. 오프라인 실험 결과 요약 (`results_log.txt`)

| 실험 | S2(497건) norm | 의미 |
|------|---------------:|------|
| E1 채점기 검증(정답→정답) | 1.0000 | 채점기 정확 |
| E2 오라클라벨 + 확신도 `fix .99` | 0.9038 | 라벨 완벽 + 적정 확신도 |
| E2 오라클라벨 + 확신도 `fix .56` | **0.0911** | **라벨 완벽해도 확신도 틀리면 붕괴** |
| E3 무조건 TRUE@.95 | 0.7414 | 다수클래스 베이스라인 |
| E4 기호엔진 단독 | 0.4321 | 어휘 커버리지 한계 → LLM 백엔드 필요 |
| E5 진단 사례(假装 등) | **6/6 정답** | 差点儿·双重叙实·내포절부정 규칙 검증 |

> E2가 보여주듯 **확신도 보정이 올해 점수의 승부처**다. E4가 낮은 건 기호 규칙이 아니라 谓词 자가식별의 커버리지 때문이며, 이를 LLM 백엔드(`run_local.py`)가 메운다 — 그래서 "신경-기호".

---

## 11. 주의 / 라이선스 / 인용

- 본 저장소의 기호 코어·채점기는 외부 의존성 없이 동작. LLM 백엔드는 키 또는 로컬 Ollama 필요.
- 데이터(`sample_sets/`)는 조직위 FIE2026 저장소의 CC BY 4.0 라이선스를 따른다.
- 인용: 본 방법은 Cong et al.(2025) HCoT, 袁毓林의 叙实性·差点儿 이론, de Marneffe(2012)/Rudinger(2018) 등급 veridicality, MacCartney & Manning(2009) 자연논리를 종합·확장한 것.
- CCL2026 双盲 심사를 위해 정식 투고 시 저자 식별 정보는 제거할 것.
