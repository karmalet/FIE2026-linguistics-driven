# -*- coding: utf-8 -*-
"""
FIE2026 로컬 LLM 실행기 (ASUS GX10 / NVIDIA GB10).
Ollama(기본 탑재) 로컬 모델로 5단계 신경-기호 파이프라인을 돌린다.

사전 준비 (GX10 터미널):
  ollama serve &                       # 서버 기동
  ollama pull deepseek-r1:32b          # 모델 내려받기(권장 목록은 --list-models)
  pip install openai                   # OpenAI 호환 클라이언트

사용 예:
  # 단일 모델 + 자기일관성 5회
  python run_local.py --data ../sample_sets/sample_20260502.json --model deepseek-r1:32b --n 5
  # 앙상블(불일치→置信도) + 기호 veto + 보정
  python run_local.py --data <test.json> --ensemble deepseek-r1:32b,qwen3:32b,gpt-oss:120b \
                      --calibrate-from ../sample_sets/sample_20260502.json --out submission.json
  # gpt-oss reasoning effort
  python run_local.py --data ... --model gpt-oss:120b --reasoning high --n 3
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer as SC
import calibration as CAL
from local_backends import OllamaBackend, EnsembleBackend, RECOMMENDED


def build(args):
    common = dict(base_url=args.base_url, temperature=args.temperature,
                  n_consistency=args.n, reasoning=args.reasoning, num_ctx=args.num_ctx)
    if args.ensemble:
        models = [m.strip() for m in args.ensemble.split(",") if m.strip()]
        bks = [OllamaBackend(m, **common) for m in models]
        return EnsembleBackend(bks, symbolic_veto=not args.no_veto), f"ensemble({len(models)})"
    return OllamaBackend(args.model, **common), args.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="입력 JSON (id,text,hypothesis[,factivity,confidence])")
    ap.add_argument("--model", default="deepseek-r1:32b")
    ap.add_argument("--ensemble", default="", help="콤마구분 모델목록(지정 시 앙상블)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1", dest="base_url")
    ap.add_argument("--n", type=int, default=1, help="자기일관성 샘플 수")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--reasoning", default=None, help="gpt-oss: low|medium|high")
    ap.add_argument("--num-ctx", type=int, default=8192, dest="num_ctx")
    ap.add_argument("--no-veto", action="store_true", help="기호 veto 비활성")
    ap.add_argument("--calibrate-from", default="", dest="cal_from", help="라벨된 샘플집으로 置信도 보정 학습")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만(테스트용)")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        print("GX10(128GB) 권장 모델 (Ollama 태그):")
        for k, v in RECOMMENDED.items():
            print(f"  {k:18s} {v}")
        return
    if not args.data:
        ap.error("--data 필요")

    data = json.load(open(args.data, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]
    backend, name = build(args)
    print(f"[run] model={name}  n={args.n}  veto={not args.no_veto}  items={len(data)}")

    preds, rows = [], []
    t0 = time.time()
    for i, x in enumerate(data, 1):
        lab, conf, info = backend.predict(x["text"], x["hypothesis"])
        preds.append((lab, conf))
        rows.append({"id": x.get("id", i), "factivity": lab, "confidence": conf})
        if i % 20 == 0 or i == len(data):
            dt = time.time() - t0
            print(f"  {i}/{len(data)}  ({dt:.0f}s, {dt/i:.1f}s/item)")

    # 置信도 보정(샘플집 학습 → 예측에 적용)
    if args.cal_from and os.path.exists(args.cal_from):
        cal_set = json.load(open(args.cal_from, encoding="utf-8"))
        modal = CAL.fit_label_modal(cal_set)
        preds = CAL.apply_label_modal([p[0] for p in preds], modal)
        for r, (l, c) in zip(rows, preds):
            r["factivity"], r["confidence"] = l, c
        print(f"[calib] label-modal={modal} 적용")

    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[out] {args.out} 저장 ({len(rows)}건)")

    # gold 있으면 채점
    if all("factivity" in x and "confidence" in x for x in data):
        res = SC.evaluate(preds, data)
        print(f"[score] norm={res['norm']:.4f}  label_acc={res['label_acc']:.3f}  "
              f"band_acc={res['band_acc']:.3f}  Σ={res['total']}/{res['n']}")


if __name__ == "__main__":
    main()
