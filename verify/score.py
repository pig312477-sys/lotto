"""
봉인 채점기 — 0단계 이후 수정 금지.
봉인 구간(1048~1247)은 최종 검증 때 단 한 번만 쓴다. 한 번 태우면 되돌릴 수 없다.
탐색·중간검증에 절대 호출하지 않는다.

사용: .venv/bin/python verify/score.py --confirm-burn
"""
import argparse, json, math, sys, os
import numpy as np
from scipy.stats import spearmanr

HOLDOUT_FROM, HOLDOUT_TO = 1048, 1247
TICKET = 1000
C45_6 = math.comb(45, 6)
PASS_RHO, PASS_P = 0.10, 0.05      # 사전에 못 박은 성공 판정 기준

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirm-burn', action='store_true',
                    help='봉인 데이터를 소모한다는 명시적 확인')
    ap.add_argument('--scores', default='runs/holdout_scores.json',
                    help='{draw_no: 인기도점수} — 봉인 구간을 보지 않고 만든 예측')
    a = ap.parse_args()
    if not a.confirm_burn:
        print("거부: 봉인 데이터는 최종 검증 전용. --confirm-burn 필요."); sys.exit(2)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = json.load(open(os.path.join(root, 'data/all.json')))
    pred = {int(k): float(v) for k, v in json.load(open(os.path.join(root, a.scores))).items()}

    s, r = [], []
    for x in data:
        n = x['draw_no']
        if not (HOLDOUT_FROM <= n <= HOLDOUT_TO) or n not in pred:
            continue
        d1, sales = x['divisions'][0], (x.get('total_sales_amount') or 0)
        if sales <= 0:
            continue
        exp = (sales / TICKET) / C45_6
        s.append(pred[n]); r.append((d1.get('winners', 0) if d1 else 0) / exp)
    s, r = np.array(s), np.array(r)
    rho, p = spearmanr(s, r)
    lo, hi = s <= np.percentile(s, 20), s >= np.percentile(s, 80)
    gap = r[hi].mean() / r[lo].mean()
    ok = (rho >= PASS_RHO) and (p < PASS_P)
    print(f"봉인 구간 {HOLDOUT_FROM}~{HOLDOUT_TO}, 채점 회차 {len(s)}")
    print(f"  Spearman rho = {rho:+.4f}  (기준 >= {PASS_RHO})")
    print(f"  p            = {p:.4f}     (기준 <  {PASS_P})")
    print(f"  하위20% R {r[lo].mean():.3f} vs 상위20% R {r[hi].mean():.3f} -> 격차 {gap:.2f}배")
    print(f"\n판정: {'통과' if ok else '불통과'}")
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
