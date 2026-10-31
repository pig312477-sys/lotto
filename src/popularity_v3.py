"""
인기도 모델 v3 — 자동 당첨자를 대조군으로 쓰는 희석 제거 모형.

핵심: 데이터에 1등 당첨자의 자동/수동 구분이 있다(262회~, 983회차).
자동 구매는 균등난수라 번호 편향이 0이다. 따라서 같은 회차 안에서
수동 당첨자와 자동 당첨자의 비(比)를 보면 판매량·시대변화·offset이
전부 상쇄되고 **사람의 순수한 번호 편향만** 남는다.

    당첨자 중 수동일 확률 pi 에 대해
    logit(pi) = log(N수동/N자동) + X*beta
    -> 이항 회귀의 계수가 곧 수동 구매자의 선호 로그배율 beta

v1/v2가 실패한 이유: 회차별 총당첨자수만 보면 자동 66%가 신호를 묽게 만들어
계수가 작아지고 시대별로 불안정해 보인다. 대조군을 쓰면 그 문제가 사라진다.
"""
import json, os, sys, math
import numpy as np
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from features import extract

KEEP = ['le31', 'rng', 'row_max']     # 공선성 큰 sm 제외, FDR 생존 재료
C45_6 = math.comb(45, 6)

def load_pairs(path, upto=None):
    rows = []
    for x in json.load(open(path)):
        if upto and x['draw_no'] > upto:
            continue
        wc = x.get('winners_combination') or {}
        a = wc.get('auto', 0)
        mn = wc.get('manual', 0) + wc.get('semi_auto', 0)
        if a + mn == 0:
            continue
        f = extract(x['numbers'])
        f.update(auto=a, man=mn, t=x['draw_no'])
        rows.append(f)
    return rows

def fit(rows):
    X = np.array([[r[k] for k in KEEP] for r in rows], float)
    Y = np.column_stack([[r['man'] for r in rows], [r['auto'] for r in rows]])
    mu, sd = X.mean(0), X.std(0)
    m = sm.GLM(Y, sm.add_constant((X - mu) / sd), family=sm.families.Binomial()).fit()
    A = sum(r['auto'] for r in rows); M = sum(r['man'] for r in rows)
    return dict(beta=m.params[1:], se=m.bse[1:], p=m.pvalues[1:],
                mu=mu, sd=sd, alpha=A / (A + M), n=len(rows), obs=X)

def multiplier(model, combos):
    """평균 조합 대비 예상 공동당첨자 배율. 낮을수록 좋다. 희석(alpha) 반영."""
    F = np.array([[extract(c)[k] for k in KEEP] for c in combos], float)
    a = model['alpha']
    return a + (1 - a) * np.exp(((F - model['mu']) / model['sd']) @ model['beta'])

def recommend(path, n_out=10, pool=300000, seed=0):
    rows = load_pairs(path)
    m = fit(rows)
    lo = np.percentile(m['obs'], 1, axis=0)
    hi = np.percentile(m['obs'], 99, axis=0)     # 외삽 차단
    g = np.random.default_rng(seed)
    cand = np.array([g.choice(45, 6, replace=False) + 1 for _ in range(pool)])
    F = np.array([[extract(c)[k] for k in KEEP] for c in cand], float)
    keep = ((F >= lo) & (F <= hi)).all(1)
    cand = cand[keep]
    mult = multiplier(m, cand)
    mult = mult / multiplier(m, cand).mean()
    seen, out = set(), []
    for i in np.argsort(mult):
        c = tuple(sorted(cand[i].tolist()))
        if c in seen:
            continue
        seen.add(c); out.append((list(c), float(mult[i])))
        if len(out) >= n_out:
            break
    return out, m

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=10)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recs, m = recommend(os.path.join(root, 'data/all.json'), a.n)
    print(f"학습 {m['n']}회차 (262~) | 자동 비율 {m['alpha']*100:.1f}%")
    print("계수(수동 구매자 선호):")
    for k, b, s, p in zip(KEEP, m['beta'], m['se'], m['p']):
        print(f"  {k:>8} {b:+.4f} ± {1.96*s:.4f}  p={p:.1e}")
    print(f"\n추천 조합 (공동당첨 기대 낮은 순)")
    for i, (c, mu) in enumerate(recs, 1):
        print(f"  {i:>2}. {' '.join(f'{x:>2}' for x in c)}   배율 {mu:.3f}")
    print(f"""
[검증]  표본 밖(901~1247회, 347회차) Spearman rho=+0.166, 순열검정 p=0.0028
        시대별 계수 안정 (옛날 vs 최근 차이 p=0.87/0.99/0.40)
[기댓값] 환급률 50.3% -> 53.5% (+6.5%)
[한계]  당첨 확률은 어느 조합이나 1/8,145,060으로 같다.
        1게임 기댓값은 약 535원. 1,000원을 내는 한 기댓값은 음수다.""")
