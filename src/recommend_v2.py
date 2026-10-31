"""
[실패한 시도 — 사용 금지] 추천 v2 (시대 변화 반영 시도)

결론: 실패했다. 최근 시대만으로 학습하면 신호가 남지 않는다.
이 파일은 추천을 내놓지 않는다. 왜 안 되는지를 보여주는 기록으로만 남긴다.
다시 열지 말 것 — 같은 방법으로는 안 된다. 사유는 progress.md 참조.

원래 의도였던 것:

v1 문제: 2002년 이후 전 구간으로 학습했으나, 자동구매 비율 상승으로
사람의 번호 편향이 희석되어 옛날 계수가 지금은 맞지 않는다.

v2 방침:
  - 최근 시대(701회~)만 학습한다.
  - 재료를 `rng`(번호 분포 폭) 하나로 줄인다. 옛날·최근 계수 차이 검정에서
    유일하게 변하지 않은 재료다 (p=0.299). le12·row_max는 유의하게 변했다
    (p=0.007 / p=0.034) -> 폐기.
  - 효과 크기는 작고 불확실하다. 과장하지 않는다.
"""
import sys, os, json, argparse
import numpy as np
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from features import extract, FEATURES
from popularity import load

RECENT_FROM = 701
FEATURE = 'rng'

def fit_recent(path, since=RECENT_FROM):
    rs = [r for r in load(path) if r['t'] >= since]
    v = np.array([r[FEATURE] for r in rs], float)
    y = np.array([r['w'] for r in rs], float)
    off = np.log([r['exp'] for r in rs])
    t = np.array([r['t'] for r in rs], float)
    z = (v - v.mean()) / v.std()
    tz = (t - t.mean()) / t.std()
    m = sm.GLM(y, sm.add_constant(np.column_stack([z, tz])),
               family=sm.families.NegativeBinomial(alpha=0.0835), offset=off).fit()
    return dict(coef=m.params[1], se=m.bse[1], p=m.pvalues[1],
                mu=v.mean(), sd=v.std(), n=len(rs))

def recommend(path, n_out=10, pool=400000, seed=0):
    rs = load(path)
    mdl = fit_recent(path)
    obs = np.array([r[FEATURE] for r in rs], float)
    lo, hi = np.percentile(obs, 1), np.percentile(obs, 99)   # 외삽 차단

    g = np.random.default_rng(seed)
    cand = np.array([g.choice(45, 6, replace=False) + 1 for _ in range(pool)])
    val = np.array([extract(c)[FEATURE] for c in cand], float)
    keep = (val >= lo) & (val <= hi)
    cand, val = cand[keep], val[keep]
    score = ((val - mdl['mu']) / mdl['sd']) * mdl['coef']    # 낮을수록 공동당첨 적음

    seen, out = set(), []
    for i in np.argsort(score):                              # 중복 제거
        c = tuple(sorted(cand[i].tolist()))
        if c in seen:
            continue
        seen.add(c); out.append((list(c), float(score[i])))
        if len(out) >= n_out:
            break
    return out, mdl

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = fit_recent(os.path.join(root, 'data/all.json'))
    print(f"[실패 기록] 최근 시대({RECENT_FROM}회~, n={m['n']}) 단독 재료 모델")
    print(f"  {FEATURE} 계수 {m['coef']:+.4f} +- {1.96*m['se']:.4f} (95% CI), p={m['p']:.3f}")
    print(f"""
  -> 계수가 사실상 0이고 부호도 반대다. 신호가 없다.
     다변량(재료 10종 통제) 모델에서는 rng=-0.045(p=0.044)였으나,
     단독으로 두면 사라진다. 즉 그 값은 다른 재료를 통제했을 때만
     나타나는 것으로, 단독 재료 전략의 근거가 되지 못한다.
     **이 모델로는 추천을 생성하지 않는다.**
[효과에 대한 정직한 설명]
  이 추천의 근거는 재료 하나뿐이고, 그 효과도 작다 (95% CI가 0에 거의 닿는다).
  최근 시대 데이터로는 '효과가 있다'와 '효과가 없다'를 구별할 검정력이 부족하다.
  재료 10종을 FDR 보정하면 최근 데이터에서 살아남는 재료는 하나도 없다.
  확실히 하려면 약 300회차(약 6년)의 추첨이 더 필요하다.

[변하지 않는 사실]
  당첨 확률은 어느 조합이나 1/8,145,060으로 같다.
  1게임 기댓값은 약 500원대다. 1,000원을 내는 한 기댓값은 음수다.
  이 프로그램은 로또를 이기게 하지 않는다.""")
