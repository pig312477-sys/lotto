"""추천 조합 생성. 인기도 점수가 낮은(= 공동당첨자가 적을) 조합을 고른다."""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from features import extract, FEATURES
from popularity import load, fit

def build(data_path, train_upto, n_out=10, pool=400000, seed=0, guard=(1, 99)):
    rows_all = load(data_path)
    m = fit(load(data_path, upto=train_upto))
    beta = m['params'][1:1 + len(FEATURES)]          # 절편·시간 제외 → 순위 전용 점수

    obs = np.array([[r[k] for k in FEATURES] for r in rows_all], float)
    lo_b = np.percentile(obs, guard[0], axis=0)
    hi_b = np.percentile(obs, guard[1], axis=0)      # 외삽 차단

    g = np.random.default_rng(seed)
    cand = np.array([g.choice(45, 6, replace=False) + 1 for _ in range(pool)])
    CF = np.array([[extract(c)[k] for k in FEATURES] for c in cand], float)
    keep = ((CF >= lo_b) & (CF <= hi_b)).all(1)
    cand, CF = cand[keep], CF[keep]
    score = ((CF - m['mu']) / m['sd']) @ beta

    order = np.argsort(score)
    out = [(sorted(cand[i].tolist()), float(score[i])) for i in order[:n_out]]
    return out, m, float(keep.mean())

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--train-upto', type=int, default=1247)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recs, m, kept = build(os.path.join(root, 'data/all.json'), a.train_upto, a.n)
    print(f"학습 {m['n']}회차 | 외삽 차단 후 후보 {kept*100:.1f}% 유지\n")
    print("추천 조합 (공동당첨 기대 낮은 순)")
    for i, (c, s) in enumerate(recs, 1):
        print(f"  {i:>2}. {' '.join(f'{n:>2}' for n in c)}   점수 {s:+.3f}")
    print("\n※ 당첨 확률은 어느 조합이나 1/8,145,060으로 같다.")
    print("※ 이 추천은 당첨됐을 때 나눠 가질 사람 수를 줄일 뿐이다.")
    print("※ 1게임 기댓값은 약 542원. 1,000원을 내고 사는 한 기댓값은 음수다.")
