"""인기도 모델: 조합 -> 예상 공동당첨 배율(1.0 = 균등 가정)."""
import json, math, sys, os
import numpy as np, statsmodels.api as sm
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(__file__))
from features import extract, FEATURES, C45_6

TICKET = 1000

def load(path, upto=None):
    rows = []
    for x in json.load(open(path)):
        if upto and x['draw_no'] > upto:
            continue
        d1 = x['divisions'][0]
        s = x.get('total_sales_amount') or 0
        if s <= 0:
            continue
        f = extract(x['numbers'])
        f['w'] = d1.get('winners', 0) if d1 else 0
        f['exp'] = (s / TICKET) / C45_6
        f['t'] = x['draw_no']
        rows.append(f)
    return rows

def fit(rows):
    X = np.array([[r[k] for k in FEATURES] for r in rows], float)
    y = np.array([r['w'] for r in rows], float)
    off = np.log([r['exp'] for r in rows])
    t = np.array([r['t'] for r in rows], float)
    mu, sd = X.mean(0), X.std(0)
    tm, ts = t.mean(), t.std()
    Xd = sm.add_constant(np.column_stack([(X - mu) / sd, (t - tm) / ts]))
    f = lambda a: -sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=a),
                          offset=off).fit().llf
    a = minimize_scalar(f, bounds=(1e-4, 2), method='bounded').x
    m = sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=a), offset=off).fit()
    return dict(params=m.params, mu=mu, sd=sd, tm=tm, ts=ts, alpha=a,
                pvalues=m.pvalues, n=len(rows))

def multiplier(model, combos, t_eval):
    """combos: (N,6) 정수배열 -> 배율 (낮을수록 공동당첨자 적음)"""
    F = np.array([[extract(c)[k] for k in FEATURES] for c in combos], float)
    Z = (F - model['mu']) / model['sd']
    tz = (t_eval - model['tm']) / model['ts']
    Xd = np.column_stack([np.ones(len(Z)), Z, np.full(len(Z), tz)])
    return np.exp(Xd @ model['params'])
