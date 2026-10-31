"""조합 특징 추출. 인기도(함께 고른 사람 수)를 설명하는 재료."""
import math

C45_6 = math.comb(45, 6)

# 동행복권 용지 배열: 7열 x 7행 (1~45, 마지막 행은 43~45)
def _cell(n):
    return ((n - 1) // 7, (n - 1) % 7)  # (row, col)

def extract(nums):
    ns = sorted(nums)
    diffs = [b - a for a, b in zip(ns, ns[1:])]
    rows = [_cell(n)[0] for n in ns]
    cols = [_cell(n)[1] for n in ns]
    last = [n % 10 for n in ns]
    return {
        "le31":    sum(1 for n in ns if n <= 31),   # 생일(일) 편향
        "le12":    sum(1 for n in ns if n <= 12),   # 생월 편향
        "consec":  sum(1 for d in diffs if d == 1), # 연속쌍
        "rng":     ns[-1] - ns[0],                  # 분포 폭
        "sm":      sum(ns),                         # 합
        "dup_last": 6 - len(set(last)),             # 끝자리 중복
        "row_max": max(rows.count(r) for r in set(rows)),  # 같은 행 몰림
        "col_max": max(cols.count(c) for c in set(cols)),  # 같은 열 몰림
        "mult5":   sum(1 for n in ns if n % 5 == 0),
        "odd":     sum(1 for n in ns if n % 2 == 1),
    }

FEATURES = list(extract([1, 2, 3, 4, 5, 6]).keys())
