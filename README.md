# lotto — 당첨금 기댓값 최적화

로또 6/45에서 **번호를 맞히려는 시도가 아니다.** 매 회차 독립 균등추첨이므로 그건 불가능하다.

1등 당첨금은 상금풀을 당첨자 수로 나누는 **파리뮤추얼** 방식이다.
사람의 번호 선택에는 체계적 편향(생일·패턴)이 있으므로, 덜 선택되는 조합을 고르면
**당첨 확률은 그대로인 채 당첨 시 금액이 커진다.**

## 결과

- 검증된 효과: 공동당첨자 **24% 감소** (표본 밖, 순열검정 p=0.011)
- 환급률 **50.3% → 54.2%** (기댓값 +7.8%)

> **기댓값은 여전히 음수다.** 1,000원을 내고 542원을 기대하는 게임이다.
> 이 도구는 로또를 이기게 하지 않는다. 손실 폭을 줄일 뿐이다.

## 사용법

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy statsmodels
curl -o data/all.json https://raw.githubusercontent.com/smok95/lotto/master/results/all.json
.venv/bin/python src/recommend.py --n 10
```

## 구조

| 경로 | 설명 |
|---|---|
| `CLAUDE.md` | 공통 틀 — 작업 규칙 |
| `domain.md` | 0단계 도메인 정의 (7칸 + 적합성 검사) |
| `progress.md` | 진행 기록. **세션 시작 시 먼저 읽을 것** |
| `src/features.py` | 조합 특징 10종 |
| `src/popularity.py` | 음이항 인기도 모델 |
| `src/recommend.py` | 추천 조합 생성 (외삽 차단 포함) |
| `verify/score.py` | **봉인 채점기 — 수정 금지.** 1048~1247회, 최종 1회만 |

## 데이터

[smok95/lotto](https://github.com/smok95/lotto) — 1~1247회, 등위별 당첨자 수 포함.
동행복권 직접 접속은 현재 환경 네트워크 정책에 차단됨.
