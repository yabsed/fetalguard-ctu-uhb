"""계획서 5.1절 Proof of Concept — 공개 데이터(CTU-UHB)에서 Cat28 파형 인자 추출.

본선에서 [건양대학교의료원] 태아 심박동 모니터링 데이터의 5분 세그먼트에 적용할 인자
추출 파이프라인을, 접근 가능한 공개 데이터인 CTU-UHB Intrapartum CTG DB(PhysioNet,
552건, 4 Hz)에서 같은 5분 창 단위로 시연한다. 목적은 세 가지다.

  1. 인자 정의가 코드 수준까지 구체화되어 있음을 보인다 (계획서 5.1절의 근거)
  2. 전체 552건 처리에 필요한 계산 자원이 미미함을 실측한다 (계획서 5.4절의 근거)
  3. 인자들이 임상 라벨(제대동맥혈 pH)과 방향성 있는 관계를 가짐을 확인한다 (4.2절 가설의
     사전 점검 — 안심존의 판독 라벨과는 다른 라벨이므로 확증이 아니라 sanity check다)

Cat28 — 계획서 4.1절(6그룹)·5.1절(28행 계산식 표)의 인자 카탈로그 28개
  [신호 수준]   fhr_mean, fhr_min, fhr_max, fhr_sd
  [변이도]      stv*, figo_baseline_var
  [기저선]      figo_baseline (FIGO 반복 추정), hist_median, hist_mode, hist_width
  [범위 일탈]   brady_frac(<110), tachy_frac(>160)
  [감속·가속]   n_decel, decel_max_depth, decel_time_frac, n_accel,
                n_early_decel, n_late_decel, n_variable_decel, n_severe_decel,
                n_prolonged_decel
  [자궁수축]    toco_mean, toco_max, toco_sd, n_contractions
  [양신호 결합] ft_corr0, ft_corr_min, ft_corr_min_lag_s

  * stv는 인접 표본 간 절대차 평균으로, 임상 표준 STV(박동 간 변이)와 다르다.
    표본 간격(이 데이터 0.25초)에 의존하므로 본선 데이터(2초 간격)와 수치 비교는
    불가하며, 같은 데이터 내 상대 비교용이다.

**정의의 준거는 Chiou et al.의 medRxiv 프리프린트(doi:10.1101/2024.03.05.24303805)
Appendix A.2다.** 계획서 4.1이 "Chiou et al.의 FIGO 17개를 포괄한다"고 선언하므로,
겹치는 인자는 그들의 실제 계산 규칙을 그대로 따라야 한다. 구현은 이 저장소의
재현 코드(`../2-google-reproduction/features.py`)에서 직접 임포트해 두 실험이
같은 코드를 쓰도록 한다 — 정의가 갈라질 여지를 없앤다.

A.2가 규정하는 규칙 (전부 원문 명시):
  수축     종형 상승이며 총 지속 45~120초인 UC 피크
  기저선   10분 창의 **평균**으로 초기화, 가속·감속을 제외하며 변화량 < 0.5까지 반복
  변이도   기저선 요동을 이루는 소규모 피크·트로프 사이 진폭 변화의 평균
  가속     +15 bpm 초과, 15초 초과 지속, 개시→정점 30초 미만, 총 지속 10분 미만
  감속     −15 bpm 초과 하강, 15초 초과 지속
  후기     감속 **개시**가 최근접 수축 **개시** 20초 후 ~ 수축 종료 전
  조기/장기/중증  후기가 아닌 감속을 지속시간으로 나눈다 — <3분 / 3~5분 / >5분
                  (중증이 심박 수치가 아니라 지속시간 기준임에 주의)
  변이     개시→최저점 30초 미만 — 위 분류와 **독립**으로 계수

5분 창의 구조적 제약 (계획서 5.1·6.2에 명시):
  - 기저선 초기 창 10분 > 세그먼트 5분이라 창 전체를 쓴다.
  - 장기(3~5분)·중증(>5분)은 5분 세그먼트에서 사실상/원리상 관측되지 않는다.
    A.2의 유형 체계가 30분 창을 전제로 설계된 결과이며, 5분 창에서 살아남는 것은
    조기·후기·변이뿐이다.
  - 수축이 하나도 검출되지 않으면 A.2 규칙상 후기가 될 수 없으므로 n_late_decel은
    0이고, 나머지는 지속시간 분류로 결정된다 (결측이 아니라 0).

A.2가 다루지 않는 인자(신호 기술통계, 범위 일탈, 감속 깊이·시간 비율, 양신호 결합
상관)는 계획서 5.1의 자체 정의를 따른다.

실행: cd 3-poc-factor-extraction && python extract_factors_ctu.py
출력: log/ctu_segment_factors.csv (전체 세그먼트 × Cat28), log/summary.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "ctu-hub-ctgdb").exists())
sys.path.insert(0, str(REPO / "2-google-reproduction"))

from ctguhb import list_record_ids, read_signals  # noqa: E402
from features import (  # noqa: E402  — A.2 규칙 구현 (단일 출처)
    baseline_variability, classify_deceleration, detect_accelerations,
    detect_contractions, detect_decelerations, estimate_baseline,
    preprocess_for_features,
)
from preprocess import trim_edge_missing  # noqa: E402

DATA = REPO / "ctu-hub-ctgdb"
LOG = HERE / "log"
LOG.mkdir(exist_ok=True)

FS = 4                       # Hz
SEG_S = 300                  # 5분 세그먼트
SEG_N = SEG_S * FS
MAX_MISSING = 0.30           # 결측 30% 초과 세그먼트는 제외 [계획서 5.1 1단계]

BRADY, TACHY = 110.0, 160.0  # 서맥·빈맥 경계 [계획서 5.1 자체 정의]
MAX_LAG = int(60 * FS)       # 결합 상관 lag 범위 ±60초 [계획서 5.1 자체 정의]
HIST_BINS = 24

CAT28 = [
    "fhr_mean", "fhr_min", "fhr_max", "fhr_sd", "stv", "brady_frac", "tachy_frac",
    "n_decel", "decel_max_depth", "decel_time_frac", "n_accel",
    "toco_mean", "toco_max", "toco_sd", "n_contractions",
    "ft_corr0", "ft_corr_min", "ft_corr_min_lag_s",
    "figo_baseline", "figo_baseline_var",
    "n_early_decel", "n_late_decel", "n_variable_decel", "n_severe_decel",
    "n_prolonged_decel", "hist_width", "hist_median", "hist_mode",
]
FACTORS = CAT28   # 하위 호환 별칭


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """겹치는 구간의 피어슨 상관 — 계획서 5.1의 corr(·,·) 정의. 분산 0이면 0."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_segment(fhr: np.ndarray, toco: np.ndarray) -> dict:
    """정제된 5분 창(4 Hz, 결측 없음) → Cat28 인자 28개.

    FIGO 계열(기저선·변이도·가속·감속·유형·수축)은 A.2 구현(features.py)을 그대로
    호출하고, 그 밖의 인자는 계획서 5.1의 자체 정의로 계산한다.
    """
    f = {}
    f["fhr_mean"], f["fhr_min"] = float(fhr.mean()), float(fhr.min())
    f["fhr_max"], f["fhr_sd"] = float(fhr.max()), float(fhr.std())
    f["stv"] = float(np.abs(np.diff(fhr)).mean())
    f["brady_frac"] = float((fhr < BRADY).mean())
    f["tachy_frac"] = float((fhr > TACHY).mean())

    # --- A.2: 기저선(10분 창 평균 초기화 — 5분 세그먼트라 창 전체)과 피크-트로프 변이도
    base, keep = estimate_baseline(fhr, FS)
    f["figo_baseline"] = float(base)
    f["figo_baseline_var"] = baseline_variability(fhr, keep, FS)

    # --- A.2: 가속·감속 검출, 수축(45~120초 종형)
    accs = detect_accelerations(fhr, base, FS)
    decs = detect_decelerations(fhr, base, FS)
    contractions = detect_contractions(toco, FS)
    f["n_decel"], f["n_accel"] = len(decs), len(accs)
    f["decel_max_depth"] = float(max((base - fhr[s:e].min() for s, e in decs), default=0.0))
    f["decel_time_frac"] = float(sum(e - s for s, e in decs) / len(fhr))
    f["toco_mean"], f["toco_max"] = float(toco.mean()), float(toco.max())
    f["toco_sd"], f["n_contractions"] = float(toco.std()), len(contractions)

    # --- A.2: 유형 분류. 조기·후기·장기·중증은 D의 분할, 변이는 독립 계수
    counts = {"early": 0, "late": 0, "prolonged": 0, "severe": 0}
    n_var = 0
    for s, e in decs:
        nadir = s + int(np.argmin(fhr[s:e]))
        main, is_var = classify_deceleration(s, e, nadir, contractions, FS)
        counts[main] += 1
        n_var += is_var
    f.update({"n_early_decel": counts["early"], "n_late_decel": counts["late"],
              "n_variable_decel": n_var, "n_severe_decel": counts["severe"],
              "n_prolonged_decel": counts["prolonged"]})

    hist, edges = np.histogram(fhr, bins=HIST_BINS)
    k = int(np.argmax(hist))
    f["hist_width"] = float(fhr.max() - fhr.min())
    f["hist_median"] = float(np.median(fhr))
    f["hist_mode"] = float((edges[k] + edges[k + 1]) / 2)

    # --- 계획서 자체 정의: FHR-TOCO 시간차 상관 (±60초). A.2에는 없는 인자다.
    f["ft_corr0"] = _corr(fhr, toco)
    best_r, best_lag = f["ft_corr0"], 0
    for lag in range(-MAX_LAG, MAX_LAG + 1, FS):   # 1초 간격 탐색
        if lag == 0:
            continue
        a = fhr[lag:] if lag > 0 else fhr[:lag]
        b = toco[:-lag] if lag > 0 else toco[-lag:]
        r = _corr(a, b)
        if r < best_r:
            best_r, best_lag = r, lag
    f["ft_corr_min"], f["ft_corr_min_lag_s"] = best_r, best_lag / FS
    return f


def main():
    t0 = time.time()
    rows, n_dropped = [], 0
    ids = list_record_ids(DATA)
    for rid in ids:
        fhr, uc, meta, fs = read_signals(DATA / f"{rid}.hea")
        assert fs == FS
        fhr, uc = trim_edge_missing(fhr, uc)
        for k in range(len(fhr) // SEG_N):
            seg_f = fhr[k * SEG_N:(k + 1) * SEG_N]
            seg_t = uc[k * SEG_N:(k + 1) * SEG_N]
            missing = float((seg_f == 0).mean())
            if missing > MAX_MISSING:
                n_dropped += 1
                continue
            # A.2 특성 전용 전처리: 짧은 결측(≤15초) 보간 → 30초 롤링 평활 →
            # 남은 결측 보간. 수축의 '종형 45~120초' 판정이 평활에 의존하므로
            # 인자 계산도 같은 신호 위에서 해야 한다.
            seg_f, seg_t = preprocess_for_features(seg_f, seg_t, FS)
            rows.append({"record_id": rid, "seg_idx": k, "missing_frac": missing,
                         "label_ph": int(meta["pH"] < 7.20),
                         "label_apgar": int(meta["Apgar1"] < 7),
                         **extract_segment(seg_f, seg_t)})
    dt = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(LOG / "ctu_segment_factors.csv", index=False)

    lines = [
        "# PoC 실행 요약 — CTU-UHB 5분 세그먼트 Cat28 인자 추출",
        "",
        f"- 레코드 {len(ids)}건 → 5분 세그먼트 {len(df):,}개 "
        f"(결측 >{MAX_MISSING:.0%} 제외 {n_dropped}개)",
        f"- Cat28 인자 28개 × {len(df):,}세그먼트 계산 시간: **{dt:.1f}초** (단일 CPU 코어)",
        f"- 세그먼트당 평균 {dt / max(len(df), 1) * 1000:.1f} ms",
        "",
        "## 인자별 요약 (pH 라벨 기준 레코드 단위 평균의 방향성 점검)",
        "",
        "| 인자 | 정상(pH≥7.20) | 비정상(pH<7.20) |",
        "|---|---:|---:|",
    ]
    rec = df.groupby(["record_id", "label_ph"])[CAT28].mean().reset_index()
    g = rec.groupby("label_ph")[CAT28].mean()
    for c in CAT28:
        lines.append(f"| {c} | {g.loc[0, c]:.3f} | {g.loc[1, c]:.3f} |")
    (LOG / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"\n저장: {LOG}/ctu_segment_factors.csv, {LOG}/summary.md")


if __name__ == "__main__":
    main()
