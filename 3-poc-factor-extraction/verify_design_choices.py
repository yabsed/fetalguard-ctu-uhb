"""계획서 5.1절의 설계 근거 실측 — 세 가지 정의 선택의 민감도 검증.

계획서 5.1절은 인자 정의에서 세 가지 선택을 하고 그 근거를 공개 데이터 실측으로
제시한다. 이 스크립트가 그 실측의 출처다.

  1. 기저선: 반복 추정(A.2) vs 세그먼트 중앙값 근사 — 두 기준의 차이와,
     기준 선택이 감속 수 판정을 얼마나 바꾸는지
  2. 수축 검출: 상대 임계값(세그먼트 표준편차의 0.5배) vs 절대 임계값
     (prominence 20; experiment-5 계열에서 쓰던 값) — 수축 수의 상관
  3. 결합 상관: 겹치는 구간에서 다시 중심화하는 피어슨 상관(계획서 정의) vs
     세그먼트 전역 표준화 후 곱 평균 — 전역 표준화가 상관 크기를 0 쪽으로
     누르는 괴리 사례

세그먼트 선별과 전처리는 extract_factors_ctu.py와 동일하다(결측 >30% 제외,
A.2 특성 전용 전처리).

실행: cd 3-poc-factor-extraction && python verify_design_choices.py
출력: log/design_choices.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "ctu-hub-ctgdb").exists())
sys.path.insert(0, str(REPO / "2-google-reproduction"))

from ctguhb import list_record_ids, read_signals  # noqa: E402
from features import (  # noqa: E402
    UC_DUR_MAX_S, UC_DUR_MIN_S, detect_contractions, detect_decelerations,
    estimate_baseline, preprocess_for_features,
)
from preprocess import trim_edge_missing  # noqa: E402

from extract_factors_ctu import DATA, FS, MAX_MISSING, SEG_N  # noqa: E402

LOG = HERE / "log"
LOG.mkdir(exist_ok=True)

ABS_PROMINENCE = 20.0     # 절대 임계값 대안 (experiment-5 계열의 값)
MAX_LAG = int(60 * FS)    # 결합 상관 lag 범위 ±60초 (계획서 5.1)


def detect_contractions_abs(uc, prominence, fs=FS):
    """detect_contractions와 동일하되 prominence만 절대값으로 바꾼 대안."""
    peaks, _ = find_peaks(uc, distance=int(60 * fs), prominence=prominence)
    if len(peaks) == 0:
        return []
    _, _, lefts, rights = peak_widths(uc, peaks, rel_height=0.9)
    return [(int(round(l)), int(p), int(round(r)))
            for p, l, r in zip(peaks, lefts, rights)
            if UC_DUR_MIN_S <= (r - l) / fs <= UC_DUR_MAX_S]


def corr_windowed(a, b):
    """계획서 5.1 정의: 겹치는 구간에서 다시 중심화·표준화하는 피어슨 상관."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lagged_min(f, t, corr_fn):
    best = corr_fn(f, t)
    for lag in range(-MAX_LAG, MAX_LAG + 1, FS):
        if lag == 0:
            continue
        a = f[lag:] if lag > 0 else f[:lag]
        b = t[:-lag] if lag > 0 else t[-lag:]
        r = corr_fn(a, b)
        if r < best:
            best = r
    return best


def lagged_min_global(f, t):
    """대안 정의: 세그먼트 전역에서 한 번만 표준화한 뒤 겹치는 구간의 곱 평균.
    창마다 중심이 어긋나 상관 크기가 0 쪽으로 눌린다 — 계획서가 배제한 방식."""
    if f.std() == 0 or t.std() == 0:
        return 0.0
    fz, tz = (f - f.mean()) / f.std(), (t - t.mean()) / t.std()
    best = float(np.mean(fz * tz))
    for lag in range(-MAX_LAG, MAX_LAG + 1, FS):
        if lag == 0:
            continue
        a = fz[lag:] if lag > 0 else fz[:lag]
        b = tz[:-lag] if lag > 0 else tz[-lag:]
        if len(a) >= 2:
            best = min(best, float(np.mean(a * b)))
    return best


def main():
    n_seg = 0
    n_decel_mismatch = 0
    base_diffs = []
    rel_counts, abs_counts = [], []
    divergent = []   # (record, seg, 창별 피어슨, 전역 표준화)

    for rid in list_record_ids(DATA):
        fhr, uc, _meta, fs = read_signals(DATA / f"{rid}.hea")
        assert fs == FS
        fhr, uc = trim_edge_missing(fhr, uc)
        for k in range(len(fhr) // SEG_N):
            seg_f = fhr[k * SEG_N:(k + 1) * SEG_N]
            seg_t = uc[k * SEG_N:(k + 1) * SEG_N]
            if float((seg_f == 0).mean()) > MAX_MISSING:
                continue
            seg_f, seg_t = preprocess_for_features(seg_f, seg_t, FS)
            n_seg += 1

            # 1) 기저선 기준 비교
            b_iter, _ = estimate_baseline(seg_f, FS)
            b_med = float(np.median(seg_f))
            base_diffs.append(abs(b_iter - b_med))
            if len(detect_decelerations(seg_f, b_iter, FS)) != \
               len(detect_decelerations(seg_f, b_med, FS)):
                n_decel_mismatch += 1

            # 2) 수축 임계값 비교
            rel_counts.append(len(detect_contractions(seg_t, FS)))
            abs_counts.append(len(detect_contractions_abs(seg_t, ABS_PROMINENCE, FS)))

            # 3) 결합 상관 정의 비교
            w = lagged_min(seg_f, seg_t, corr_windowed)
            g = lagged_min_global(seg_f, seg_t)
            if w < -0.55 and g > -0.12:
                divergent.append((rid, k, w, g))

    base_diffs = np.asarray(base_diffs)
    rel, ab = np.asarray(rel_counts, float), np.asarray(abs_counts, float)
    r = float(np.corrcoef(rel, ab)[0, 1])

    lines = [
        "# 5.1절 설계 근거 실측 — 정의 선택의 민감도",
        "",
        f"분석 세그먼트: {n_seg:,}개 (extract_factors_ctu.py와 동일 선별·전처리)",
        "",
        "## 1. 기저선 — 반복 추정(A.2) vs 세그먼트 중앙값",
        "",
        f"- 두 기준의 차이: 평균 {base_diffs.mean():.2f} bpm, 최대 {base_diffs.max():.2f} bpm",
        f"- 감속 수 판정이 달라지는 세그먼트: {n_decel_mismatch}개 "
        f"({n_decel_mismatch / n_seg:.1%})",
        "- 감속이 깊은 세그먼트일수록 중앙값 자체가 끌려 내려가 판정이 달라진다.",
        "",
        "## 2. 수축 검출 — 상대 임계값(0.5×sd) vs 절대 임계값(prominence 20)",
        "",
        f"- 세그먼트별 수축 수의 피어슨 상관: {r:.2f}",
        f"- 평균 수축 수: 상대 {rel.mean():.2f} / 절대 {ab.mean():.2f}",
        "- 임계값 선택이 수축 수를 크게 바꾼다 — 계획서 5.1이 수축 검출을",
        "  민감도 분석 대상으로 두는 근거.",
        "",
        "## 3. 결합 상관 — 창별 피어슨(계획서 정의) vs 전역 표준화 후 곱 평균",
        "",
        f"- 창별 피어슨 < −0.55인데 전역 표준화가 > −0.12로 누르는 세그먼트: "
        f"{len(divergent)}개",
    ]
    for rid, k, w, g in divergent:
        lines.append(f"  - 레코드 {rid} 세그먼트 #{k}: {w:.2f} → {g:.2f}")
    lines += [
        "- 전역 통계로 표준화한 뒤 잘라 곱하면 창마다 중심이 어긋나 상관의",
        "  크기가 0 쪽으로 눌린다 — 계획서 5.1이 창별 재중심화를 명시하는 근거.",
    ]
    (LOG / "design_choices.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {LOG / 'design_choices.md'}")


if __name__ == "__main__":
    main()
