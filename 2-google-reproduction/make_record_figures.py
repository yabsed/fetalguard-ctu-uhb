"""552개 레코드 각각에 대해 Chiou et al. (2025) 재구현 파이프라인의 판정 근거를 한 장씩 그린다.

레코드 하나당 PNG 한 장 (figures/records/<record_id>.png) — 3단:
  1단 — 원신호 FHR 전체. 결측(FHR=0)은 끊어서 표시하고, 1단계에서 잘려나가는 양끝 결측은
         회색, >15초라 0으로 유지되는 결측 구간은 주황 음영으로 표시한다.
  2단 — 규칙 기반 특성 파이프라인이 실제로 보는 구간(마지막 30분, 전처리 완료) 위에
         FIGO 반복 기저선(검정), 감속(주황)·가속(초록), UC 수축 정점(점선)을 겹친다.
  3단 — 신경망 파이프라인의 최종 산출물인 모델 입력 (2 채널 × 1,800포인트, 1 Hz, max-abs).

즉 run_all.py가 CSV·NPZ의 숫자로 만든 결과를 레코드마다 육안으로 검증할 수 있게 한다.
전처리·기저선·이벤트 판정 로직은 전부 preprocess.py / features.py에서 그대로 가져온다
(별도 재구현 없음).

실행: cd 2-google-reproduction && python make_record_figures.py   (약 3~5분)
출력: figures/records/*.png (552장) — 진행 상황은 50건 단위로 출력
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ctguhb import list_record_ids, read_signals  # noqa: E402
from features import (  # noqa: E402
    ACC_MIN_BPM, DEC_MIN_BPM, detect_accelerations, detect_contractions,
    detect_decelerations, estimate_baseline, preprocess_for_features,
)
from preprocess import (  # noqa: E402
    CROP_MIN, FS, LONG_GAP_S, crop_last_minutes, gap_runs, impute_short_gaps,
    make_model_input, preprocess_record, trim_edge_missing,
)
from run_all import DATA, MAX_CROP_MISSING, MAX_CROP_MISSING_TXT  # noqa: E402

OUT = HERE / "figures" / "records"
OUT.mkdir(parents=True, exist_ok=True)

C_FHR, C_UC = "#e34948", "#2a78d6"
C_DECEL, C_ACCEL, C_BASE = "#eb6834", "#1baf7a", "#1c2430"
C_DROP, MUTED = "#8a94a0", "#5a6673"

try:
    fm.findfont("NanumGothic", fallback_to_default=False)
    plt.rcParams["font.family"] = "NanumGothic"
except ValueError:
    pass
plt.rcParams["axes.unicode_minus"] = False


def trim_bounds(fhr: np.ndarray) -> tuple[int, int]:
    """trim_edge_missing이 잘라내는 경계 (start, end) — 같은 규칙."""
    present = fhr != 0
    if not present.any():
        return 0, len(fhr)
    return int(np.argmax(present)), len(present) - int(np.argmax(present[::-1]))


def render_record(rid: str) -> None:
    fhr_raw, uc_raw, meta, fs = read_signals(DATA / f"{rid}.hea")
    first, last = trim_bounds(fhr_raw)
    fhr_t, uc_t = trim_edge_missing(fhr_raw, uc_raw)

    # (A) 신경망 파이프라인 — 모델 입력
    fhr_p, uc_p = preprocess_record(fhr_raw, uc_raw)
    x = make_model_input(fhr_p, uc_p)

    # (B) 특성 파이프라인 — 마지막 30분과 그 위의 이벤트
    fhr_f, uc_f = preprocess_for_features(fhr_t, uc_t)
    fhr_c = crop_last_minutes(fhr_f, fs=FS)
    uc_c = crop_last_minutes(uc_f, fs=FS)

    crop_raw = crop_last_minutes(impute_short_gaps(fhr_t, int(LONG_GAP_S * FS)), fs=FS)
    crop_miss = float((crop_raw == 0).mean()) if crop_raw is not None else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.0))
    ax_raw, ax_evt, ax_in = axes

    # ---- 1단: 원신호와 결측 구조
    t = np.arange(len(fhr_raw)) / FS / 60
    ax_raw.plot(t, np.where(fhr_raw == 0, np.nan, fhr_raw), lw=0.5, color=C_FHR)
    if first > 0:
        ax_raw.axvspan(t[0], t[first], color=C_DROP, alpha=0.30, lw=0)
    if last < len(fhr_raw):
        ax_raw.axvspan(t[last - 1], t[-1], color=C_DROP, alpha=0.30, lw=0)
    n_long = 0
    for s, e in gap_runs(fhr_t == 0):
        if e - s > LONG_GAP_S * FS:
            n_long += 1
            ax_raw.axvspan(t[first + s], t[first + e - 1], color=C_DECEL, alpha=0.22, lw=0)
    ax_raw.set_ylim(40, 220)
    ax_raw.set_ylabel("FHR (bpm)")
    ax_raw.set_xlabel("기록 시작 후 시간 (분)")
    ax_raw.set_title(
        f"1) 원신호 {len(fhr_raw) / FS / 60:.0f}분 — 양끝 결측 제거 구간(회색) · "
        f">{LONG_GAP_S}초라 0으로 유지되는 결측 {n_long}구간(주황) · "
        f"나머지 결측은 선형 보간", fontsize=9.5)

    # ---- 2단: 마지막 30분 + 기저선·이벤트
    tc = np.arange(len(fhr_c)) / FS / 60
    base, _keep = estimate_baseline(fhr_c, FS)
    decs = detect_decelerations(fhr_c, base, FS)
    accs = detect_accelerations(fhr_c, base, FS)
    contractions = detect_contractions(uc_c, FS)   # A.2: 종형 상승, 총 지속 45~120초
    ax_evt.plot(tc, fhr_c, lw=0.8, color=C_FHR)
    ax_evt.axhline(base, color=C_BASE, lw=1.6)
    ax_evt.fill_between(tc, base - DEC_MIN_BPM, base + ACC_MIN_BPM,
                        color=C_BASE, alpha=0.07)
    for s, e in decs:
        ax_evt.axvspan(tc[s], tc[e - 1], color=C_DECEL, alpha=0.28, lw=0)
    for s, e in accs:
        ax_evt.axvspan(tc[s], tc[e - 1], color=C_ACCEL, alpha=0.28, lw=0)
    for on, p, end in contractions:
        ax_evt.axvspan(tc[on], tc[min(end, len(tc) - 1)], color=C_UC, alpha=0.10, lw=0)
        ax_evt.axvline(tc[p], color=MUTED, ls="--", lw=0.7, alpha=0.75)
    ax_evt.set_ylim(40, 220)
    ax_evt.set_ylabel("FHR (bpm)")
    ax_evt.set_xlabel(f"마지막 {CROP_MIN}분 내 시간 (분)")
    ax_evt.set_title(
        f"2) 특성 파이프라인 — 기저선 {base:.1f} bpm(검정) · 감속 {len(decs)}회(주황) · "
        f"가속 {len(accs)}회(초록) · 수축 {len(contractions)}회(파랑 띠, 정점 점선)",
        fontsize=9.5)

    # ---- 3단: 모델 입력
    if x is not None:
        t1 = np.arange(x.shape[1]) / 60
        ax_in.plot(t1, x[0], lw=0.8, color=C_FHR, label="FHR")
        ax_in.plot(t1, x[1], lw=0.8, color=C_UC, label="UC")
        ax_in.legend(loc="lower left", frameon=False, fontsize=8)
        ax_in.set_ylabel("스케일값")
        ax_in.set_xlabel(f"마지막 {CROP_MIN}분 내 시간 (분)")
        ax_in.set_title(
            f"3) 신경망 입력 — 1 Hz 다운샘플 {x.shape[1]:,}포인트 × 2채널, "
            "채널별 max-abs 스케일", fontsize=9.5)
    else:
        ax_in.text(0.5, 0.5, f"{CROP_MIN}분 크롭 불가", ha="center", va="center",
                   transform=ax_in.transAxes)

    usable = "분석 대상" if crop_miss <= MAX_CROP_MISSING else "제외 (결측 과다)"
    fig.suptitle(
        f"레코드 {rid} — pH {meta['pH']:.2f} · Apgar1 {meta['Apgar1']:.0f} · "
        f"마지막 {CROP_MIN}분 결측 {crop_miss:.1%} → {usable} "
        f"(기준 ≤{MAX_CROP_MISSING_TXT})", y=1.0, fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{rid}.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    ids = list_record_ids(DATA)
    for i, rid in enumerate(ids, 1):
        render_record(rid)
        if i % 50 == 0 or i == len(ids):
            print(f"{i}/{len(ids)} ({time.time() - t0:.0f}초)", flush=True)
    print(f"완료: {OUT} ({len(ids)}장, {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
