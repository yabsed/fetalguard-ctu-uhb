"""계획서 5.1절 그림 — 5분 세그먼트 하나에서 인자가 계산되는 과정의 예시.

실행: cd 3-poc-factor-extraction && python make_figure.py (extract_factors_ctu.py 이후)
출력: figures/poc_factor_anatomy.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_factors_ctu import (  # noqa: E402
    DATA, FS, SEG_N, extract_segment, read_signals, trim_edge_missing,
)
from features import (  # noqa: E402  — A.2 규칙 구현
    ACC_MIN_BPM, DEC_MIN_BPM, classify_deceleration, detect_accelerations,
    detect_contractions, detect_decelerations, estimate_baseline,
    preprocess_for_features,
)

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# dataviz 검증 팔레트
C_FHR, C_UC = "#e34948", "#2a78d6"
C_DECEL, C_ACCEL, C_BASE = "#eb6834", "#1baf7a", "#1c2430"
MUTED = "#5a6673"

try:
    fm.findfont("NanumGothic", fallback_to_default=False)
    plt.rcParams["font.family"] = "NanumGothic"
except ValueError:
    pass
plt.rcParams["axes.unicode_minus"] = False

RID, SEG = "1001", 10   # 후기감속이 있는 비정상 레코드(pH 7.14)의 한 세그먼트


KOR = {"early": "조기", "late": "후기", "prolonged": "장기", "severe": "중증"}


def main():
    fhr, uc, meta, _ = read_signals(DATA / f"{RID}.hea")
    fhr, uc = trim_edge_missing(fhr, uc)
    seg_f, seg_t = preprocess_for_features(fhr[SEG * SEG_N:(SEG + 1) * SEG_N],
                                           uc[SEG * SEG_N:(SEG + 1) * SEG_N], FS)
    t = np.arange(SEG_N) / FS / 60
    f = extract_segment(seg_f, seg_t)

    base, _ = estimate_baseline(seg_f, FS)
    decs = detect_decelerations(seg_f, base, FS)
    accs = detect_accelerations(seg_f, base, FS)
    contractions = detect_contractions(seg_t, FS)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, seg_f, lw=1.0, color=C_FHR, label="FHR")
    ax.axhline(base, color=C_BASE, lw=1.8)
    ax.fill_between(t, base - DEC_MIN_BPM, base + ACC_MIN_BPM, color=C_BASE, alpha=0.08)
    ax.annotate(f"반복 기저선 {base:.1f} bpm (±15 bpm 판정 띠)",
                xy=(0.01, base + ACC_MIN_BPM), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=9.5, color=C_BASE)
    for s, e in decs:
        ax.axvspan(t[s], t[e - 1], color=C_DECEL, alpha=0.25)
        nadir = s + int(np.argmin(seg_f[s:e]))
        main_type, is_var = classify_deceleration(s, e, nadir, contractions, FS)
        tag = KOR[main_type] + ("+변이" if is_var else "")
        ax.annotate(tag, xy=(t[(s + e) // 2], seg_f[nadir] - 3), ha="center",
                    va="top", fontsize=8.5, color=C_DECEL)
    for s, e in accs:
        ax.axvspan(t[s], t[e - 1], color=C_ACCEL, alpha=0.25)
    ax.set_ylabel("FHR (bpm)")
    ax.set_title(
        f"레코드 {RID}(제대동맥혈 pH {meta['pH']:.2f}) — 5분 세그먼트 #{SEG}의 인자 값: "
        f"감속 {f['n_decel']:.0f}회(조기 {f['n_early_decel']:.0f}·후기 {f['n_late_decel']:.0f}, "
        f"그중 변이 {f['n_variable_decel']:.0f}), 가속 {f['n_accel']:.0f}회, "
        f"최대 하강 {f['decel_max_depth']:.0f} bpm, 기저선변이도 {f['figo_baseline_var']:.2f}",
        fontsize=10.5)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.plot(t, seg_t, lw=1.0, color=C_UC, label="UC")
    for on, p, end in contractions:
        ax2.axvspan(t[on], t[min(end, SEG_N - 1)], color=C_UC, alpha=0.12, lw=0)
        ax2.axvline(t[p], color=MUTED, ls="--", lw=1)
        ax.axvline(t[p], color=MUTED, ls="--", lw=1, alpha=0.6)
    ax2.set_ylabel("UC (상대 단위)")
    ax2.set_xlabel("세그먼트 내 시간 (분)")
    ax2.set_title(f"자궁수축 {f['n_contractions']:.0f}회 — 총 지속 45~120초의 종형(파랑 띠)만 계수, "
                  f"정점은 점선. 감속 개시가 수축 개시 +20초~종료 전이면 후기 · "
                  f"양신호 최소 상관 {f['ft_corr_min']:.2f} (lag {f['ft_corr_min_lag_s']:+.0f}초)",
                  fontsize=10)
    ax2.grid(alpha=0.25)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)

    fig.suptitle("공개 데이터(CTU-UHB) 5분 세그먼트에서의 인자 추출 예시 — "
                 "감속 유형은 Chiou et al. Appendix A.2 규칙", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "poc_factor_anatomy.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("저장:", FIG / "poc_factor_anatomy.png")


if __name__ == "__main__":
    main()
