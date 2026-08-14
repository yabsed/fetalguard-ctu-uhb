"""552개 레코드 각각의 Cat28 변환 결과를 한 장씩 시각화한다.

레코드 하나당 PNG 한 장 (figures/records/<record_id>.png):
  위 패널  — FHR 전체 곡선(결측은 끊김) 위에, 5분 세그먼트별 FIGO 반복 기저선(수평선),
             검출된 감속(주황 음영)·가속(초록 음영), 제외 세그먼트(결측 >30%, 회색 음영)
  아래 패널 — UC 곡선과 세그먼트별 수축 정점(점선)

즉 extract_factors_ctu.py가 CSV의 숫자로 만든 판정 근거를 레코드 전체 위에 펼쳐
육안으로 검증할 수 있게 한다. 세그먼트 분할·결측 기준·이벤트 판정 로직은 전부
extract_factors_ctu.py에서 그대로 가져온다(별도 재구현 없음).

실행: cd 3-poc-factor-extraction && python make_record_figures.py   (약 5~10분)
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
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_factors_ctu import (  # noqa: E402
    DATA, FS, MAX_MISSING, SEG_N, extract_segment, list_record_ids,
    read_signals, trim_edge_missing,
)
from features import (  # noqa: E402  — A.2 규칙 구현
    ACC_MIN_BPM, DEC_MIN_BPM, detect_accelerations, detect_contractions,
    detect_decelerations, estimate_baseline, preprocess_for_features,
)

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


def render_record(rid: str) -> dict:
    """레코드 한 건을 그려 저장하고 세그먼트 판정 요약을 반환한다."""
    fhr, uc, meta, _ = read_signals(DATA / f"{rid}.hea")
    fhr, uc = trim_edge_missing(fhr, uc)
    n_seg = len(fhr) // SEG_N
    t = np.arange(len(fhr)) / FS / 60          # 분 단위

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax_f, ax_u = axes
    ax_f.plot(t, np.where(fhr == 0, np.nan, fhr), lw=0.55, color=C_FHR)
    ax_u.plot(t, np.where(uc == 0, np.nan, uc), lw=0.55, color=C_UC)

    used = dropped = 0
    tot = {"n_decel": 0, "n_accel": 0, "n_contractions": 0}
    for k in range(n_seg):
        s0, s1 = k * SEG_N, (k + 1) * SEG_N
        t0, t1 = t[s0], t[s1 - 1]
        seg_f = fhr[s0:s1]
        if float((seg_f == 0).mean()) > MAX_MISSING:
            dropped += 1
            for ax in axes:
                ax.axvspan(t0, t1, color=C_DROP, alpha=0.15, lw=0)
            continue
        used += 1
        seg_f, seg_t = preprocess_for_features(seg_f, uc[s0:s1], FS)
        f = extract_segment(seg_f, seg_t)
        for key in tot:
            tot[key] += int(f[key])

        base, _keep = estimate_baseline(seg_f, FS)
        ax_f.hlines(base, t0, t1, color=C_BASE, lw=1.4)
        for s, e in detect_decelerations(seg_f, base, FS):
            ax_f.axvspan(t[s0 + s], t[s0 + e - 1], color=C_DECEL, alpha=0.30, lw=0)
        for s, e in detect_accelerations(seg_f, base, FS):
            ax_f.axvspan(t[s0 + s], t[s0 + e - 1], color=C_ACCEL, alpha=0.30, lw=0)
        for on, p, end in detect_contractions(seg_t, FS):
            ax_u.axvspan(t[s0 + on], t[s0 + min(end, SEG_N - 1)],
                         color=C_UC, alpha=0.12, lw=0)
            ax_u.axvline(t[s0 + p], color=MUTED, ls="--", lw=0.8, alpha=0.8)

    for k in range(1, n_seg):                   # 세그먼트 경계
        for ax in axes:
            ax.axvline(t[k * SEG_N], color=MUTED, ls=":", lw=0.6, alpha=0.5)

    ax_f.set_ylim(40, 220)
    ax_f.set_ylabel("FHR (bpm)")
    ax_u.set_ylabel("UC (상대 단위)")
    ax_u.set_xlabel("기록 시작 후 시간 (분)")
    ax_f.set_title(
        f"레코드 {rid} — pH {meta['pH']:.2f} · Apgar1 {meta['Apgar1']:.0f} · "
        f"5분 세그먼트 {used}개 사용 / {dropped}개 제외(결측 >30%, 회색) · "
        f"감속 {tot['n_decel']}회(주황) · 가속 {tot['n_accel']}회(초록) · "
        f"수축 정점 {tot['n_contractions']}회(아래 점선) · 검정 수평선 = 세그먼트별 FIGO 기저선",
        fontsize=9.5)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{rid}.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"record_id": rid, "used": used, "dropped": dropped, **tot}


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
