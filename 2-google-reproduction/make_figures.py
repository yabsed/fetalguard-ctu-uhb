"""계획서 3.3절 그림 — Chiou et al. (2025) 전처리 재구현의 단계별 시각화.

  figures/11_preprocessing.png — 레코드 1001의 전처리 4단계 (계획서 그림 3-3-1)

실행: cd 2-google-reproduction && python make_figures.py
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from ctguhb import read_signals
from preprocess import crop_last_minutes, make_model_input, preprocess_record

C_FHR, C_UC = "#e34948", "#2a78d6"

try:
    fm.findfont("NanumGothic", fallback_to_default=False)
    plt.rcParams["font.family"] = "NanumGothic"
except ValueError:
    pass
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = next(p for p in HERE.resolve().parents
                if (p / "ctu-hub-ctgdb").exists()) / "ctu-hub-ctgdb"


def fig_preprocessing(rid="1001"):
    """원신호 → 정제 → 마지막 30분 크롭 → 1 Hz 모델 입력의 4단계."""
    fhr_raw, uc_raw, meta, fs = read_signals(DATA_DIR / f"{rid}.hea")
    fhr_p, uc_p = preprocess_record(fhr_raw, uc_raw)
    fhr_c = crop_last_minutes(fhr_p, fs=fs)
    x = make_model_input(fhr_p, uc_p)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10.5))

    t = np.arange(len(fhr_raw)) / fs / 60
    axes[0].plot(t, fhr_raw, lw=0.5, color=C_FHR)
    axes[0].set_title(f"1) 원신호 FHR (4 Hz, {t[-1]:.0f}분) — 0으로 떨어지는 구간이 결측", fontsize=11)
    axes[0].set_ylabel("bpm")

    tp = np.arange(len(fhr_p)) / fs / 60
    axes[1].plot(tp, fhr_p, lw=0.5, color=C_FHR)
    axes[1].set_title("2) 전처리 후 — 양끝 결측 제거, ≤15초 결측 선형 보간(>15초는 0 표지 유지), "
                      "유효 표본만 이동평균 평활", fontsize=11)
    axes[1].set_ylabel("bpm")

    t30 = np.arange(len(fhr_c)) / fs / 60
    axes[2].plot(t30, fhr_c, lw=0.6, color=C_FHR)
    axes[2].set_title("3) 분만 직전 마지막 30분 크롭 (학습·평가 기준 구간)", fontsize=11)
    axes[2].set_ylabel("bpm")

    t1 = np.arange(x.shape[1]) / 60
    axes[3].plot(t1, x[0], lw=0.8, color=C_FHR, label="FHR")
    axes[3].plot(t1, x[1], lw=0.8, color=C_UC, label="UC")
    axes[3].set_title("4) 1 Hz 다운샘플(1,800 포인트) + 채널별 max-abs 스케일링 → CNN 입력 (2×1800)",
                      fontsize=11)
    axes[3].set_ylabel("스케일값")
    axes[3].set_xlabel("시간 (분)")
    axes[3].legend(loc="lower left", frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"레코드 {rid}의 전처리 파이프라인 (Chiou et al., 2025 Methods 재구현)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_preprocessing.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    print("저장:", FIG_DIR / "11_preprocessing.png")


if __name__ == "__main__":
    fig_preprocessing()
