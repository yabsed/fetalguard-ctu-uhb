"""Chiou et al. (2025) Methods 기반 CTG 전처리.

명세의 준거는 medRxiv 프리프린트(doi:10.1101/2024.03.05.24303805) Appendix
A.1이다 — 출판본에는 단계 이름만 있으나 프리프린트에는 수치가 명시되어 있다.

신경망 파이프라인 단계 (A.1 명시):
 1) 기록 앞/뒤의 반복 결측(FHR=0) 제거 — 동시 기록된 UC 구간도 함께 제거
 2) 품질 평가: 5분 슬라이딩 창(1분 보폭)에서 FHR 결측이 50% 이상인 창을
    표시해 두고, 보간·평활이 끝난 뒤 해당 구간을 0으로 재지정
 3) 결측 구간 분류: 15초 초과 → 결측으로 표시하고 0 유지(시간적 의존성 보존),
    15초 이하 → 선형 보간
 4) FHR/UC 평활: 길이 15포인트 Hamming 창 양방향 이동평균 (Ogasawara et al.을
    따른다고 명시) — 결측 0은 값이 아니라 표지이므로, 평활은 유효 표본만
    가중평균하고 표지는 그대로 보존한다(smooth_masked 주석 참조)
 5) 30분 크롭 (기본: 분만 직전 마지막 30분)
 6) 1 Hz 다운샘플 → 1,800 포인트 (모델 입력 차원 절감)
 7) 채널별 최대절대값(max-abs) 스케일링

학습 데이터 증강 (A.1 명시): 레코드당 무작위 크롭 10개 — 마지막 30분의 시작점을
최대 4분 앞으로 이동 — 에 채널별로 전역(신호 전체, N(0, 분산 4)) 또는 국소
(1~5개 구간) 가산 노이즈. 증강 표본은 학습에만 쓰고, 평가는 결정론적 마지막
30분 크롭을 쓴다.

논문이 공개하지 않은 세부만 [재구현 선택]으로 표기한다
(논문 코드는 비공개 — "may be made available upon reasonable request").
"""

import numpy as np

FS = 4                    # 원본 샘플링 (Hz)
CROP_MIN = 30             # 크롭 길이 (분)
CROP_LEN = CROP_MIN * 60  # 초
LONG_GAP_S = 15           # 이 초과 길이의 결측은 '결측'으로 유지 (논문 명시)
SMOOTH_WINDOW = 15        # 평활 창 15포인트, Hamming (A.1 명시; Ogasawara 방식)
QUALITY_WIN_S = 300       # 품질 평가 창 5분 (A.1 명시)
QUALITY_STRIDE_S = 60     # 품질 평가 보폭 1분 (A.1 명시)
QUALITY_MAX_MISSING = 0.5  # 창 내 결측 상한 — 이상이면 창 전체 0 재지정 (A.1 명시)


# ---------------------------------------------------------------- 결측 처리

def gap_runs(mask):
    """True 구간의 (start, end) 런 목록. end는 exclusive."""
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def trim_edge_missing(fhr, uc):
    """1단계: 기록 앞/뒤의 연속 결측(FHR=0) 제거. UC도 같은 구간 제거."""
    present = fhr != 0
    if not present.any():
        return fhr, uc
    first, last = np.argmax(present), len(present) - np.argmax(present[::-1])
    return fhr[first:last], uc[first:last]


def impute_short_gaps(x, max_gap_samples):
    """3단계: max_gap_samples 이하의 0-런을 선형 보간. 더 긴 런은 0 유지.

    [재구현 선택] 논문은 "shorter than 15 seconds"를 보간한다고 쓴다 — 정확히
    15초(4 Hz에서 60표본)인 런의 귀속은 이산화 관례의 문제로, 여기서는 보간에
    포함한다(≤).
    """
    x = x.copy()
    for s, e in gap_runs(x == 0):
        if e - s > max_gap_samples:
            continue  # 긴 결측: 0 유지 (논문: set them to zero)
        left = x[s - 1] if s > 0 else np.nan
        right = x[e] if e < len(x) else np.nan
        if np.isnan(left) and np.isnan(right):
            continue
        if np.isnan(left):
            x[s:e] = right
        elif np.isnan(right):
            x[s:e] = left
        else:
            x[s:e] = np.linspace(left, right, e - s + 2)[1:-1]
    return x


def smooth(x, window):
    """단순 이동평균 — 결측이 없는(전부 보간된) 신호 전용.

    결측이 0으로 남아 있는 신호에는 쓰지 말 것: 0이 평균에 섞여 결측 경계의
    유효 표본을 끌어내리고, 긴 결측을 0이 아닌 완만한 계곡으로 바꾼다.
    그런 신호에는 smooth_masked를 쓴다.
    """
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def smooth_masked(x, window, missing, kernel=None):
    """결측 표지를 존중하는 rolling (가중)평균 — 창 안의 **유효 표본만** 평균한다.

    논문의 결측 0은 심박수가 아니라 "여기는 신호가 없었다"는 표지다(set them to
    zero to maintain temporal dependencies). 뒤이은 평활(smoothed to reduce the
    effect of noise / smoothed with a rolling window)은 신호의 노이즈를 누르는
    단계이지 표지를 값으로 평균하는 단계가 아니므로, 분자·분모를 함께 합성곱해
    유효 표본만으로 평균을 내고 결측 위치는 0으로 되돌린다. 결측을 NaN으로 둔
    pandas의 rolling(window, min_periods=1).mean()과 동치이며(boxcar 기준),
    0을 값으로 취급하는 단순 이동평균이 만드는 두 인공물(경계 오염, 긴 결측의
    계곡화)이 없다.

    kernel: 가중 창 (기본 None = boxcar). 신경망 파이프라인은 A.1 명시대로
    np.hamming(15)를 넘긴다.
    """
    if window <= 1:
        return x
    valid = (~missing).astype(np.float64)
    kernel = np.ones(window) if kernel is None else np.asarray(kernel, dtype=np.float64)
    num = np.convolve(np.where(missing, 0.0, x), kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.divide(num, den, out=np.zeros(len(x), dtype=np.float64), where=den > 0)
    out[missing] = 0.0
    return out


def flag_low_quality(fhr, fs=FS, win_s=QUALITY_WIN_S, stride_s=QUALITY_STRIDE_S,
                     max_missing=QUALITY_MAX_MISSING):
    """2단계 품질 평가 (A.1 명시): 결측 50% 이상인 5분 창의 표본을 표시한다.

    A.1: "A 5-minute sliding window with a 1-minute stride was used to ensure
    that the FHR signal loss was less than 50% within the window. Time steps
    that violate this condition were flagged for zero-value assignment after
    the imputation and smoothing steps." (Asfaw et al. 방식)

    보간 전의 결측(FHR=0) 기준으로 계산하고, 반환된 마스크는 보간·평활이 끝난
    신호에 적용한다. [재구현 선택] 모든 표본이 최소 한 창에 들어가도록 신호 끝에
    창 하나를 추가로 정렬한다(신호가 5분보다 짧으면 전체를 한 창으로 평가).
    """
    miss = fhr == 0
    flags = np.zeros(len(fhr), dtype=bool)
    win, stride = int(win_s * fs), int(stride_s * fs)
    if len(fhr) == 0:
        return flags
    last = max(len(fhr) - win, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)          # 끝 정렬 창 — 꼬리 표본도 평가
    for s in starts:
        w = miss[s:s + win]
        if w.mean() >= max_missing:
            flags[s:s + win] = True
    return flags


# ---------------------------------------------------------------- 파이프라인

def preprocess_record(fhr, uc, fs=FS, smooth_window=SMOOTH_WINDOW):
    """1~4단계: 가장자리 결측 제거 → 품질 평가 → 짧은 결측 보간 → Hamming 평활
    → 저품질 창 0 재지정.

    평활 창은 A.1 명시값(15포인트, Hamming, 양방향 이동평균 — Ogasawara 방식)이다.
    평활 후에도 긴 결측(>15초)은 정확히 0으로 남고, 품질 평가에서 표시된 5분 창
    (결측 ≥50%)은 보간·평활 뒤 통째로 0으로 재지정된다 — 논문이 의도한 결측
    표지가 모델 입력까지 보존된다.

    [재구현 선택] A.1의 품질 평가·0 재지정은 FHR 기준으로 서술되므로 FHR에만
    적용한다(UC는 언급이 없다).

    반환: (fhr, uc) — 길이가 가변인 정제 신호 (fs Hz).
    """
    fhr, uc = trim_edge_missing(fhr, uc)
    low_q = flag_low_quality(fhr, fs)
    fhr = impute_short_gaps(fhr, int(LONG_GAP_S * fs))
    uc = impute_short_gaps(uc, int(LONG_GAP_S * fs))
    ham = np.hamming(smooth_window) if smooth_window > 1 else None
    fhr = smooth_masked(fhr, smooth_window, fhr == 0, ham)
    uc = smooth_masked(uc, smooth_window, uc == 0, ham)
    fhr = fhr.copy()
    fhr[low_q] = 0.0
    return fhr, uc


# ---------------------------------------------------------------- 크롭

def crop_last_minutes(x, minutes=CROP_MIN, fs=FS, end_offset_min=0):
    """기록 끝에서 end_offset_min만큼 물러난 지점 기준 minutes 길이 크롭.

    길이가 부족하면 None 반환. CTU-UHB에서는 발생하지 않는다 — 원신호가 전부
    60분 이상이라 마지막 30분 크롭은 552건 모두 가능하다. (논문의 496/56은
    크롭 탈락이 아니라 90%/10% 학습·테스트 분할이다 — run_all.write_summary 참조.)
    """
    n = int(minutes * 60 * fs)
    end = len(x) - int(end_offset_min * 60 * fs)
    if end - n < 0 or end <= 0:
        return None
    return x[end - n:end]


def sliding_crops_excluding_last(x, crop_min=CROP_MIN, stride_min=1,
                                 exclude_last_min=30, fs=FS):
    """Alternative interval (1): 마지막 exclude_last_min분을 제외한 구간에서
    crop_min 창을 stride_min 간격으로 슬라이딩한 크롭 목록 (사전학습용)."""
    crop_n = int(crop_min * 60 * fs)
    stride_n = int(stride_min * 60 * fs)
    stop = len(x) - int(exclude_last_min * 60 * fs)
    crops = []
    for start in range(0, stop - crop_n + 1, stride_n):
        crops.append(x[start:start + crop_n])
    return crops


def random_crop(x, rng, crop_min=CROP_MIN, fs=FS):
    """Alternative interval (2): 전체 기록에서 무작위 30분 창 하나."""
    n = int(crop_min * 60 * fs)
    if len(x) < n:
        return None
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start:start + n]


# ---------------------------------------------------------------- 다운샘플·스케일

def downsample_to_1hz(x, fs=FS):
    """6단계: 1 Hz 다운샘플 (fs-포인트 블록 평균). 30분 크롭 → 1,800 포인트."""
    n = len(x) // fs * fs
    return x[:n].reshape(-1, fs).mean(axis=1).astype(np.float32)


def maxabs_scale(fhr, uc):
    """7단계: 채널별 최대절대값 스케일링 (논문: channel-specific maximum
    absolute value scaling).

    [재구현 선택] 논문은 "채널별"이라고만 명시하고 배율의 단위(데이터셋 전역 /
    레코드 / 크롭)는 밝히지 않는다. 여기서는 크롭 하나를 그 크롭의 채널별
    최대절대값(결측 0 포함)으로 나눈다 — 크롭마다 배율이 달라지므로 절대
    심박수 수준은 크롭 간에 보존되지 않는다."""
    fhr_s = fhr / np.abs(fhr).max() if np.abs(fhr).max() > 0 else fhr
    uc_s = uc / np.abs(uc).max() if np.abs(uc).max() > 0 else uc
    return fhr_s.astype(np.float32), uc_s.astype(np.float32)


def make_model_input(fhr, uc, fs=FS):
    """전처리 완료 신호의 마지막 30분 → (2, 1800) float32 스케일 완료 입력."""
    fhr_c = crop_last_minutes(fhr, fs=fs)
    uc_c = crop_last_minutes(uc, fs=fs)
    if fhr_c is None or uc_c is None:
        return None
    fhr_d, uc_d = downsample_to_1hz(fhr_c, fs), downsample_to_1hz(uc_c, fs)
    fhr_s, uc_s = maxabs_scale(fhr_d, uc_d)
    return np.stack([fhr_s, uc_s])


# ---------------------------------------------------------------- 증강

N_AUG_CROPS = 10          # 레코드당 증강 크롭 수 (A.1 명시)
AUG_MAX_SHIFT_S = 240     # 크롭 시작점 최대 이동 4분 (A.1 명시)
AUG_GLOBAL_VAR = 4.0      # 전역 가산 노이즈 분산 (A.1 명시)
AUG_MAX_LOCAL_SEG = 5     # 국소 노이즈 구간 수 1~5 (A.1 명시)


def augment_crops(fhr, uc, rng, fs=FS, n_crops=N_AUG_CROPS,
                  max_shift_s=AUG_MAX_SHIFT_S, global_var=AUG_GLOBAL_VAR,
                  local_var=AUG_GLOBAL_VAR, max_local_segments=AUG_MAX_LOCAL_SEG):
    """학습용 증강 (A.1 명시, Zhou et al. 방식): 무작위 크롭 + 다중 스케일 노이즈.

    전처리 완료(4 Hz) 레코드에서 —
      · 마지막 30분의 시작점을 0~4분 균등추출로 앞당긴 크롭을 레코드당 10개 생성
      · 크롭·채널마다 베르누이 추첨: 전역 노이즈(신호 전체에 N(0, 분산 4)) 또는
        국소 노이즈(1~5개 구간, 구간별 시작·끝 무작위)
    증강 표본은 학습 전용이다 — 평가는 결정론적 마지막 30분 크롭(무노이즈).

    [재구현 선택] 국소 노이즈의 분산은 논문에 없어 전역과 같은 값(4)을 쓰고,
    국소 구간 경계는 크롭 안에서 균등추출한 두 점을 정렬해 얻는다. 노이즈는
    A.1의 "added to the entire signal"대로 결측 0 표지 위에도 더해진다.

    반환: [(fhr_crop, uc_crop), ...] — 4 Hz 원척도 쌍. 이후 downsample_to_1hz와
    maxabs_scale을 평가 크롭과 동일하게 적용한다.
    """
    n = int(CROP_LEN * fs)
    last_start = len(fhr) - n
    if last_start < 0:
        return []
    out = []
    for _ in range(n_crops):
        shift = int(rng.integers(0, int(max_shift_s * fs) + 1))
        start = max(last_start - shift, 0)
        fc = fhr[start:start + n].astype(np.float64).copy()
        tc = uc[start:start + n].astype(np.float64).copy()
        for ch in (fc, tc):
            if rng.integers(0, 2):   # 전역
                ch += rng.normal(0.0, np.sqrt(global_var), n)
            else:                    # 국소
                for _ in range(int(rng.integers(1, max_local_segments + 1))):
                    a, b = sorted(int(v) for v in rng.integers(0, n, 2))
                    if b > a:
                        ch[a:b] += rng.normal(0.0, np.sqrt(local_var), b - a)
        out.append((fc, tc))
    return out
