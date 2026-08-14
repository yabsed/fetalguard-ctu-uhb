"""CTG-net (Ogasawara et al., 2021)의 PyTorch 재구현 + Chiou et al. (2025) 변형.

원논문(Sci. Rep. 11:13367) 구조:
  입력: FHR+UC, 1,800 포인트 (30분 @ 1 Hz)
  conv1: 30초-bin 시간 커널 합성곱 → BatchNorm
  conv2: depthwise 합성곱 (FHR-UC 관계 학습) → BatchNorm → ELU → AvgPool → Dropout(0.25)
  conv3: separable 합성곱 → BatchNorm → ELU → AvgPool → Dropout(0.25)
  flatten → dense(sigmoid) → abnormality 점수
  파라미터 ~2,130개, Adam(lr=1e-3, eps=1e-5)

Chiou et al. 변형:
  - FHR 또는 UC 단일 채널 입력 (1채널 변형)
  - flatten 뒤 메타데이터 벡터 연결 (FHR+UC+metadata)
  - flatten 뒤 완전연결 은닉층 — 본문 명시("flattened and passed to
    fully-connected hidden layers"); 층수·차원은 프리프린트 Appendix B의
    아키텍처 탐색 대상이며 선택값은 미공개
  - 필터 수·은닉층 구성은 아키텍처 탐색 대상 (500개 무작위 구성 중 검증
    AUROC 최고 선택)

학습 하이퍼파라미터는 medRxiv 프리프린트 Appendix B에 명시된 탐색 시 고정값을
따른다: BCE 손실, 학습률 3e-4, 배치 128, dropout 0.2. 커널·풀 크기, 필터 수,
은닉층 수·차원은 어느 판에도 공개되지 않아 [재구현 선택] 값을 기본값으로 둔다 —
필터 4개는 원논문의 "4–8 filters" 범위 하한이자 파라미터 수(~2.1k)와 일치하는
조합이고, 은닉층 기본값은 없음(원논문 CTG-net의 flatten → dense(sigmoid) 구조;
hidden_dims 인자로 Chiou et al.의 은닉층 변형을 지원한다).
"""

import numpy as np
import torch
import torch.nn as nn


class CTGNet(nn.Module):
    def __init__(self, in_channels=2, f1=4, f2=4, f3=4, k1=30, k2=15, k3=15,
                 pool=2, dropout=0.2, input_len=1800, n_metadata=0,
                 hidden_dims=()):
        """
        in_channels: 2 (FHR+UC) 또는 1 (FHR only / UC only)
        f1, f2, f3 : 각 합성곱 층 필터 수 (구글 논문의 탐색 대상)
        k1        : 30초-bin 시간 커널 (원논문 명시)
        k2, k3    : [재구현 선택] 15
        dropout   : 0.2 (프리프린트 Appendix B 명시)
        n_metadata: >0이면 flatten 뒤 메타데이터 벡터를 연결
        hidden_dims: flatten(+메타데이터) 뒤 완전연결 은닉층 차원들 — Chiou et
                    al.이 본문에 명시하고 층수·차원을 탐색한 부분(선택값 미공개).
                    [재구현 선택] 기본값 ()은 은닉층 없음(원논문 구조), 층간
                    활성함수는 ELU
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, f1, k1)
        self.bn1 = nn.BatchNorm1d(f1)
        # depthwise: 채널(필터)별 독립 합성곱
        self.conv2 = nn.Conv1d(f1, f2, k2, groups=f1)
        self.bn2 = nn.BatchNorm1d(f2)
        # separable: depthwise + pointwise
        self.conv3_dw = nn.Conv1d(f2, f2, k3, groups=f2)
        self.conv3_pw = nn.Conv1d(f2, f3, 1)
        self.bn3 = nn.BatchNorm1d(f3)
        self.act = nn.ELU()
        self.pool = nn.AvgPool1d(pool)
        self.drop = nn.Dropout(dropout)

        # 유효 길이 계산 (padding 없음, 원논문 Keras 기본 'valid'과 동일)
        L = input_len - k1 + 1
        L = (L - k2 + 1) // pool
        L = (L - k3 + 1) // pool
        self.flatten_dim = f3 * L + n_metadata
        fc_layers, dim = [], self.flatten_dim
        for h in hidden_dims:          # Chiou et al.의 완전연결 은닉층 변형
            fc_layers += [nn.Linear(dim, h), nn.ELU()]
            dim = h
        fc_layers.append(nn.Linear(dim, 1))
        self.fc = nn.Sequential(*fc_layers)
        self.n_metadata = n_metadata

    def forward(self, x, metadata=None):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.drop(self.pool(self.act(x)))
        x = self.bn3(self.conv3_pw(self.conv3_dw(x)))
        x = self.drop(self.pool(self.act(x)))
        x = torch.flatten(x, 1)
        if self.n_metadata > 0:
            x = torch.cat([x, metadata], dim=1)
        return self.fc(x).squeeze(1)   # 로짓 (평가 시 sigmoid)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(channels="fhr_uc", n_metadata=0, **kwargs):
    """채널 설정별 모델 생성. channels: fhr_uc | fhr | uc"""
    in_ch = 1 if channels in ("fhr", "uc") else 2
    model = CTGNet(in_channels=in_ch, n_metadata=n_metadata, **kwargs)
    return model


def make_optimizer(model, lr=3e-4, eps=1e-5):
    """Adam. 학습률 3e-4는 프리프린트 Appendix B 명시값,
    eps 1e-5는 원논문(Ogasawara) 설정 [재구현 선택]."""
    return torch.optim.Adam(model.parameters(), lr=lr, eps=eps)


if __name__ == "__main__":
    # 참고: GPU 없이도 확인 가능한 형태 점검 (데이터 불필요)
    for ch in ("fhr_uc", "fhr"):
        m = build_model(ch)
        x = torch.zeros(2, 1 if ch == "fhr" else 2, 1800)
        print(f"{ch:7s} 파라미터 {count_parameters(m):,}개, 출력 {m(x).shape}")
    m = build_model("fhr_uc", n_metadata=11)
    print(f"메타데이터 포함 flatten_dim={m.flatten_dim}")
