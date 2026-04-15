import torch
import torch.nn as nn
from src.utils.logger import logger

class EventSequenceLSTM(nn.Module):
    """
    HFT_DL_002: Event Sequence LSTM
    Tick / Quote 이벤트 시퀀스 자체를 입력받아 다음 5-tick 수익률/방향성을 추정하는
    경량화된 LSTM 기반 HFT 예측 모델.
    """
    def __init__(self, input_features: int = 8, sequence_length: int = 30, hidden_size: int = 64, num_layers: int = 2, output_dim: int = 1):
        """
        :param input_features: 이벤트당 피처의 수 (e.g. trade_sign, size, spread, obi)
        :param sequence_length: 이벤트 길이 (최근 30틱 등)
        :param hidden_size: LSTM 은닉층
        :param num_layers: LSTM 층 수
        :param output_dim: 1(수익률 회귀) 또는 3(방향성 분류)
        """
        super(EventSequenceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        x: [Batch Size, Sequence Length, Features]
        """
        # H_0, C_0는 초기값 0으로 처리 (batch_first=True)
        out, (hn, cn) = self.lstm(x)

        # 마지막 timestep의 output만 사용
        last_out = out[:, -1, :]

        y = self.fc(last_out)
        return y
