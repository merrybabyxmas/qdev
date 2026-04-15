import torch
import torch.nn as nn
from src.utils.logger import logger

class CompactDeepLOB(nn.Module):
    """
    HFT_DL_001: Compact DeepLOB Classifier.
    Order Book의 Snapshot 시퀀스를 입력받아 Next-K-Tick Direction(UP/FLAT/DOWN)을 분류하는
    가벼운 CNN + LSTM 하이브리드 모델. 레이턴시(Latency) 제약조건에 맞춰 채널 수와 레이어를 간소화함.
    """
    def __init__(self, input_features: int = 10, sequence_length: int = 50, num_classes: int = 3):
        super(CompactDeepLOB, self).__init__()
        self.input_features = input_features
        self.seq_len = sequence_length
        self.num_classes = num_classes

        # 1D-CNN Feature Extractor (Microstructure local patterns)
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # seq_len / 2

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # seq_len / 4

        # LSTM Temporal Extractor (Sequence patterns)
        self.lstm_hidden_size = 32
        self.lstm = nn.LSTM(input_size=32, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True)

        # Classifier Head
        self.fc1 = nn.Linear(self.lstm_hidden_size, 16)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        """
        x: [Batch Size, Sequence Length, Features]
        """
        # CNN은 (N, C, L) 형식을 취하므로 Transpose
        x = x.transpose(1, 2)

        # CNN Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # LSTM을 위해 다시 (N, L, C) 형식으로 변환
        x = x.transpose(1, 2)

        # LSTM Block (Output of last timestep)
        out, (hn, cn) = self.lstm(x)
        last_hidden_state = out[:, -1, :]

        # Classifier Block
        y = self.fc1(last_hidden_state)
        y = self.relu3(y)
        y = self.dropout(y)
        y = self.fc2(y)

        # Return raw logits (CrossEntropyLoss expects raw logits)
        return y
