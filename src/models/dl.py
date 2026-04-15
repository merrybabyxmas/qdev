import torch
from torch import nn
import pandas as pd
import numpy as np
from src.utils.logger import logger

class LSTMPredictor(nn.Module):
    """
    F003 / F004: LSTM Return/Direction Model.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class MLPFeatureExtractor(nn.Module):
    """
    F005 / F010: MLP Ranker / Event Alpha.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

class TransformerForecaster(nn.Module):
    """
    F006 / F008: Transformer Forecaster.
    """
    def __init__(self, input_dim: int, num_heads: int = 2, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        # Project input to model dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out

class FactorAutoencoder(nn.Module):
    """
    F017: Factor + Autoencoder Latent Alpha.
    """
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim)
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        alpha = self.head(latent)
        return reconstruction, alpha


class TFTForecaster(nn.Module):
    """
    F007: Temporal Fusion Transformer proxy.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.static_encoder = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.transformer(lstm_out)
        return self.fc(out[:, -1, :])


class PatchTSTModel(nn.Module):
    """
    F021: PatchTST Baseline proxy.
    """
    def __init__(self, input_dim: int, patch_len: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(input_dim * patch_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        # pad if needed to make divisible by patch_len
        pad_len = (self.patch_len - seq_len % self.patch_len) % self.patch_len
        if pad_len > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
            seq_len += pad_len

        num_patches = seq_len // self.patch_len
        x = x.view(batch_size, num_patches, self.patch_len * input_dim)
        x = self.proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


class GNNAlphaModel(nn.Module):
    """
    F018: Graph Neural Stock Relation Model proxy.
    (Operates on flattened features treating feature elements as node attributes)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Simple GNN-like linear mixer (since we lack actual graph topologies in basic df)
        self.node_proj = nn.Linear(1, hidden_dim)
        self.message_passing = nn.Linear(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim * input_dim, 1)

    def forward(self, x):
        # x shape: (batch, input_dim)
        x = x.unsqueeze(-1) # (batch, num_nodes, 1)
        x = torch.relu(self.node_proj(x))
        x = torch.relu(self.message_passing(x)) # Simulate 1 layer of message passing
        x = x.view(x.size(0), -1) # flatten nodes
        return self.readout(x)


class MultimodalFusionModel(nn.Module):
    """
    F011: Multimodal fusion proxy (e.g., Price + News).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Assume first half is price, second half is news embedding
        self.dim1 = input_dim // 2
        self.dim2 = input_dim - self.dim1

        self.branch1 = nn.Sequential(nn.Linear(self.dim1, hidden_dim), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Linear(self.dim2, hidden_dim), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x1 = x[:, :self.dim1]
        x2 = x[:, self.dim1:]
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        fused = torch.cat([out1, out2], dim=-1)
        return self.fusion(fused)


class DeepLearningModel:
    """
    Unified wrapper for training and predicting with PyTorch Financial DL Models.
    """
    def __init__(self, model_type: str, features: list[str], target: str = 'target_return', seq_len: int = 5):
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.is_fitted = False
        self.model_type = model_type

        input_dim = len(features)

        from torch.utils.data import TensorDataset, DataLoader

        if model_type == 'LSTM':
            self.model = LSTMPredictor(input_dim)
        elif model_type == 'MLP':
            self.model = MLPFeatureExtractor(input_dim)
            self.seq_len = 1 # MLP takes flat inputs
        elif model_type == 'Transformer':
            self.model = TransformerForecaster(input_dim)
        elif model_type == 'Autoencoder':
            self.model = FactorAutoencoder(input_dim)
            self.seq_len = 1
        elif model_type == 'TFT':
            self.model = TFTForecaster(input_dim)
        elif model_type == 'PatchTST':
            self.model = PatchTSTModel(input_dim)
        elif model_type == 'GNN':
            self.model = GNNAlphaModel(input_dim)
            self.seq_len = 1
        elif model_type == 'Multimodal':
            self.model = MultimodalFusionModel(input_dim)
            self.seq_len = 1
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def _prepare_data(self, df: pd.DataFrame):
        X_raw = df[self.features].values
        Y_raw = df[self.target].values

        if self.seq_len == 1:
            return torch.tensor(X_raw, dtype=torch.float32), torch.tensor(Y_raw, dtype=torch.float32).unsqueeze(1)

        X_seq, Y_seq = [], []
        for i in range(len(X_raw) - self.seq_len + 1):
            X_seq.append(X_raw[i : i + self.seq_len])
            Y_seq.append(Y_raw[i + self.seq_len - 1])

        return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(Y_seq), dtype=torch.float32).unsqueeze(1)

    def fit(self, df: pd.DataFrame, epochs: int = 5, batch_size: int = 32):
        if any(f not in df.columns for f in self.features) or self.target not in df.columns:
            logger.error("Missing columns for DL Model fit.")
            return

        logger.info(f"Fitting {self.model_type} DL Model...")
        X, Y = self._prepare_data(df)

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()

                if self.model_type == 'Autoencoder':
                    recon, alpha = self.model(batch_x)
                    loss_recon = nn.MSELoss()(recon, batch_x)
                    loss_alpha = self.criterion(alpha, batch_y)
                    loss = loss_recon + loss_alpha
                else:
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)

                loss.backward()
                self.optimizer.step()

        self.is_fitted = True
        logger.info(f"{self.model_type} Model fitted.")

    def save(self, path) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "features": self.features,
                "target": self.target,
                "seq_len": self.seq_len,
                "state_dict": self.model.state_dict(),
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.debug(f"DeepLearningModel ({self.model_type}) saved to {path}")

    @classmethod
    def load(cls, path) -> "DeepLearningModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            model_type=ckpt["model_type"],
            features=ckpt["features"],
            target=ckpt["target"],
            seq_len=ckpt["seq_len"],
        )
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.is_fitted = ckpt["is_fitted"]
        logger.debug(f"DeepLearningModel ({ckpt['model_type']}) loaded from {path}")
        return obj

    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        if not self.is_fitted or df.empty:
            return np.zeros(len(df))

        X, _ = self._prepare_data(df)

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for batch_x in dataloader:
                if self.model_type == 'Autoencoder':
                    _, preds = self.model(batch_x[0])
                else:
                    preds = self.model(batch_x[0])
                preds_list.extend(preds.view(-1).cpu().numpy().tolist())

        # Pad beginning for sequence models to match input df length
        out = np.zeros(len(df))
        out[self.seq_len - 1:] = np.array(preds_list)
        return out
