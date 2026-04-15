import json
import os
from typing import Dict, Any
from src.utils.logger import logger

class ChampionRegistry:
    """
    오프라인 평가 파이프라인이 갱신하는 챔피언 모델 레지스트리.
    장기적으로 HFT 전용, 일봉 매크로 전용, 분봉/스윙 전용 리더보드를 분리하기 위해
    `horizon`별로 독립된 챔피언 모델을 기록 및 조회합니다.
    """
    def __init__(self, registry_path: str = "data/models/champion_registry.json"):
        self.registry_path = registry_path

        # Initialize default structure if missing
        if not os.path.exists(self.registry_path):
            self._init_default_registry()

    def _init_default_registry(self):
        default_data = {
            "macro_daily": {
                "model_id": "M-BASE-LGBM-001",
                "sharpe": 1.2
            },
            "intraday_swing": {
                "model_id": "M-PRO-XGB-SWING-001",
                "sharpe": 1.5
            },
            "hft_microstructure": {
                "model_id": "HFT_BASE_001",
                "sharpe": 2.0,
                "notes": "OBI + OnlineSGD"
            },
            "history": []
        }
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(default_data, f, indent=4)
        logger.info("Initialized multi-horizon Champion Model Registry.")

    def get_champion(self, horizon: str = "macro_daily") -> Dict[str, Any]:
        """지정된 horizon의 챔피언 모델을 조회합니다."""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            return data.get(horizon, {})
        except Exception as e:
            logger.error(f"Failed to read champion registry for horizon '{horizon}': {e}")
            return {}

    def register_new_champion(self, horizon: str, model_metadata: Dict[str, Any]):
        """특정 horizon의 챔피언 모델을 승격합니다."""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            old_champion = data.get(horizon, {})
            if old_champion:
                old_champion["horizon"] = horizon
                data["history"].append(old_champion)

            data[horizon] = model_metadata

            tmp_path = self.registry_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, self.registry_path)

            logger.info(f"Successfully promoted new [{horizon}] Champion Model: {model_metadata.get('model_id')} "
                        f"(Sharpe: {model_metadata.get('sharpe')})")
        except Exception as e:
            logger.error(f"Failed to register new champion for {horizon}: {e}")
