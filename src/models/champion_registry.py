import json
import os
import shutil
from typing import Dict, Any
from src.utils.logger import logger

class ChampionRegistry:
    """
    오프라인 평가 파이프라인이 갱신하는 챔피언 모델 레지스트리.
    라이브 스트리밍 엔진은 항상 여기서 1위 모델(Champion)을 조회하여
    현재 시장 상태(Regime)에 맞는 최적의 인퍼런스를 수행함.
    """
    def __init__(self, registry_path: str = "data/models/champion_registry.json"):
        self.registry_path = registry_path

        # Initialize default structure if missing
        if not os.path.exists(self.registry_path):
            self._init_default_registry()

    def _init_default_registry(self):
        default_data = {
            "current_champion": {
                "model_id": "M-BASE-LGBM-001",
                "family": "LightGBM",
                "sharpe": 1.2,
                "max_drawdown": 0.15,
                "last_updated": "2024-01-01T00:00:00Z",
                "path": "data/models/m_base_lgbm_001.pkl"
            },
            "history": []
        }
        with open(self.registry_path, "w") as f:
            json.dump(default_data, f, indent=4)
        logger.info("Initialized default Champion Model Registry.")

    def get_champion(self) -> Dict[str, Any]:
        """라이브 엔진이 주기적으로 챔피언 모델 설정을 조회할 때 호출"""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            return data.get("current_champion", {})
        except Exception as e:
            logger.error(f"Failed to read champion registry: {e}")
            return {}

    def register_new_champion(self, model_metadata: Dict[str, Any]):
        """
        오프라인 배치 스케줄러가 모델을 훈련하고 검증한 뒤,
        Sharpe Ratio 등의 지표가 기존 챔피언을 능가하면 이 메서드를 통해 챔피언을 교체.
        """
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            old_champion = data.get("current_champion", {})
            if old_champion:
                data["history"].append(old_champion)

            data["current_champion"] = model_metadata

            # Safe write with temp file
            tmp_path = self.registry_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, self.registry_path)

            logger.info(f"Successfully promoted new Champion Model: {model_metadata.get('model_id')} "
                        f"(Sharpe: {model_metadata.get('sharpe')})")
        except Exception as e:
            logger.error(f"Failed to register new champion: {e}")
