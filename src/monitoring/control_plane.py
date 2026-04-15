import json
import os
from datetime import datetime, timezone
from typing import Dict, Any
from src.utils.logger import logger

class HFTControlPlane:
    """
    Model Scheduler (Macro/Daily/Intraday) 역할을 하는 상위 제어기(Control Plane).
    현재 시장의 거시적/중기적 상태(Context)를 분석하여 HFT 엔진(Execution Plane)이
    어떤 자산을, 어떤 조건(Threshold)하에서 거래할지 결정하는 정책(Policy)을 발행합니다.
    """
    def __init__(self, policy_path: str = "artifacts/control_plane/hft_policy.json"):
        self.policy_path = policy_path
        os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)

    def generate_policy(self,
                        regime: str,
                        allow_hft: bool,
                        symbol_configs: Dict[str, Dict[str, Any]],
                        global_thresholds: Dict[str, float]):
        """
        상위 로직(스케줄러)에서 결정된 HFT 가이드라인을 JSON 아티팩트로 발행합니다.
        HFT 엔진은 이 파일을 주기적으로 읽어 안전하게 틱(tick) 단위 실행을 통제받습니다.
        """
        policy = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "allow_hft": allow_hft,
            "symbols": symbol_configs,
            "thresholds": global_thresholds
        }

        # 원자적(Atomic) 쓰기를 위해 임시 파일 사용
        tmp_path = self.policy_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(policy, f, indent=4)
            os.replace(tmp_path, self.policy_path)
            logger.info(f"Published new HFT Policy: {regime}, allow_hft={allow_hft}")
        except Exception as e:
            logger.error(f"Failed to publish HFT Policy: {e}")

    def read_policy(self) -> Dict[str, Any]:
        """
        HFT 엔진(Execution Plane)이 주기적으로 상위 정책을 읽어올 때 사용.
        """
        if not os.path.exists(self.policy_path):
            logger.warning("HFT Policy file not found. Defaulting to safe halt mode.")
            return self._default_safe_policy()

        try:
            with open(self.policy_path, "r") as f:
                policy = json.load(f)
            return policy
        except Exception as e:
            logger.error(f"Failed to read HFT Policy: {e}. Defaulting to safe halt mode.")
            return self._default_safe_policy()

    def _default_safe_policy(self) -> Dict[str, Any]:
        """정책 파일이 깨지거나 없을 때 시스템을 보호하는 기본(Default) Safe 정책"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": "UNKNOWN",
            "allow_hft": False,
            "symbols": {},
            "thresholds": {
                "spread_max_bps": 5.0,
                "toxicity_max": 0.5,
                "prediction_bps_min": 1.0
            }
        }
