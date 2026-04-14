import os
from enum import Enum

from pydantic import BaseModel, Field

from src.utils.env import load_repo_env


load_repo_env()


class RuntimeMode(str, Enum):
    dev = "dev"
    research = "research"
    paper = "paper"
    live = "live"

class BrokerConfig(BaseModel):
    api_key: str = Field(default="mock_api_key", description="Broker API Key")
    secret_key: str = Field(default="mock_secret_key", description="Broker Secret Key")
    paper: bool = Field(default=True, description="Use paper trading environment")

class SystemConfig(BaseModel):
    mode: RuntimeMode = Field(default=RuntimeMode.dev, description="System mode: dev, research, paper, live")
    broker: BrokerConfig = Field(default_factory=BrokerConfig)

    @staticmethod
    def _parse_bool(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def load(cls):
        config = cls(
            mode=os.getenv("SYS_MODE", RuntimeMode.dev.value),
            broker=BrokerConfig(
                api_key=os.getenv("BROKER_API_KEY", "mock_key"),
                secret_key=os.getenv("BROKER_SECRET_KEY", "mock_secret"),
                paper=cls._parse_bool(os.getenv("BROKER_PAPER"), True),
            ),
        )

        allow_live = cls._parse_bool(os.getenv("ALLOW_LIVE_TRADING"), False)
        if config.mode == RuntimeMode.live and not allow_live:
            raise ValueError("SYS_MODE=live requires ALLOW_LIVE_TRADING=true")

        if config.mode == RuntimeMode.live:
            config.broker.paper = False
        elif config.mode == RuntimeMode.paper:
            config.broker.paper = True

        return config

    @property
    def is_live(self) -> bool:
        return self.mode == RuntimeMode.live

config = SystemConfig.load()
