import os
from pydantic import BaseModel, Field

class BrokerConfig(BaseModel):
    api_key: str = Field(default="mock_api_key", description="Broker API Key")
    secret_key: str = Field(default="mock_secret_key", description="Broker Secret Key")
    paper: bool = Field(default=True, description="Use paper trading environment")

class SystemConfig(BaseModel):
    mode: str = Field(default="dev", description="System mode: dev, research, paper, live")
    broker: BrokerConfig = BrokerConfig()

    @classmethod
    def load(cls):
        return cls(
            mode=os.getenv("SYS_MODE", "dev"),
            broker=BrokerConfig(
                api_key=os.getenv("BROKER_API_KEY", "mock_key"),
                secret_key=os.getenv("BROKER_SECRET_KEY", "mock_secret"),
                paper=os.getenv("BROKER_PAPER", "True").lower() == "true"
            )
        )

config = SystemConfig.load()
