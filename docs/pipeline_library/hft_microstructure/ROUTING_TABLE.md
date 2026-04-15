# HFT Routing Table

| Regime | HFT Allow | Preferred Pipeline | Risk Overlay | Notes |
| --- | --- | --- | --- | --- |
| LOW_VOL_TREND | yes | HFT_BASE_001 | HFT_RISK_001 | threshold 완화 |
| LOW_VOL_MEAN_REVERT | yes | HFT_SDE_001 | HFT_RISK_001 | quote skew 허용 |
| NORMAL_BALANCED | yes | HFT_BASE_001 / HFT_DL_001 | HFT_RISK_001 | 기본 운영 상태 |
| HIGH_VOL_TREND | limited | HFT_BASE_001 | HFT_SDE_002 | size 축소 |
| HIGH_TOXICITY | no | none | HFT_RISK_001 | halt |
| WIDE_SPREAD_ILLIQUID | no | none | HFT_RISK_001 | halt |
| EVENT_WINDOW | limited | HFT_HYB_002 | HFT_SDE_002 | threshold 강화 |
| CRISIS | no | none | HFT_RISK_001 | full halt |
