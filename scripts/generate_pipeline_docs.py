import os

docs_to_create = [
    "docs/pipeline_library/00_pipeline_catalog.md",
    "docs/pipeline_library/01_bayesian_pipelines.md",
    "docs/pipeline_library/02_sde_pipelines.md",
    "docs/pipeline_library/03_rl_pipelines.md",
    "docs/pipeline_library/04_financial_dl_pipelines.md",
    "docs/pipeline_library/05_priority_shortlist.md",
    "docs/pipeline_library/06_rejected_or_archived_pipelines.md"
]

template_content = """# Title

- 목적:
- 핵심 질문:
- 이론 요약:
- 대표 참고자료:
- 적용 가능한 알고리즘:
- 장점:
- 한계:
- 구현 난이도:
- 데이터 요구사항:
- 리스크 및 실패 모드:
- 실제 시스템 적용 여부:
- 향후 개선 방향:
"""

for filepath in docs_to_create:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(template_content)

print(f"Generated {len(docs_to_create)} pipeline documentation templates.")
