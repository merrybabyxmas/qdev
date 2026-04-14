SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: bootstrap check-env smoke test pipeline hft docs candidate-docs references quant-resources monitor paper-validate paper-soak candidate-batch collector model-scheduler dashboard-refresh dashboard clean

bootstrap:
	bash scripts/bootstrap.sh

check-env:
	$(PYTHON) scripts/check_env.py

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	$(PYTHON) -m pytest -q

pipeline:
	$(PYTHON) scripts/test_pipeline.py

hft:
	$(PYTHON) scripts/test_hft_pipeline.py

references:
	$(PYTHON) scripts/download_pipeline_references.py

quant-resources:
	$(PYTHON) scripts/generate_quant_resource_bundle.py
	bash scripts/download_quant_resource_bundle.sh

docs:
	$(PYTHON) scripts/download_pipeline_references.py
	$(PYTHON) scripts/generate_candidate_pipeline_docs.py
	$(PYTHON) scripts/generate_pipeline_docs.py

candidate-docs:
	$(PYTHON) scripts/generate_candidate_pipeline_docs.py
	$(PYTHON) scripts/generate_pipeline_docs.py

monitor:
	$(PYTHON) scripts/monitor_health.py

paper-validate:
	$(PYTHON) scripts/validate_paper_broker.py

paper-soak:
	$(PYTHON) scripts/run_paper_soak.py $(SOAK_ARGS)

candidate-batch:
	$(PYTHON) scripts/run_candidate_batch.py --suite shortlist

collector:
	$(PYTHON) scripts/run_data_collector.py --iterations 1 --refresh-snapshot

model-scheduler:
	$(PYTHON) scripts/run_model_scheduler.py --iterations 1 --suite shortlist --refresh-dataset

dashboard-refresh:
	$(PYTHON) scripts/refresh_control_plane.py

dashboard:
	$(VENV)/bin/streamlit run dashboard/app.py --server.headless true

clean:
	rm -rf .pytest_cache
