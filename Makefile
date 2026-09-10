.PHONY: setup data eval reproduce reproduce-nocache test label judge clean

# ============================================================
# Hiver SDE Intern Take-Home: Makefile
# ============================================================

PYTHON = python
VENV_PYTHON = .venv/Scripts/python

setup:
	python -m venv .venv
	$(VENV_PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.ingest

label:
	$(PYTHON) scripts/label_tool.py --sample data/sample/brand_sample.parquet --output data/golden/golden_v1.jsonl --count 200

eval:
	$(PYTHON) -m src.evaluate --golden data/golden/golden_v1.jsonl --sample data/sample/brand_sample.parquet --output results --use-cache

judge:
	$(PYTHON) scripts/calibrate_judge.py --golden data/golden/golden_v1.jsonl --output results/judge_agreement.json --size 60

reproduce: ## <8 min, uses committed sample + cached LLM calls
	@echo "Reproducing headline results from committed data and cache..."
	$(PYTHON) -m src.evaluate --use-cache
	@echo "Results written to results/metrics.json"
	@echo "Confusion matrix: results/confusion_matrix.png"
	@echo "Judge agreement: results/judge_agreement.json"

reproduce-nocache: ## Full live run (~25 min, requires API keys)
	@echo "Running full live pipeline with API calls..."
	$(PYTHON) -m src.evaluate
	@echo "Full live results written to results/"

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf results/*.json results/*.png data/index/*.pkl data/cache/*.json
