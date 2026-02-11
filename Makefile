.PHONY: setup lint format test pipeline report all clean

PY := python

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

test:
	$(PY) -m pytest -q

pipeline:
	$(PY) scripts/01_fetch_universe_sec.py
	$(PY) scripts/02_fetch_prices.py
	$(PY) scripts/03_fetch_events.py
	$(PY) scripts/04_event_study_mvp.py
	$(PY) scripts/05_event_study_robustness.py
	$(PY) scripts/06_build_model_dataset.py
	$(PY) scripts/07b_train_cv_lgbm.py

backtest:
	$(PY) scripts/09_backtest_costed.py
	$(PY) scripts/10_backtest_engine.py

report:
	$(PY) scripts/11_build_report.py

all: lint test pipeline backtest report

clean:
	@echo "Cleaning reports + processed artifacts (keep raw/cache)..."
	-rm -rf reports/*
	-rm -rf data/processed/*
