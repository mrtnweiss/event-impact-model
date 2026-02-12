.PHONY: setup lint format test pipeline backtest robustness report all clean build

PY := python

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q

pipeline:
	$(PY) scripts/01_fetch_universe_sec.py
	$(PY) scripts/02_fetch_prices.py
	$(PY) scripts/03_fetch_events.py
	$(PY) scripts/04_event_study_mvp.py
	$(PY) scripts/05_event_study_robustness.py
	$(PY) scripts/06_build_model_dataset.py
	$(PY) scripts/07_train_cv_baselines.py
	$(PY) scripts/07b_train_cv_lgbm.py

backtest:
	$(PY) scripts/09_backtest_costed.py
	$(PY) scripts/10_backtest_engine.py

robustness:
	$(PY) scripts/12_robustness_sanity.py
	$(PY) scripts/13_robustness_subsamples.py
	$(PY) scripts/14_robustness_within_day_shuffle.py

report:
	$(PY) scripts/11_build_report.py

all: lint format test pipeline backtest robustness report

build:
	$(PY) -m build

clean:
	@echo "Cleaning reports + processed artifacts (keep raw/cache)..."
	-rm -rf reports/*
	-rm -rf data/processed/*
