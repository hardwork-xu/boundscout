PYTHON ?= .venv/bin/python

.PHONY: install demo test lint typecheck benchmark analyze build check docs
install:
	$(PYTHON) -m pip install --require-hashes -r requirements.lock
	$(PYTHON) -m pip install --no-deps --no-build-isolation -e .
demo:
	$(PYTHON) -m boundscout demo
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
typecheck:
	$(PYTHON) -m mypy src/boundscout
benchmark:
	$(PYTHON) benchmark.py --config configs/smoke.json --output results/raw/smoke.json
analyze:
	$(PYTHON) scripts/analyze.py --input results/raw/full.json
build:
	$(PYTHON) -m build --no-isolation
docs:
	$(PYTHON) scripts/check_docs.py
check: test lint typecheck docs build
