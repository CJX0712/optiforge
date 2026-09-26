PY := .venv/Scripts/python        # Windows
# PY := .venv/bin/python          # Linux/macOS

venv:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning

demo:
	PYTHONPATH=. $(PY) examples/run_demo.py

cli:
	$(PY) -m optiforge.cli --problem all --size 40 --seed 42 --output benchmark.json

fallback:
	OPTIFORGE_BACKEND=fallback $(PY) -m optiforge.cli --problem all --size 12 --seed 7

lock:
	$(PY) -m pip freeze > requirements.lock.txt

.PHONY: venv test demo cli fallback lock
