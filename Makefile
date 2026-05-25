.PHONY: install run-oss run-frontier eval eval-multiturn test lint typecheck

install:
	pip install -e ".[dev]"

run-oss:
	python apps/oss-assistant/app.py --server_port 7860

run-frontier:
	python apps/frontier-assistant/app.py --server_port 7861

eval:
	python eval/run_eval.py

eval-multiturn:
	python eval/run_eval_multiturn.py

test:
	pytest tests/ --cov=core --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check .

typecheck:
	mypy core/ --strict
