.PHONY: install test lint typecheck example clean

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/ examples/
	uv run ruff format --check src/ tests/ examples/

format:
	uv run ruff format src/ tests/ examples/

typecheck:
	uv run pyright src/

example:
	uv run python examples/cartpole_env.py

clean:
	rm -rf .venv __pycache__ src/mjlabcpu/**/__pycache__ tests/__pycache__
	rm -rf dist .pytest_cache
