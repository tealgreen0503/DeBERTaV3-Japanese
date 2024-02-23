format:
	poetry run ruff format .
	poetry run ruff check --fix .
	poetry run mypy .
