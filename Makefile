format:
	poetry run black .
	poetry run ruff check --fix .
	poetry run mypy .
