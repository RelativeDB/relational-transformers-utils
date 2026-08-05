.PHONY: test lint docs

test:
	python -m pytest

lint:
	python -m ruff check relational_transformers_utils relben tests

docs:
	python -m sphinx -W -c docs -d docs/_build/doctrees -b html . docs/_build/html
