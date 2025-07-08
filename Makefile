# Makefile — Intelligent Complaint Analysis Project

PYTHON=python3.10
VENV=llama_venv
ACTIVATE=source $(VENV)/bin/activate

.PHONY: setup install run app test notebooks clean

setup:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && pip install -r requirements.txt

run: app

app:
	$(ACTIVATE) && streamlit run src/services/streamlit/rag_interface.py

test:
	$(ACTIVATE) && pytest tests/ -v

notebooks:
	$(ACTIVATE) && jupyter notebook notebooks/

clean:
	rm -rf __pycache__ */__pycache__ llama_venv .pytest_cache
	find . -name "*.pyc" -delete