.PHONY: setup fetch preprocess train evaluate serve test lint clean docker-up docker-down

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

fetch:
	python -m src.data.fetch_data

preprocess:
	python -m src.data.preprocess

train:
	python -m src.train.train

evaluate:
	python -m src.evaluate.evaluate

serve:
	uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

pipeline: fetch preprocess train evaluate
	@echo "Full pipeline complete."

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf models/ data/raw/*.csv data/processed/*.csv mlruns/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
