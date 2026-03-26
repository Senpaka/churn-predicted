
install:
	pip install -r requirements.txt

train:
	python scripts/train_pipeline.py

test:
	pytest api/tests

run:
	uvicorn api.app:app --reload
