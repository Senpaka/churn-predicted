
APP_NAME = churn-container

install:
	pip install -r requirements.txt

train:
	python scripts/train_pipeline.py

test:
	pytest api/tests

run-local:
	uvicorn api.app:app --reload

build:
	docker build -t churn-prediction-app .

run-docker:
	docker run -p 8000:8000 --rm --name $(APP_NAME) churn-prediction-app

build-run:
	docker build -t churn-prediction-app .
	docker run -p 8000:8000 --rm --name $(APP_NAME) churn-prediction-app

stop:
	-docker stop $(APP_NAME)
	-docker rm $(APP_NAME)

restart: stop build-run