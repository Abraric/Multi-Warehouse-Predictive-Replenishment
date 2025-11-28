.PHONY: install data etl forecast optimize simulate run test docker clean

install:
	pip install -r requirements.txt

data:
	python src/data/synthetic_generator.py --out-dir data/raw --seed 42 --days 180 --warehouses 5 --skus 50

etl:
	python src/etl/etl.py --in-dir data/raw --out-dir data/processed

forecast:
	python src/forecast/forecast_manager.py --train --data-dir data/processed --out-dir data/forecasts --horizon 14

optimize:
	python -m src.optimization.replenishment_optimizer --snapshot data/processed/inventory_snapshot.csv --forecasts data/forecasts --horizon 14 --out output/reports/replenishment_plan_sample.csv

simulate:
	python src/simulator/transfer_simulator.py --plan output/reports/replenishment_plan_sample.csv --out output/reports/simulation_report_sample.csv

run:
	uvicorn src.api.app:app --reload --port 8000

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

docker:
	docker-compose up -d

clean:
	rm -rf data/raw/* data/processed/* data/forecasts/* output/reports/* __pycache__ .pytest_cache

