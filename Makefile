.PHONY: help install install-dev test lint format clean run-train run-eval run-predict

help:
	@echo "Available commands:"
	@echo "  make install       - Install package dependencies"
	@echo "  make install-dev   - Install package with development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run code quality checks (flake8, mypy)"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean build artifacts and cache"
	@echo "  make run-train     - Run training script"
	@echo "  make run-eval      - Run evaluation script"
	@echo "  make run-predict   - Run prediction script"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src/ocr_plate_detector --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/

run-train:
	python scripts/train.py --config config/config.yaml

run-eval:
	@echo "Usage: make run-eval MODEL=path/to/model.pt DATA=path/to/data"
	@echo "Example: make run-eval MODEL=models/plate_detection/best.pt DATA=data/test/"

run-predict:
	@echo "Usage: make run-predict IMAGE=path/to/image.jpg"
	@echo "Example: make run-predict IMAGE=data/test/sample.jpg"
