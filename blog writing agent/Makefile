.PHONY: install run test lint docker-up docker-down clean help

## ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Blog Writing Agent — Makefile commands"
	@echo ""
	@echo "  make install      Install Python dependencies"
	@echo "  make run          Start Streamlit app locally"
	@echo "  make test         Run pytest test suite"
	@echo "  make lint         Run ruff linter"
	@echo "  make docker-up    Build and start app + postgres via Docker Compose"
	@echo "  make docker-down  Stop and remove containers"
	@echo "  make clean        Remove __pycache__, .pytest_cache, output/"
	@echo ""

## ── Local development ────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

run:
	streamlit run frontend/app.py \
	  --server.port=8501 \
	  --browser.gatherUsageStats=false

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check app/ frontend/ tests/ --fix

## ── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d
	@echo "✅ App running at http://localhost:8501"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

## ── Database migrations ──────────────────────────────────────────────────────
migrate:
	alembic upgrade head

migrate-new:
	alembic revision --autogenerate -m "$(msg)"

## ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf output/*.md output/images/
	@echo "🧹 Cleaned."
