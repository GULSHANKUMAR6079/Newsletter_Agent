#!/bin/sh
# entrypoint.sh — run Alembic migrations then start Streamlit

set -e

echo "⏳ Waiting for PostgreSQL..."
until python -c "
import asyncio, asyncpg, os
async def check():
    url = os.getenv('DATABASE_URL','postgresql+asyncpg://postgres:postgres@postgres:5432/blogwriter')
    conn_url = url.replace('+asyncpg','')
    await asyncpg.connect(conn_url)
asyncio.run(check())
" 2>/dev/null; do
  sleep 2
done
echo "✅ PostgreSQL is ready."

echo "⬆️  Running Alembic migrations..."
alembic upgrade head || echo "⚠️  Alembic skipped (no migrations folder or DB issue)"

echo "🚀 Starting Streamlit..."
exec streamlit run frontend/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
