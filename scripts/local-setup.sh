#!/bin/bash


set -e

echo "🔧 Setting up local development environment..."

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "📁 Creating directories..."
mkdir -p data/models data/raw data/processed output logs temp

echo "🗄️ Initializing database..."
python -m src.database.migrate

echo "✅ Local setup completed!"
echo "🚀 To start the development server:"
echo "   source venv/bin/activate"
echo "   uvicorn app:app --reload --host 0.0.0.0 --port 8000"
echo "🌐 Then visit: http://localhost:8000"
