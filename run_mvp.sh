#!/bin/bash

echo "🔬 Research File Manager MVP"
echo "AI-Powered File Organization & Semantic Search"
echo ""
echo "Starting server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python setup_mvp.py"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if backend exists
if [ ! -f "backend/main.py" ]; then
    echo "❌ Backend not found. Please ensure the setup is complete."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# Run the application
echo "🚀 Starting Research File Manager on http://localhost:8000"
echo "📱 Open your browser and navigate to: http://localhost:8000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

python backend/main.py