#!/bin/bash


set -e

echo "🚀 Starting deployment to Fly.io..."

if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "curl -L https://fly.io/install.sh | sh"
    exit 1
fi

if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run: flyctl auth login"
    exit 1
fi

if ! flyctl apps list | grep -q "weed-diversity-analyzer"; then
    echo "📱 Creating Fly.io app..."
    flyctl apps create weed-diversity-analyzer --org personal
fi

if ! flyctl apps list | grep -q "weed-diversity-analyzer-db"; then
    echo "🗄️ Creating PostgreSQL database..."
    flyctl postgres create --name weed-diversity-analyzer-db --org personal --region nrt
fi

DB_URL=$(flyctl postgres connect --app weed-diversity-analyzer-db --command "echo \$DATABASE_URL" 2>/dev/null || echo "")

if [ -z "$DB_URL" ]; then
    echo "⚠️ Could not get database URL automatically. Please set it manually after deployment."
    DB_URL="postgresql://postgres:password@weed-diversity-analyzer-db.internal:5432/postgres"
fi

echo "🔐 Setting secrets..."
flyctl secrets set DATABASE_URL="$DB_URL" --app weed-diversity-analyzer

echo "🚀 Deploying application..."
flyctl deploy --app weed-diversity-analyzer

echo "✅ Deployment completed!"
echo "🌐 Your app should be available at: https://weed-diversity-analyzer.fly.dev"
echo "📊 Check status: flyctl status --app weed-diversity-analyzer"
echo "📝 View logs: flyctl logs --app weed-diversity-analyzer"
