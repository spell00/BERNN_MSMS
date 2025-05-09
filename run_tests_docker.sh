#!/bin/bash
set -e

# Build the test Docker image
docker build -t bernn-test -f Dockerfile.test .

# Run tests in Docker container
docker run --rm -v $(pwd)/coverage.xml:/app/coverage.xml bernn-test

echo "Tests completed. Coverage report saved to coverage.xml"
echo "To view HTML coverage report, run: pytest --cov=bernn --cov-report=html tests/" 