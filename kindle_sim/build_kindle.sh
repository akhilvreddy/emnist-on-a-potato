#!/bin/bash

echo "Building Kindle simulation container..."

docker builder prune -f
docker build -t kindle-sim ./kindle_sim

echo "Build completed. Image built as 'kindle-sim'"