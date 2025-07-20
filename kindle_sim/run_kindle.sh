#!/bin/bash

echo "Running model in Kindle simulation container..."
docker run -it --cpus="0.3" --memory="256m" kindle-sim /bin/bash