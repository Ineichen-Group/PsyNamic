PORT=$1

if [ -z "$PORT" ]; then
  echo "Usage: $0 PORT"
  exit 1
fi

# Find containers by port
CONTAINER_IDS=$(docker ps -q --filter "publish=${PORT}")

if [ -n "$CONTAINER_IDS" ]; then
    echo "Stopping and removing the existing containers using port ${PORT}..."
    docker stop $CONTAINER_IDS
    docker rm $CONTAINER_IDS
else
    echo "No containers using port ${PORT} found."
fi