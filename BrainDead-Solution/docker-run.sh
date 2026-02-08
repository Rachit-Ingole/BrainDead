#!/bin/bash
set -e

IMAGE_NAME="braindead-solution"
CONTAINER_NAME="braindead-solution-app"

function build() {
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
    echo "Build complete!"
}

function run() {
    echo "Starting BrainDead-Solution container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p 8501:8501 \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/checkpoints:/app/checkpoints \
        $IMAGE_NAME
    echo "Container started! Access at http://localhost:8501"
}

function stop() {
    echo "Stopping container..."
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
    echo "Container stopped and removed."
}

function logs() {
    docker logs -f $CONTAINER_NAME
}

function shell() {
    docker exec -it $CONTAINER_NAME /bin/bash
}

function clean() {
    echo "Cleaning up Docker resources..."
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
    docker rmi $IMAGE_NAME || true
    docker system prune -f
    echo "Cleanup complete."
}

function help() {
    echo "BrainDead-Solution Docker Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image"
    echo "  run      Run the container"
    echo "  stop     Stop and remove the container"
    echo "  logs     Show container logs"
    echo "  shell    Open shell in container"
    echo "  clean    Remove image and clean up"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 run"
    echo "  $0 logs"
    echo "  $0 shell"
}

case "${1:-help}" in
    build) build ;;
    run) run ;;
    stop) stop ;;
    logs) logs ;;
    shell) shell ;;
    clean) clean ;;
    help) help ;;
    *) echo "Unknown command: $1"; help; exit 1 ;;
esac