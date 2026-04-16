#!/usr/bin/env bash
# Build and run BEQ Designer in Docker.
#
# Usage:
#   ./run.sh                    # Build and launch interactive menu
#   ./run.sh extract            # Build and extract LFE WAVs
#   ./run.sh -p extract         # Git pull first, then build and run
#   ./run.sh -nb                # Skip build, run with existing image
#   ./run.sh --help             # Show all subcommands
#
# Requires docker-compose.yml in this directory. Copy from
# docker-compose.example.yml and edit volume paths for your setup.

set -e
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo ""
    echo "ERROR: $COMPOSE_FILE not found."
    echo ""
    echo "Copy the example and edit it for your setup:"
    echo ""
    echo "  cp docker-compose.example.yml docker-compose.yml"
    echo "  \$EDITOR docker-compose.yml"
    echo ""
    echo "You'll need to set:"
    echo "  • BEQ shared directory volume mount"
    echo "  • Media library volume mounts"
    echo ""
    exit 1
fi

# Parse flags.
PULL=false
BUILD=true
while [ $# -gt 0 ]; do
    case "$1" in
        -p)  PULL=true; shift ;;
        -nb) BUILD=false; shift ;;
        *)   break ;;
    esac
done

if [ "$PULL" = true ] && [ "$BUILD" = false ]; then
    echo "WARNING: -p ignored — not pulling because -nb (no build) was set"
    PULL=false
fi

if [ "$PULL" = true ]; then
    echo "Pulling latest changes..."
    git -C .. pull
fi

if [ "$BUILD" = true ]; then
    # Write git version info for the Docker build (container has no .git).
    GIT_BRANCH=$(git -C .. rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GIT_COMMIT=$(git -C .. rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "{\"branch\": \"$GIT_BRANCH\", \"commit\": \"$GIT_COMMIT\"}" > ../build/version.json

    docker compose -f "$COMPOSE_FILE" build
fi

docker compose -f "$COMPOSE_FILE" run --remove-orphans beq-designer "$@"
