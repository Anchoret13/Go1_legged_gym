#!/bin/bash

# Function to check if TeamViewer is running
is_teamviewer_running() {
    ps aux | grep -v grep | grep TeamViewer > /dev/null 2>&1
}

# Function to start TeamViewer
start_teamviewer() {
    teamviewer & # Start TeamViewer in the background
    if [ $? -eq 0 ]; then
        echo "TeamViewer restarted successfully at $(date)"
    else
        echo "Failed to restart TeamViewer at $(date)" >&2
    fi
}

# Main loop to monitor TeamViewer
while true; do
    if ! is_teamviewer_running; then
        echo "TeamViewer is not running. Restarting..."
        start_teamviewer
    fi
    sleep 10 # Check every 10 seconds
done