#!/bin/bash

# Start Tailwind CSS watcher in background
npx tailwindcss -i ./assets/main.css -o ./assets/output.css --watch &
TAILWIND_PID=$!

# Start Dioxus dev server
dx serve --port 3000

# Kill Tailwind when dx serve exits
kill $TAILWIND_PID
