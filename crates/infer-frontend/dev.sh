#!/bin/bash

# Start the lockfile-pinned Tailwind CSS watcher in background.
npm run tailwind &
TAILWIND_PID=$!

# Start Dioxus dev server
dx serve --port 3000

# Kill Tailwind when dx serve exits
kill $TAILWIND_PID
