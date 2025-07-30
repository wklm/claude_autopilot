#!/bin/bash
set -e

echo "Starting Firebase Emulators..."

# Export data directory
export FIREBASE_EMULATOR_DATA_DIR="/home/app/.cache/firebase"

# Create data directory if not exists
mkdir -p $FIREBASE_EMULATOR_DATA_DIR

# Check if firebase-tools is installed
if ! command -v firebase &> /dev/null; then
    echo "Firebase CLI not found. Installing..."
    npm install -g firebase-tools
fi

# Import seed data if available
if [ -f "./seed-data/firestore-export/firestore_export.overall_export_metadata" ]; then
  echo "Importing Firestore seed data..."
  firebase emulators:start --import=./seed-data --export-on-exit
else
  echo "Starting emulators without seed data..."
  firebase emulators:start --export-on-exit=$FIREBASE_EMULATOR_DATA_DIR
fi