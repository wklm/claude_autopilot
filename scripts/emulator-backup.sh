#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/emulator_$TIMESTAMP"

# Create backups directory if it doesn't exist
mkdir -p ./backups

echo "Backing up emulator data to $BACKUP_DIR..."

# Check if firebase-tools is installed
if ! command -v firebase &> /dev/null; then
    echo "Error: Firebase CLI not found. Please install firebase-tools."
    exit 1
fi

# Export emulator data
firebase emulators:export $BACKUP_DIR

echo "Backup completed successfully!"
echo "Backup location: $BACKUP_DIR"