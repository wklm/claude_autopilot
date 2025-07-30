#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./emulator-restore.sh <backup-directory>"
  echo "Example: ./emulator-restore.sh ./backups/emulator_20240130_120000"
  exit 1
fi

BACKUP_DIR=$1

# Check if backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
  echo "Error: Backup directory '$BACKUP_DIR' does not exist."
  exit 1
fi

# Check if firebase-tools is installed
if ! command -v firebase &> /dev/null; then
    echo "Error: Firebase CLI not found. Please install firebase-tools."
    exit 1
fi

echo "Restoring emulator data from $BACKUP_DIR..."
firebase emulators:start --import=$BACKUP_DIR --export-on-exit

echo "Restore completed successfully!"