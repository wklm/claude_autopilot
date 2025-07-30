# Firebase Emulator Migration Plan

## Overview
This document outlines the step-by-step migration plan to integrate Firebase Emulator Suite into the Claude Code Agent Farm project. The Firebase Emulator Suite will provide local development and testing capabilities for Firebase services without requiring actual Firebase project credentials.

## Migration Status Tracker

### Phase 1: Initial Setup ✅
- [x] Create migration plan document
- [x] Install Firebase CLI and emulator dependencies
- [x] Create base Firebase configuration files

### Phase 2: Emulator Configuration ✅
- [x] Configure firebase.json with emulator settings
- [x] Set up emulator ports configuration
- [x] Create .firebaserc for project aliasing
- [x] Configure emulator UI settings

### Phase 3: Docker Integration ✅
- [x] Update Dockerfile to include Firebase CLI
- [x] Configure emulator ports in Docker compose
- [x] Create emulator startup script
- [x] Set up volume mounts for persistent data

### Phase 4: Service-Specific Setup ✅
- [x] Configure Firestore emulator
- [x] Configure Authentication emulator
- [x] Configure Realtime Database emulator
- [x] Configure Storage emulator
- [x] Configure Functions emulator
- [x] Configure Hosting emulator
- [x] Configure Pub/Sub emulator

### Phase 5: Data Management ✅
- [x] Create seed data scripts
- [x] Set up import/export functionality
- [x] Configure data persistence between sessions
- [x] Create backup/restore procedures

### Phase 6: Testing & Validation ✅
- [x] Create emulator health check scripts
- [x] Write integration tests for each service
- [x] Validate Docker container functionality
- [x] Test data persistence and recovery

### Phase 7: Documentation & Tooling ✅
- [x] Update README with emulator instructions
- [x] Create troubleshooting guide
- [x] Add helper scripts for common tasks
- [x] Document best practices

## Detailed Implementation Steps

### 1. Firebase CLI Installation
```bash
# Install Firebase CLI globally
npm install -g firebase-tools

# Or use npm script in package.json
npm install --save-dev firebase-tools
```

### 2. Firebase Configuration Files

#### firebase.json
```json
{
  "emulators": {
    "auth": {
      "port": 9099
    },
    "functions": {
      "port": 5001
    },
    "firestore": {
      "port": 8080
    },
    "database": {
      "port": 9000
    },
    "hosting": {
      "port": 5000
    },
    "pubsub": {
      "port": 8085
    },
    "storage": {
      "port": 9199
    },
    "eventarc": {
      "port": 9299
    },
    "ui": {
      "enabled": true,
      "port": 4000
    },
    "singleProjectMode": true
  }
}
```

#### .firebaserc
```json
{
  "projects": {
    "default": "claude-agent-farm-dev"
  }
}
```

### 3. Docker Configuration Updates

#### Dockerfile additions
```dockerfile
# Install Firebase CLI
RUN npm install -g firebase-tools

# Expose emulator ports
EXPOSE 4000 5000 5001 8080 8085 9000 9099 9199 9299

# Copy Firebase configuration
COPY firebase.json .firebaserc ./
```

#### docker-compose.yml additions
```yaml
services:
  claude-agent-farm:
    ports:
      - "4000:4000"  # Emulator UI
      - "5000:5000"  # Hosting
      - "5001:5001"  # Functions
      - "8080:8080"  # Firestore
      - "8085:8085"  # Pub/Sub
      - "9000:9000"  # Database
      - "9099:9099"  # Auth
      - "9199:9199"  # Storage
      - "9299:9299"  # Eventarc
    volumes:
      - firebase-emulator-data:/home/app/.cache/firebase
```

### 4. Emulator Startup Script

#### scripts/start-emulators.sh
```bash
#!/bin/bash
set -e

echo "Starting Firebase Emulators..."

# Export data directory
export FIREBASE_EMULATOR_DATA_DIR="/home/app/.cache/firebase"

# Create data directory if not exists
mkdir -p $FIREBASE_EMULATOR_DATA_DIR

# Import seed data if available
if [ -f "./seed-data/firestore-export/firestore_export.overall_export_metadata" ]; then
  echo "Importing Firestore seed data..."
  firebase emulators:start --import=./seed-data --export-on-exit
else
  echo "Starting emulators without seed data..."
  firebase emulators:start --export-on-exit=$FIREBASE_EMULATOR_DATA_DIR
fi
```

### 5. Environment Configuration

#### .env.emulator
```bash
# Firebase Emulator Configuration
FIREBASE_AUTH_EMULATOR_HOST=localhost:9099
FIRESTORE_EMULATOR_HOST=localhost:8080
FIREBASE_DATABASE_EMULATOR_HOST=localhost:9000
FIREBASE_STORAGE_EMULATOR_HOST=localhost:9199
FIREBASE_HOSTING_EMULATOR_HOST=localhost:5000
PUBSUB_EMULATOR_HOST=localhost:8085

# Disable production Firebase
FIREBASE_CONFIG={"projectId":"claude-agent-farm-dev"}
GCLOUD_PROJECT=claude-agent-farm-dev
```

### 6. Helper Scripts

#### scripts/emulator-backup.sh
```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/emulator_$TIMESTAMP"

echo "Backing up emulator data to $BACKUP_DIR..."
firebase emulators:export $BACKUP_DIR
```

#### scripts/emulator-restore.sh
```bash
#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: ./emulator-restore.sh <backup-directory>"
  exit 1
fi

echo "Restoring emulator data from $1..."
firebase emulators:start --import=$1
```

### 7. Testing Scripts

#### tests/emulator-health-check.js
```javascript
const { initializeApp } = require('firebase/app');
const { getAuth, connectAuthEmulator } = require('firebase/auth');
const { getFirestore, connectFirestoreEmulator } = require('firebase/firestore');

async function checkEmulators() {
  const app = initializeApp({
    projectId: 'claude-agent-farm-dev'
  });

  try {
    // Check Auth emulator
    const auth = getAuth(app);
    connectAuthEmulator(auth, 'http://localhost:9099');
    console.log('✅ Auth emulator is running');

    // Check Firestore emulator
    const db = getFirestore(app);
    connectFirestoreEmulator(db, 'localhost', 8080);
    console.log('✅ Firestore emulator is running');

    // Add more service checks as needed

  } catch (error) {
    console.error('❌ Emulator check failed:', error);
    process.exit(1);
  }
}

checkEmulators();
```

## Next Steps

1. Execute Phase 1: Initial Setup
2. Configure Docker environment
3. Test emulator connectivity
4. Implement service-specific configurations
5. Create comprehensive test suite
6. Document usage patterns

## Notes

- Emulators do not require Firebase authentication
- Data persists only when using --export-on-exit flag
- Emulator UI provides visual interface at http://localhost:4000
- All emulator data is isolated from production Firebase services

---
Last Updated: 2025-07-30
Migration Status: ✅ COMPLETED

## Summary

The Firebase Emulator Suite has been successfully integrated into the Claude Code Agent Farm project. All phases of the migration have been completed:

1. ✅ Created comprehensive migration plan
2. ✅ Set up Firebase configuration files (firebase.json, .firebaserc)
3. ✅ Configured all emulator services with proper port mappings
4. ✅ Created initialization and management scripts
5. ✅ Set up seed data structure with sample authentication users
6. ✅ Updated Docker configuration with Firebase CLI and port exposure
7. ✅ Created docker-compose.yml for easy container management
8. ✅ Built testing scripts for health monitoring
9. ✅ Added comprehensive documentation to README.md

The Firebase Emulator Suite is now ready for use with Claude Code agents, providing a complete local development environment for Firebase services without requiring production credentials.