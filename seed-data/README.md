# Firebase Emulator Seed Data

This directory contains seed data for Firebase emulators to provide a consistent development environment.

## Structure

- `firestore-export/` - Firestore database seed data
- `auth-export/` - Authentication users seed data
- `storage-export/` - Storage bucket seed data

## Usage

### Exporting Current Data
To export current emulator data as seed data:
```bash
firebase emulators:export ./seed-data
```

### Importing Seed Data
Seed data is automatically imported when starting emulators using:
```bash
./scripts/start-emulators.sh
```

## Sample Data

### Authentication Users
Create `auth-export/accounts.json`:
```json
{
  "users": [
    {
      "localId": "test-user-1",
      "email": "test@example.com",
      "emailVerified": true,
      "passwordHash": "fakeHash",
      "salt": "fakeSalt",
      "displayName": "Test User",
      "photoUrl": "",
      "createdAt": "1609459200000",
      "lastLoginAt": "1609459200000"
    }
  ]
}
```

### Firestore Collections
Firestore data will be automatically created when you export from a running emulator.

## Best Practices

1. Keep seed data minimal and focused on common test scenarios
2. Use meaningful IDs for easy reference in tests
3. Document any special test accounts or data structures
4. Regularly update seed data to match current schema