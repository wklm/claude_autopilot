#!/usr/bin/env node

const { spawn } = require('child_process');
const http = require('http');

// Color codes for terminal output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  reset: '\x1b[0m'
};

// Emulator configuration
const emulators = [
  { name: 'Emulator UI', port: 4000, path: '/' },
  { name: 'Hosting', port: 5000, path: '/' },
  { name: 'Functions', port: 5001, path: '/' },
  { name: 'Firestore', port: 8080, path: '/' },
  { name: 'Pub/Sub', port: 8085, path: '/' },
  { name: 'Realtime Database', port: 9000, path: '/.json' },
  { name: 'Auth', port: 9099, path: '/' },
  { name: 'Storage', port: 9199, path: '/' },
  { name: 'Eventarc', port: 9299, path: '/' },
  { name: 'Hub', port: 4400, path: '/' }
];

// Check if a port is accessible
function checkPort(host, port, path, timeout = 2000) {
  return new Promise((resolve) => {
    const options = {
      hostname: host,
      port: port,
      path: path,
      method: 'GET',
      timeout: timeout
    };

    const req = http.request(options, (res) => {
      resolve({ success: true, statusCode: res.statusCode });
    });

    req.on('error', (error) => {
      resolve({ success: false, error: error.message });
    });

    req.on('timeout', () => {
      req.destroy();
      resolve({ success: false, error: 'Timeout' });
    });

    req.end();
  });
}

// Check Firebase CLI installation
function checkFirebaseCLI() {
  return new Promise((resolve) => {
    const firebase = spawn('firebase', ['--version'], { shell: true });
    
    let output = '';
    firebase.stdout.on('data', (data) => {
      output += data.toString();
    });

    firebase.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, version: output.trim() });
      } else {
        resolve({ success: false, error: 'Firebase CLI not found' });
      }
    });

    firebase.on('error', () => {
      resolve({ success: false, error: 'Firebase CLI not found' });
    });
  });
}

// Main health check function
async function runHealthCheck() {
  console.log('🔍 Firebase Emulator Health Check\n');

  // Check Firebase CLI
  console.log('Checking Firebase CLI...');
  const cliCheck = await checkFirebaseCLI();
  if (cliCheck.success) {
    console.log(`${colors.green}✅ Firebase CLI installed${colors.reset} (${cliCheck.version})`);
  } else {
    console.log(`${colors.red}❌ Firebase CLI not found${colors.reset}`);
    console.log('   Run: npm install -g firebase-tools');
  }

  console.log('\nChecking emulator ports...');
  
  let allHealthy = true;
  const results = [];

  for (const emulator of emulators) {
    const result = await checkPort('localhost', emulator.port, emulator.path);
    
    if (result.success) {
      console.log(`${colors.green}✅ ${emulator.name}${colors.reset} (port ${emulator.port})`);
      results.push({ ...emulator, status: 'running' });
    } else {
      console.log(`${colors.red}❌ ${emulator.name}${colors.reset} (port ${emulator.port}) - ${result.error}`);
      results.push({ ...emulator, status: 'not running', error: result.error });
      allHealthy = false;
    }
  }

  // Summary
  console.log('\n📊 Summary:');
  const runningCount = results.filter(r => r.status === 'running').length;
  console.log(`   Running: ${runningCount}/${emulators.length} emulators`);

  if (allHealthy) {
    console.log(`\n${colors.green}🎉 All emulators are healthy!${colors.reset}`);
    console.log('   Emulator UI: http://localhost:4000');
    process.exit(0);
  } else {
    console.log(`\n${colors.yellow}⚠️  Some emulators are not running${colors.reset}`);
    console.log('   Run: ./scripts/start-emulators.sh');
    process.exit(1);
  }
}

// Run the health check
runHealthCheck().catch(console.error);