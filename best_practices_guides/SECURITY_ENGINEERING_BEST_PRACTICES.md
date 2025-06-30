# The Comprehensive Guide to Security Engineering and Defensive Programming (mid-2025)

This guide synthesizes modern security practices across application security, network defense, reverse engineering, and hardware mitigations. It emphasizes free, open-source tools and practical patterns that scale from individual projects to enterprise deployments.

## Prerequisites & Philosophy

This guide assumes basic programming knowledge and a Linux/Unix environment. All tools mentioned are FOSS (Free and Open Source Software) unless explicitly noted. Security is not a checkbox—it's a continuous process of defense-in-depth.

**Core Principles:**
- **Assume Breach**: Design systems expecting compromise
- **Least Privilege**: Grant minimal necessary permissions
- **Defense in Depth**: Multiple overlapping security layers
- **Shift Left**: Integrate security early in development
- **Zero Trust**: Never trust, always verify

---

## 1. Application Security: Building Secure Software

Modern application security starts with secure coding practices and automated testing throughout the development lifecycle.

### 1.1 Dependency Scanning and Supply Chain Security

The 2025 landscape has seen sophisticated supply chain attacks. Your first line of defense is knowing what's in your codebase.

#### ✅ DO: Implement Comprehensive Dependency Scanning

**1. Use Multiple Scanners for Coverage**

```bash
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # 1. Trivy - Fast, comprehensive scanner
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      # 2. OSV-Scanner - Google's unified vulnerability database
      - name: Run OSV-Scanner
        uses: google/osv-scanner-action@v1
        with:
          scan-args: |-
            --verbosity=3
            --format=sarif
            --output=osv-results.sarif
            ./
      
      # 3. Grype - Anchore's vulnerability scanner
      - name: Run Grype scan
        uses: anchore/scan-action@v3
        with:
          path: '.'
          output-format: sarif
          fail-build: true
          severity-cutoff: high
```

**2. Lock Files and Reproducible Builds**

```toml
# Cargo.toml for Rust projects - use exact versions
[dependencies]
tokio = "=1.45.0"  # Exact version, not "^1.45.0"
```

```python
# pip-tools for Python - generate locked requirements
# requirements.in
django>=5.2,<6.0
requests>=2.35.0

# Generate locked file
$ pip-compile --generate-hashes requirements.in
```

#### ✅ DO: Implement Software Bill of Materials (SBOM)

```bash
# Generate SBOM using syft (SPDX/CycloneDX formats)
syft packages dir:. -o spdx-json > sbom.spdx.json
syft packages dir:. -o cyclonedx-json > sbom.cyclonedx.json

# Verify against known vulnerabilities
grype sbom:./sbom.spdx.json
```

### 1.2 Static Application Security Testing (SAST)

Modern SAST tools use sophisticated analysis to find bugs before runtime.

#### ✅ DO: Layer Multiple SAST Tools

**1. Semgrep - Pattern-Based Analysis**

```yaml
# .semgrep/rules/custom-security.yml
rules:
  - id: hardcoded-secret
    patterns:
      - pattern-either:
          - pattern: $KEY = "..."
          - pattern: $KEY = '...'
      - metavariable-regex:
          metavariable: $KEY
          regex: '.*(password|secret|token|api_key).*'
      - metavariable-regex:
          metavariable: $VALUE
          regex: '.{10,}'
    message: Potential hardcoded secret found
    languages: [python, javascript, go]
    severity: ERROR

# Run with custom and community rules
$ semgrep --config=.semgrep/rules --config=auto .
```

**2. CodeQL - Semantic Code Analysis**

```yaml
# .github/workflows/codeql.yml
name: "CodeQL"
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  analyze:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        language: [ 'cpp', 'python', 'javascript', 'go' ]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
        queries: security-and-quality
        
    - name: Autobuild
      uses: github/codeql-action/autobuild@v3
      
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
```

**3. Language-Specific Tools**

```bash
# Rust - built-in security with cargo
cargo audit          # Check for vulnerable dependencies
cargo clippy         # Linting with security checks
cargo fuzz           # Fuzzing harness

# Go - comprehensive security scanning
gosec -fmt sarif -out gosec-results.sarif ./...
staticcheck ./...
govulncheck ./...    # Go vulnerability database check

# Python - layered analysis
bandit -r . -f json -o bandit-results.json
safety check --json
mypy --strict .      # Type checking prevents many vulnerabilities
```

### 1.3 Fuzzing: Finding the Unfindable

Fuzzing has evolved significantly. Modern fuzzers use coverage-guided, grammar-aware, and AI-enhanced techniques.

#### ✅ DO: Implement Continuous Fuzzing

**1. AFL++ - The Gold Standard**

```c
// fuzz_target.c - Harness for AFL++
#include <stdint.h>
#include <string.h>

// Your parsing function to test
int parse_protocol(const uint8_t* data, size_t size);

// AFL++ persistent mode for 100x speedup
__AFL_FUZZ_INIT();

int main() {
    // Setup code here
    
    __AFL_INIT();
    
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;
    
    while (__AFL_LOOP(10000)) {
        int len = __AFL_FUZZ_TESTCASE_LEN;
        
        // Call your target function
        parse_protocol(buf, len);
    }
    
    return 0;
}
```

```bash
# Compile with AFL++ instrumentation
AFL_USE_ASAN=1 afl-clang-fast++ -g -O2 fuzz_target.c -o fuzz_target

# Create corpus
mkdir input_corpus
echo "test" > input_corpus/seed1

# Run fuzzer with advanced mutations
AFL_IMPORT_FIRST=1 AFL_CUSTOM_MUTATOR_LIBRARY=radamsa.so \
  afl-fuzz -i input_corpus -o findings -M main -- ./fuzz_target
```

**2. LibFuzzer Integration with Sanitizers**

```cpp
// libfuzzer_target.cc
#include <stdint.h>
#include <stddef.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Prevent empty input
    if (size == 0) return 0;
    
    // Your code here - this example tests JSON parsing
    try {
        auto doc = json::parse(data, data + size);
        // Process the document
    } catch (...) {
        // Catching exceptions is fine in fuzzing
    }
    
    return 0;
}
```

```bash
# Compile with all sanitizers
clang++ -g -O1 -fsanitize=fuzzer,address,undefined \
    -fno-omit-frame-pointer libfuzzer_target.cc -o fuzzer

# Run with corpus and dictionary
./fuzzer corpus/ -dict=json.dict -jobs=8 -workers=4
```

**3. OSS-Fuzz Integration**

```python
# oss-fuzz/build.sh for Python projects
#!/bin/bash -eu

pip3 install atheris

# Build fuzzer with coverage instrumentation
compile_python_fuzzer $SRC/fuzz_json.py

# Add seed corpus
zip -j $OUT/fuzz_json_seed_corpus.zip $SRC/corpus/*
```

```python
# fuzz_json.py - Python fuzzing with Atheris
import atheris
import sys
import json

@atheris.instrument_func
def TestOneInput(data):
    try:
        obj = json.loads(data.decode('utf-8'))
        # Your processing logic here
        process_json(obj)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Expected errors are fine
        pass
    except Exception as e:
        # Unexpected errors indicate bugs
        raise e

atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
```

### 1.4 Runtime Security and Sandboxing

#### ✅ DO: Implement Multiple Layers of Runtime Protection

**1. Seccomp-BPF for System Call Filtering**

```c
// seccomp_sandbox.c
#include <seccomp.h>
#include <unistd.h>

void enable_sandbox() {
    scmp_filter_ctx ctx;
    
    // Whitelist approach - deny all, then allow specific syscalls
    ctx = seccomp_init(SCMP_ACT_KILL);
    
    // Allow basic operations
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    
    // Conditional rules - only allow writing to stdout/stderr
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_LE, 2));
    
    // Apply the filter
    seccomp_load(ctx);
    seccomp_release(ctx);
}
```

**2. Landlock LSM for Filesystem Isolation (Linux 5.13+)**

```c
// landlock_sandbox.c
#include <linux/landlock.h>
#include <sys/syscall.h>

void restrict_filesystem() {
    // Create ruleset
    struct landlock_ruleset_attr ruleset_attr = {
        .handled_access_fs = LANDLOCK_ACCESS_FS_READ_FILE |
                            LANDLOCK_ACCESS_FS_READ_DIR,
    };
    
    int ruleset_fd = syscall(SYS_landlock_create_ruleset,
                            &ruleset_attr, sizeof(ruleset_attr), 0);
    
    // Allow read access to /usr
    struct landlock_path_beneath_attr path_beneath = {
        .allowed_access = LANDLOCK_ACCESS_FS_READ_FILE |
                         LANDLOCK_ACCESS_FS_READ_DIR,
        .parent_fd = open("/usr", O_PATH | O_CLOEXEC),
    };
    
    syscall(SYS_landlock_add_rule, ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH, &path_beneath, 0);
    
    // Enforce restrictions
    syscall(SYS_landlock_restrict_self, ruleset_fd, 0);
}
```

---

## 2. Network Security: Perimeter and Beyond

Network security in 2025 embraces zero-trust principles while maintaining defense-in-depth.

### 2.1 Modern Firewall Architecture with nftables

nftables has fully replaced iptables. Here's production-ready configuration:

```bash
#!/usr/sbin/nft -f
# /etc/nftables.conf

# Flush existing rules
flush ruleset

# Define variables
define WAN_IFC = eth0
define LAN_IFC = eth1
define SSH_PORT = 22
define ALLOWED_PORTS = { 80, 443 }

# Create tables
table inet filter {
    # Connection tracking
    chain conntrack {
        ct state established,related accept
        ct state invalid drop
    }
    
    # DDoS protection
    chain ratelimit {
        # Limit new connections
        tcp flags syn tcp dport $SSH_PORT \
            meter ssh-meter { ip saddr timeout 60s limit rate 3/minute } accept
        
        # SYNFLOOD protection
        tcp flags & (syn|ack) == syn \
            meter synflood { ip saddr timeout 10s limit rate 50/second } accept
    }
    
    # Input chain
    chain input {
        type filter hook input priority 0; policy drop;
        
        # Apply connection tracking
        jump conntrack
        
        # Accept loopback
        iif lo accept
        
        # ICMP rate limiting
        meta l4proto icmp icmp type { echo-request } \
            limit rate 5/second accept
        
        # Apply rate limits
        jump ratelimit
        
        # Accept established services
        tcp dport $ALLOWED_PORTS accept
        
        # Log and drop
        log prefix "[nftables-drop] " level info
        counter drop
    }
    
    # Port knocking implementation
    set knock_stage_1 {
        type ipv4_addr
        timeout 5s
    }
    
    set knock_stage_2 {
        type ipv4_addr
        timeout 5s
    }
    
    set knock_authenticated {
        type ipv4_addr
        timeout 3600s
    }
    
    chain port_knocking {
        # Stage 1: Port 1234
        tcp dport 1234 add @knock_stage_1 { ip saddr }
        
        # Stage 2: Port 2345 (only if completed stage 1)
        tcp dport 2345 ip saddr @knock_stage_1 \
            add @knock_stage_2 { ip saddr }
        
        # Stage 3: Port 3456 (complete sequence)
        tcp dport 3456 ip saddr @knock_stage_2 \
            add @knock_authenticated { ip saddr }
        
        # Allow SSH for authenticated IPs
        tcp dport 22 ip saddr @knock_authenticated accept
    }
}
```

### 2.2 Intrusion Detection with Suricata

Suricata 8.0+ provides multi-threaded IDS/IPS with protocol identification and file extraction.

```yaml
# /etc/suricata/suricata.yaml
%YAML 1.1
---
vars:
  address-groups:
    HOME_NET: "[192.168.0.0/16,10.0.0.0/8,172.16.0.0/12]"
    EXTERNAL_NET: "!$HOME_NET"

# AF-PACKET with eBPF bypass for performance
af-packet:
  - interface: eth0
    threads: auto
    cluster-id: 99
    cluster-type: cluster_qm  # RSS-like load balancing
    defrag: yes
    use-mmap: yes
    ring-size: 200000
    # eBPF bypass for trusted flows
    bypass: yes
    xdp-mode: driver  # Best performance with NIC support

# Rule processing
detect:
  profile: high
  custom-values:
    toclient-groups: 200
    toserver-groups: 200
  
  # ML-based anomaly detection (2025 feature)
  ml-anomaly:
    enabled: yes
    models:
      - type: isolation-forest
        window: 3600
        features: [packet-size, flow-duration, byte-freq]

# Protocol detection
app-layer:
  protocols:
    http:
      enabled: yes
      memcap: 512mb
      # Detect web shells and malware
      libhtp:
        personality: IDS
        request-body-limit: 100kb
        response-body-limit: 100kb
    
    tls:
      enabled: yes
      # JA3/JA3S fingerprinting
      ja3-fingerprints: yes
      
    dns:
      enabled: yes
      # Detect DNS tunneling
      request-flood:
        enabled: yes
        threshold: 500

# Custom Lua scripting
lua:
  enabled: yes
  scripts:
    - /etc/suricata/scripts/detect_c2.lua
    - /etc/suricata/scripts/sandbox_check.lua
```

**Custom Detection Rule**

```lua
-- /etc/suricata/scripts/detect_c2.lua
function init()
    local needs = {}
    needs["payload"] = tostring(true)
    needs["packet"] = tostring(true)
    return needs
end

function match(args)
    local payload = args["payload"]
    if not payload then return 0 end
    
    -- Detect base64 encoded PowerShell
    local b64_pattern = "powershell%-encodedcommand"
    if string.find(payload:lower(), b64_pattern) then
        -- Check for common malicious patterns
        local decoded = base64_decode(payload)
        if decoded and (
            string.find(decoded, "invoke%-expression") or
            string.find(decoded, "downloadstring") or
            string.find(decoded, "hidden") 
        ) then
            return 1
        end
    end
    
    return 0
end
```

### 2.3 WireGuard VPN with Zero-Trust Extensions

```bash
# Generate keys with quantum-resistant algorithm preparation
wg genkey | tee privatekey | wg pubkey > publickey

# Server configuration with ECDSA + Kyber hybrid (future-proofing)
# /etc/wireguard/wg0.conf
[Interface]
Address = 10.200.0.1/24
ListenPort = 51820
PrivateKey = <server-private-key>

# Implement zero-trust continuous verification
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; /usr/local/bin/verify-clients.sh
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
Table = off  # We'll handle routing ourselves

# Client peer with time-based access
[Peer]
PublicKey = <client-public-key>
AllowedIPs = 10.200.0.2/32
PersistentKeepalive = 25

# MFA extension via webhook
# Requires external validation before allowing traffic
PostUp = /usr/local/bin/validate-mfa.sh %i %a
```

**Zero-Trust Validation Script**

```python
#!/usr/bin/env python3
# /usr/local/bin/verify-clients.py
import asyncio
import json
import aiohttp
from datetime import datetime, timedelta

async def verify_client(client_ip, public_key):
    """Continuously verify client authorization"""
    
    # Check device compliance
    async with aiohttp.ClientSession() as session:
        # Verify with identity provider
        async with session.post('https://idp.company.com/verify', json={
            'client_ip': client_ip,
            'public_key': public_key,
            'checks': ['device_health', 'user_active', 'location_allowed']
        }) as resp:
            result = await resp.json()
            
    if not result['authorized']:
        # Revoke access
        subprocess.run(['wg', 'set', 'wg0', 'peer', public_key, 'remove'])
        
    return result['authorized']

# Run verification every 5 minutes
async def continuous_verification():
    while True:
        peers = get_wireguard_peers()
        tasks = [verify_client(p['ip'], p['key']) for p in peers]
        await asyncio.gather(*tasks)
        await asyncio.sleep(300)
```

### 2.4 eBPF-Based Security Monitoring

eBPF provides kernel-level observability without kernel modules.

```c
// network_monitor.bpf.c
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

// Map for suspicious connection tracking
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 10000);
    __type(key, __u64);  // Source IP + Port
    __type(value, __u32); // Connection count
} suspicious_conns SEC(".maps");

// Detect port scanning
SEC("xdp")
int detect_port_scan(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;
    
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;
    
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;
    
    if (ip->protocol != IPPROTO_TCP)
        return XDP_PASS;
    
    struct tcphdr *tcp = (void *)ip + (ip->ihl * 4);
    if ((void *)(tcp + 1) > data_end)
        return XDP_PASS;
    
    // Check for SYN flag (potential scan)
    if (tcp->syn && !tcp->ack) {
        __u64 key = ((__u64)ip->saddr << 32) | bpf_ntohs(tcp->source);
        __u32 *count = bpf_map_lookup_elem(&suspicious_conns, &key);
        
        if (count) {
            __sync_fetch_and_add(count, 1);
            if (*count > 100) {
                // Drop packets from this source (rate limit)
                return XDP_DROP;
            }
        } else {
            __u32 init_count = 1;
            bpf_map_update_elem(&suspicious_conns, &key, &init_count, BPF_ANY);
        }
    }
    
    return XDP_PASS;
}

// Detect data exfiltration patterns
SEC("tc")
int detect_exfiltration(struct __sk_buff *skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;
    
    // Parse headers (similar to above)
    // ...
    
    // Check for unusual data patterns
    if (skb->len > 10000) {  // Large upload
        // Extract flow tuple
        struct flow_key {
            __u32 src_ip;
            __u32 dst_ip;
            __u16 src_port;
            __u16 dst_port;
        } key = {
            .src_ip = ip->saddr,
            .dst_ip = ip->daddr,
            .src_port = tcp->source,
            .dst_port = tcp->dest,
        };
        
        // Log to perf buffer for userspace analysis
        bpf_perf_event_output(skb, &events, BPF_F_CURRENT_CPU,
                             &key, sizeof(key));
    }
    
    return TC_ACT_OK;
}
```

**Userspace Controller**

```python
#!/usr/bin/env python3
# ebpf_monitor.py
from bcc import BPF
import ctypes as ct
import socket
import struct

# Load and attach eBPF program
bpf = BPF(src_file="network_monitor.bpf.c")

# Attach XDP program
device = "eth0"
bpf.attach_xdp(device, fn=bpf.load_func("detect_port_scan", BPF.XDP))

# Attach TC program  
tc = bpf.load_func("detect_exfiltration", BPF.SCHED_CLS)

# Define event structure
class FlowKey(ct.Structure):
    _fields_ = [
        ("src_ip", ct.c_uint32),
        ("dst_ip", ct.c_uint32),
        ("src_port", ct.c_uint16),
        ("dst_port", ct.c_uint16),
    ]

def ip_to_string(ip):
    return socket.inet_ntoa(struct.pack("I", ip))

# Process events
def process_event(cpu, data, size):
    event = ct.cast(data, ct.POINTER(FlowKey)).contents
    print(f"Potential exfiltration: {ip_to_string(event.src_ip)}:{event.src_port} -> "
          f"{ip_to_string(event.dst_ip)}:{event.dst_port}")
    
    # Add to threat intelligence
    # Send alert to SIEM
    # Block if confidence high

# Read events
bpf["events"].open_perf_buffer(process_event)

while True:
    try:
        bpf.perf_buffer_poll()
    except KeyboardInterrupt:
        break
```

---

## 3. Reverse Engineering and Binary Analysis

Modern reverse engineering combines static analysis, dynamic instrumentation, and symbolic execution.

### 3.1 Ghidra: NSA's Gift to Reversers

Ghidra 11.2+ includes significant improvements in decompilation and collaboration.

**Advanced Ghidra Scripting**

```python
# FindCryptoConstants.py - Ghidra script
# @category Crypto
# @toolbar crypto.png

from ghidra.program.model.listing import CodeUnit
from ghidra.program.model.scalar import Scalar
import struct

# Common crypto constants
CRYPTO_CONSTANTS = {
    0x67452301: "MD5_A",
    0xEFCDAB89: "MD5_B", 
    0x98BADCFE: "MD5_C",
    0x10325476: "MD5_D",
    0x6A09E667: "SHA256_H0",
    0xBB67AE85: "SHA256_H1",
    0x3C6EF372: "SHA256_H2",
    0xA54FF53A: "SHA256_H3",
    0x428A2F98: "SHA256_K[0]",
    0x71374491: "SHA256_K[1]",
    # AES S-box values
    0x63636363: "Possible AES",
    # ChaCha20 constants
    0x61707865: "ChaCha20 'expa'",
    0x3320646e: "ChaCha20 'nd 3'",
    0x79622d32: "ChaCha20 '2-by'",
    0x6b206574: "ChaCha20 'te k'",
}

def find_crypto_constants():
    listing = currentProgram.getListing()
    monitor.setMessage("Searching for cryptographic constants...")
    
    # Search data sections
    memory = currentProgram.getMemory()
    for block in memory.getBlocks():
        if not block.isInitialized():
            continue
            
        start = block.getStart()
        end = block.getEnd()
        
        # Read 4 bytes at a time
        addr = start
        while addr <= end and not monitor.isCancelled():
            try:
                value = memory.getInt(addr)
                if value in CRYPTO_CONSTANTS:
                    print(f"Found {CRYPTO_CONSTANTS[value]} at {addr}")
                    createBookmark(addr, "Crypto", CRYPTO_CONSTANTS[value])
                    
                    # Check for crypto function patterns
                    analyze_crypto_usage(addr)
                    
            except:
                pass
            
            addr = addr.add(4)
            
def analyze_crypto_usage(const_addr):
    """Find references to crypto constants"""
    refs = getReferencesTo(const_addr)
    
    for ref in refs:
        func = getFunctionContaining(ref.getFromAddress())
        if func:
            # Tag function as crypto-related
            func.setComment(f"Uses crypto constant at {const_addr}")
            createBookmark(func.getEntryPoint(), "Analysis", 
                         f"Possible crypto function: {func.getName()}")

# Run the analysis
find_crypto_constants()
```

**P-Code Analysis for Deobfuscation**

```java
// DeobfuscateStrings.java - Ghidra script
import ghidra.app.script.GhidraScript;
import ghidra.program.model.pcode.*;
import ghidra.program.model.listing.*;

public class DeobfuscateStrings extends GhidraScript {
    
    @Override
    public void run() throws Exception {
        // Get high-level P-code
        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);
        
        FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
        
        while (functions.hasNext()) {
            Function func = functions.next();
            
            // Decompile to P-code
            DecompileResults results = decomp.decompileFunction(func, 30, monitor);
            if (!results.decompileCompleted()) {
                continue;
            }
            
            HighFunction hfunc = results.getHighFunction();
            Iterator<PcodeOpAST> ops = hfunc.getPcodeOps();
            
            while (ops.hasNext()) {
                PcodeOpAST op = ops.next();
                
                // Look for XOR decryption patterns
                if (op.getOpcode() == PcodeOp.INT_XOR) {
                    analyzeXorOperation(op, func);
                }
            }
        }
    }
    
    private void analyzeXorOperation(PcodeOpAST xorOp, Function func) {
        // Check if XORing with constant
        Varnode input1 = xorOp.getInput(0);
        Varnode input2 = xorOp.getInput(1);
        
        if (input2.isConstant()) {
            long xorKey = input2.getOffset();
            
            // Trace back to find string data
            if (isStringDecryption(input1, xorKey)) {
                println("Found string decryption in " + func.getName() + 
                       " with key 0x" + Long.toHexString(xorKey));
                       
                // Attempt to decrypt and annotate
                decryptStrings(func, xorKey);
            }
        }
    }
}
```

### 3.2 Radare2/Rizin Advanced Usage

Radare2's scriptable architecture excels at automated analysis.

```bash
# Advanced r2pipe script for malware analysis
#!/usr/bin/env python3
import r2pipe
import json
import networkx as nx

class MalwareAnalyzer:
    def __init__(self, binary_path):
        # Open with ESIL emulation
        self.r2 = r2pipe.open(binary_path, ["-2"])  # stderr for warnings
        self.r2.cmd("aaa")  # Analyze all
        self.r2.cmd("e emu.str=true")  # String emulation
        
    def find_encryption_loops(self):
        """Detect loops with XOR/rotation operations"""
        # Get all basic blocks
        blocks = json.loads(self.r2.cmd("agj"))
        
        for func in blocks:
            cfg = self.build_cfg(func)
            
            # Find loops using graph analysis
            cycles = nx.simple_cycles(cfg)
            
            for cycle in cycles:
                if self.contains_crypto_ops(cycle):
                    print(f"Potential encryption loop in {func['name']}: {cycle}")
    
    def contains_crypto_ops(self, basic_blocks):
        """Check if blocks contain crypto operations"""
        crypto_ops = ['xor', 'ror', 'rol', 'aes', 'rc4']
        
        for bb in basic_blocks:
            disasm = self.r2.cmd(f"pdb @ {bb}")
            for op in crypto_ops:
                if op in disasm.lower():
                    return True
        return False
    
    def extract_iocs(self):
        """Extract Indicators of Compromise"""
        # Extract strings
        strings = json.loads(self.r2.cmd("izj"))
        
        iocs = {
            'ips': [],
            'domains': [],
            'urls': [],
            'hashes': [],
            'mutexes': []
        }
        
        import re
        
        # Patterns for IOC extraction  
        patterns = {
            'ips': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'domains': r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}',
            'urls': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'hashes': r'\b[a-fA-F0-9]{32,64}\b',
            'mutexes': r'Global\\[A-Za-z0-9_]+' 
        }
        
        for s in strings:
            string_val = s['string']
            for ioc_type, pattern in patterns.items():
                matches = re.findall(pattern, string_val)
                iocs[ioc_type].extend(matches)
        
        # Emulate and extract runtime strings
        self.emulate_and_extract(iocs)
        
        return iocs
    
    def emulate_and_extract(self, iocs):
        """Use ESIL emulation to decrypt strings"""
        # Initialize ESIL VM
        self.r2.cmd("aei")  # Initialize ESIL VM
        self.r2.cmd("aeim") # Initialize ESIL VM memory
        
        # Find string decryption functions
        funcs = json.loads(self.r2.cmd("aflj"))
        
        for func in funcs:
            # Heuristic: small functions with loops often decrypt
            if func['nlocals'] > 0 and func['size'] < 500:
                # Emulate function
                self.r2.cmd(f"aepc {func['offset']}")
                
                # Check for new strings in memory
                new_strings = self.r2.cmd("ps @ rsp")
                # Add to IOCs...
```

### 3.3 Binary Instrumentation with Frida

Dynamic instrumentation for runtime analysis and bypass.

```javascript
// anti-debug-bypass.js - Frida script
'use strict';

// Bypass common anti-debugging techniques
const bypassAntiDebug = () => {
    // 1. Bypass ptrace detection
    const ptrace = Module.findExportByName(null, 'ptrace');
    if (ptrace) {
        Interceptor.attach(ptrace, {
            onEnter: function(args) {
                const request = args[0].toInt32();
                // PTRACE_TRACEME = 0
                if (request === 0) {
                    console.log('[*] Bypassing ptrace anti-debug');
                    args[0] = ptr(-1); // Change to invalid request
                }
            },
            onLeave: function(retval) {
                retval.replace(0); // Always return success
            }
        });
    }
    
    // 2. Bypass /proc/self/status checks
    const fopen = Module.findExportByName(null, 'fopen');
    Interceptor.attach(fopen, {
        onEnter: function(args) {
            const path = args[0].readUtf8String();
            if (path.includes('/proc/self/status') || 
                path.includes('/proc/self/stat')) {
                console.log('[*] Redirecting proc check:', path);
                // Redirect to clean file
                args[0] = Memory.allocUtf8String('/dev/null');
            }
        }
    });
    
    // 3. Bypass debugger timing checks
    const gettimeofday = Module.findExportByName(null, 'gettimeofday');
    let lastTime = 0;
    Interceptor.attach(gettimeofday, {
        onLeave: function(retval) {
            const tv = this.context.rdi; // First argument
            if (tv.isNull()) return;
            
            // Make time appear to advance normally
            if (lastTime === 0) {
                lastTime = tv.readU64();
            } else {
                // Add small increment
                tv.writeU64(lastTime + 1000);
                lastTime += 1000;
            }
        }
    });
};

// Advanced API monitoring
const monitorCrypto = () => {
    // Monitor OpenSSL/BoringSSL
    const SSL_write = Module.findExportByName(null, 'SSL_write');
    if (SSL_write) {
        Interceptor.attach(SSL_write, {
            onEnter: function(args) {
                const buf = args[1];
                const len = args[2].toInt32();
                
                console.log('\n[SSL_write]');
                console.log('Data:', buf.readByteArray(len));
                
                // Log backtrace for context
                console.log('Backtrace:\n' + 
                    Thread.backtrace(this.context, Backtracer.ACCURATE)
                        .map(DebugSymbol.fromAddress).join('\n'));
            }
        });
    }
    
    // Monitor system crypto APIs
    if (Process.platform === 'linux') {
        // AF_ALG socket crypto
        const socket = Module.findExportByName(null, 'socket');
        Interceptor.attach(socket, {
            onEnter: function(args) {
                const domain = args[0].toInt32();
                const type = args[1].toInt32();
                const protocol = args[2].toInt32();
                
                // AF_ALG = 38
                if (domain === 38) {
                    console.log('[*] AF_ALG crypto socket created');
                    this.cryptoSocket = true;
                }
            },
            onLeave: function(retval) {
                if (this.cryptoSocket) {
                    const fd = retval.toInt32();
                    console.log('[*] Crypto socket fd:', fd);
                    // Track this fd for monitoring
                }
            }
        });
    }
};

// Hook native code obfuscation
const deobfuscateNative = () => {
    // Find JNI functions in Android
    if (Java.available) {
        Java.perform(() => {
            const System = Java.use('java.lang.System');
            const Runtime = Java.use('java.lang.Runtime');
            
            // Hook loadLibrary to catch native library loading
            System.loadLibrary.implementation = function(libname) {
                console.log('[*] Loading native library:', libname);
                const result = this.loadLibrary(libname);
                
                // Wait for library to load then hook exports
                setTimeout(() => {
                    hookNativeLibrary(libname);
                }, 100);
                
                return result;
            };
        });
    }
};

// Entry point
console.log('[*] Starting advanced Frida hooks...');
bypassAntiDebug();
monitorCrypto();
deobfuscateNative();
```

### 3.4 Symbolic Execution with angr

Solve complex constraints and find vulnerabilities automatically.

```python
#!/usr/bin/env python3
# angr_analysis.py - Advanced symbolic execution

import angr
import claripy
import logging

logging.getLogger('angr').setLevel(logging.INFO)

class VulnerabilityFinder:
    def __init__(self, binary_path):
        # Load binary with auto-detection
        self.proj = angr.Project(binary_path, auto_load_libs=False)
        
        # Configure symbolic execution
        self.cfg = self.proj.analyses.CFGFast()
        
    def find_buffer_overflows(self):
        """Find potential buffer overflow vulnerabilities"""
        
        # Look for dangerous functions
        dangerous_funcs = ['strcpy', 'strcat', 'gets', 'sprintf', 'scanf']
        
        for func_name in dangerous_funcs:
            func = self.proj.loader.find_symbol(func_name)
            if not func:
                continue
                
            # Find all calls to dangerous function
            for node in self.cfg.nodes():
                if func.rebased_addr in node.instruction_addrs:
                    print(f"Found {func_name} at {hex(node.addr)}")
                    self.analyze_overflow(node.addr, func_name)
    
    def analyze_overflow(self, call_addr, func_name):
        """Symbolically execute to find overflow conditions"""
        
        # Create initial state at function entry
        state = self.proj.factory.blank_state(addr=call_addr)
        
        # Create symbolic buffer
        sym_buffer = claripy.BVS('input_buffer', 8 * 1024)  # 1KB symbolic buffer
        buffer_addr = 0x10000
        state.memory.store(buffer_addr, sym_buffer)
        
        # Set up arguments based on function
        if func_name == 'strcpy':
            state.regs.rdi = buffer_addr  # dst
            state.regs.rsi = buffer_addr + 512  # src (symbolic)
        
        # Create simulation manager
        simgr = self.proj.factory.simulation_manager(state)
        
        # Explore paths
        simgr.explore(
            find=lambda s: b"OVERFLOW" in s.posix.dumps(1),
            avoid=lambda s: s.addr == 0
        )
        
        if simgr.found:
            print(f"[!] Overflow possible at {hex(call_addr)}")
            
            # Get concrete input that triggers overflow
            solution = simgr.found[0]
            concrete_input = solution.solver.eval(sym_buffer, cast_to=bytes)
            
            print(f"Trigger input: {concrete_input[:100]}...")
            return concrete_input
            
    def find_authentication_bypass(self):
        """Find ways to bypass authentication"""
        
        # Common authentication patterns
        auth_patterns = [
            b'Login successful',
            b'Access granted',
            b'Authentication passed'
        ]
        
        # Find main or entry point
        entry_state = self.proj.factory.entry_state()
        
        # Add constraints for username/password
        username = claripy.BVS('username', 8 * 32)
        password = claripy.BVS('password', 8 * 32)
        
        # Common locations for input
        entry_state.memory.store(0x20000, username)
        entry_state.memory.store(0x20020, password)
        
        simgr = self.proj.factory.simulation_manager(entry_state)
        
        # Find successful authentication
        def is_authenticated(state):
            for pattern in auth_patterns:
                if state.posix.dumps(1).find(pattern) != -1:
                    return True
            return False
        
        simgr.explore(find=is_authenticated, num_find=5)
        
        if simgr.found:
            print("[!] Authentication bypass found!")
            
            for solution in simgr.found:
                user = solution.solver.eval(username, cast_to=bytes)
                pwd = solution.solver.eval(password, cast_to=bytes)
                
                print(f"Username: {user.decode('utf-8', errors='ignore')}")
                print(f"Password: {pwd.decode('utf-8', errors='ignore')}")
    
    def solve_crackme(self, success_addr, failure_addr):
        """Generic crackme solver"""
        
        # Start from entry point
        initial_state = self.proj.factory.entry_state(
            add_options={
                angr.options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
                angr.options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS
            }
        )
        
        # Create symbolic input
        flag_chars = [claripy.BVS(f'flag_{i}', 8) for i in range(32)]
        flag = claripy.Concat(*flag_chars)
        
        # Constrain to printable ASCII
        for char in flag_chars:
            initial_state.solver.add(char >= 0x20)
            initial_state.solver.add(char <= 0x7e)
        
        # Store flag in memory (adjust address as needed)
        flag_addr = 0x400000
        initial_state.memory.store(flag_addr, flag)
        
        # Run symbolic execution
        simgr = self.proj.factory.simulation_manager(initial_state)
        simgr.explore(find=success_addr, avoid=failure_addr)
        
        if simgr.found:
            solution = simgr.found[0]
            flag_str = solution.solver.eval(flag, cast_to=bytes)
            print(f"[+] Flag found: {flag_str.decode('utf-8')}")
            return flag_str

# Usage example
if __name__ == "__main__":
    analyzer = VulnerabilityFinder("./target_binary")
    analyzer.find_buffer_overflows()
    analyzer.find_authentication_bypass()
```

---

## 4. Hardware Security and CPU Vulnerabilities

Understanding and mitigating CPU-level vulnerabilities is crucial for system security.

### 4.1 Spectre/Meltdown and Beyond

Modern CPUs require careful programming to avoid speculative execution vulnerabilities.

**Detecting Speculative Execution Vulnerabilities**

```c
// spectre_poc.c - Detect and demonstrate Spectre vulnerability
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>

// Flush+Reload threshold
#define CACHE_HIT_THRESHOLD 80

unsigned int array1_size = 16;
uint8_t array1[160] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
uint8_t array2[256 * 512];
uint8_t temp = 0;

// Victim function vulnerable to Spectre
void victim_function(size_t x) {
    if (x < array1_size) {
        temp &= array2[array1[x] * 512];
    }
}

// Measure access time
uint64_t rdtsc_begin() {
    uint64_t a, d;
    asm volatile ("mfence\n\t"
                  "rdtsc\n\t"
                  "lfence\n\t"
                  : "=a" (a), "=d" (d)
                  :
                  : "memory");
    return (d<<32) | a;
}

uint64_t rdtsc_end() {
    uint64_t a, d;
    asm volatile ("lfence\n\t"
                  "rdtsc\n\t"
                  "mfence\n\t"
                  : "=a" (a), "=d" (d)
                  :
                  : "memory");
    return (d<<32) | a;
}

// Flush cache line
void flush(void *p) {
    asm volatile ("clflush (%0)\n\t" : : "r"(p) : "memory");
}

// Spectre attack
void spectre_attack(size_t malicious_x) {
    int i, j, k, mix_i;
    uint64_t time1, time2;
    int junk = 0;
    size_t training_x, x;
    
    // Initialize array2
    for (i = 0; i < 256; i++) {
        array2[i * 512] = 1;
    }
    
    // Flush array2 from cache
    for (i = 0; i < 256; i++) {
        flush(&array2[i * 512]);
    }
    
    // Training pattern
    training_x = 0;
    
    // Mistrain branch predictor
    for (j = 29; j >= 0; j--) {
        flush(&array1_size);
        
        // Delay to let flush complete
        for (volatile int z = 0; z < 100; z++) {}
        
        // Set x to training or malicious value
        x = ((j % 6) - 1) & ~0xFFFF;
        x = (x | (x >> 16));
        x = training_x ^ (x & (malicious_x ^ training_x));
        
        // Call victim function
        victim_function(x);
    }
    
    // Time reads to infer the secret
    for (i = 0; i < 256; i++) {
        mix_i = ((i * 167) + 13) & 255;
        time1 = rdtsc_begin();
        junk = array2[mix_i * 512];
        time2 = rdtsc_end() - time1;
        
        if (time2 <= CACHE_HIT_THRESHOLD) {
            printf("array1[%ld] = %d\n", malicious_x, mix_i);
        }
    }
}

// Mitigation: Use LFENCE
void victim_function_safe(size_t x) {
    if (x < array1_size) {
        asm volatile("lfence");  // Serializing instruction
        temp &= array2[array1[x] * 512];
    }
}
```

**Compiler-Based Mitigations**

```makefile
# Makefile with security flags
CC = clang
CFLAGS = -Wall -Wextra -O2 \
         -mretpoline \                    # Retpoline mitigation
         -mindirect-branch=thunk \        # Indirect branch thunking
         -mfunction-return=thunk \        # Function return thunking
         -fcf-protection=full \           # Control-flow protection
         -mbranch-protection=standard \   # ARM pointer authentication
         -mspeculative-load-hardening \   # SLH for Spectre v1
         -fstack-clash-protection \       # Stack clash protection
         -D_FORTIFY_SOURCE=3              # Runtime buffer checks

# Link-time optimizations for security
LDFLAGS = -Wl,-z,relro \      # Read-only relocations
          -Wl,-z,now \        # Immediate binding
          -Wl,-z,noexecstack \ # NX stack
          -Wl,-z,separate-code # Separate code/data

secure_app: main.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $
```

### 4.2 Side-Channel Analysis and Protection

```python
#!/usr/bin/env python3
# side_channel_analyzer.py - Detect timing vulnerabilities

import time
import statistics
import numpy as np
from scipy import stats

class SideChannelAnalyzer:
    def __init__(self, target_function, samples=1000):
        self.target = target_function
        self.samples = samples
        
    def timing_analysis(self, inputs):
        """Analyze timing variations across different inputs"""
        timings = {inp: [] for inp in inputs}
        
        # Warm up
        for _ in range(100):
            self.target(inputs[0])
        
        # Collect timing samples
        for inp in inputs:
            for _ in range(self.samples):
                start = time.perf_counter_ns()
                self.target(inp)
                end = time.perf_counter_ns()
                timings[inp].append(end - start)
        
        # Statistical analysis
        results = {}
        for inp, times in timings.items():
            results[inp] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times),
                'min': min(times),
                'max': max(times)
            }
        
        # Detect timing leaks
        self.detect_timing_leaks(results)
        
        return results
    
    def detect_timing_leaks(self, results):
        """Use statistical tests to detect timing dependencies"""
        
        # Convert to arrays for analysis
        inputs = list(results.keys())
        means = [results[inp]['mean'] for inp in inputs]
        
        # Perform ANOVA test
        f_stat, p_value = stats.f_oneway(*[results[inp]['times'] 
                                           for inp in inputs])
        
        if p_value < 0.01:
            print(f"[!] Timing leak detected (p-value: {p_value:.6f})")
            
            # Find which inputs differ
            for i, inp1 in enumerate(inputs):
                for inp2 in inputs[i+1:]:
                    t_stat, p_val = stats.ttest_ind(
                        results[inp1]['times'],
                        results[inp2]['times']
                    )
                    if p_val < 0.01:
                        print(f"    {inp1} vs {inp2}: {p_val:.6f}")
    
    def cache_analysis(self, addresses):
        """Detect cache-based side channels"""
        # Implement FLUSH+RELOAD attack
        pass
    
    def power_analysis(self, traces):
        """Simple Power Analysis (SPA) - requires hardware"""
        # Correlation Power Analysis (CPA)
        pass

# Example: Timing attack on password comparison
def unsafe_compare(password, user_input):
    """Vulnerable string comparison"""
    if len(password) != len(user_input):
        return False
    
    for i in range(len(password)):
        if password[i] != user_input[i]:
            return False  # Early return leaks information
        # Simulate some work
        time.sleep(0.0001)
    
    return True

def constant_time_compare(a, b):
    """Constant-time comparison"""
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    
    return result == 0

# Test for timing vulnerabilities
if __name__ == "__main__":
    analyzer = SideChannelAnalyzer(
        lambda x: unsafe_compare("secret123", x)
    )
    
    # Test with different prefixes
    test_inputs = [
        "axxxxxxxx",  # Wrong first char
        "sxxxxxxxx",  # Correct first char
        "sexxxxxxx",  # Correct first two chars
        "secxxxxxx",  # Correct first three chars
    ]
    
    results = analyzer.timing_analysis(test_inputs)
    
    print("\nTiming Analysis Results:")
    for inp, stats in results.items():
        print(f"{inp}: {stats['mean']:.2f}ns (σ={stats['stdev']:.2f})")
```

### 4.3 Kernel Security Hardening

```c
// kernel_hardening.c - Linux kernel security module
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/security.h>
#include <linux/slab.h>
#include <linux/fs.h>

// Custom LSM for additional security checks
static int custom_file_open(struct file *file)
{
    struct dentry *dentry = file->f_path.dentry;
    struct inode *inode = d_backing_inode(dentry);
    
    // Check for suspicious file operations
    if (file->f_flags & O_RDWR) {
        // Log dual read/write operations on sensitive files
        if (strstr(dentry->d_name.name, ".ssh") ||
            strstr(dentry->d_name.name, ".gnupg")) {
            pr_warn("Suspicious R/W operation on: %s\n", 
                    dentry->d_name.name);
        }
    }
    
    return 0;
}

static int custom_ptrace_access_check(struct task_struct *child,
                                     unsigned int mode)
{
    // Prevent debugging of sensitive processes
    if (strstr(child->comm, "sshd") ||
        strstr(child->comm, "gpg")) {
        pr_err("Blocked ptrace on sensitive process: %s\n", 
               child->comm);
        return -EPERM;
    }
    
    return 0;
}

// Hook structure
static struct security_hook_list custom_hooks[] __lsm_ro_after_init = {
    LSM_HOOK_INIT(file_open, custom_file_open),
    LSM_HOOK_INIT(ptrace_access_check, custom_ptrace_access_check),
};

static int __init custom_lsm_init(void)
{
    pr_info("Custom LSM: Initializing\n");
    security_add_hooks(custom_hooks, ARRAY_SIZE(custom_hooks), "custom");
    return 0;
}

DEFINE_LSM(custom) = {
    .name = "custom",
    .init = custom_lsm_init,
};
```

**sysctl Security Hardening**

```bash
#!/bin/bash
# kernel_hardening.sh - Comprehensive kernel hardening

cat << 'EOF' > /etc/sysctl.d/99-security.conf
# Kernel hardening settings for 2025

# Memory protections
kernel.yama.ptrace_scope = 2            # Strict ptrace
kernel.kptr_restrict = 2                # Hide kernel pointers
kernel.dmesg_restrict = 1               # Restrict dmesg
kernel.kexec_load_disabled = 1          # Disable kexec
kernel.sysrq = 0                        # Disable SysRq
kernel.unprivileged_bpf_disabled = 1   # Disable unprivileged BPF
kernel.unprivileged_userns_clone = 0   # Disable user namespaces

# ASLR and NX
kernel.randomize_va_space = 2           # Full ASLR
kernel.exec-shield = 1                  # NX bit

# Speculative execution mitigations
kernel.speculation_mitigations = auto   # Auto-detect CPU vulnerabilities

# Network hardening
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.tcp_timestamps = 0

# File system hardening
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
fs.suid_dumpable = 0

# Core dumps (disable in production)
kernel.core_uses_pid = 1
kernel.core_pattern = /dev/null
EOF

# Apply settings
sysctl -p /etc/sysctl.d/99-security.conf

# CPU vulnerability mitigations
echo "Checking CPU vulnerabilities..."
for vuln in /sys/devices/system/cpu/vulnerabilities/*; do
    echo "$(basename $vuln): $(cat $vuln)"
done

# Disable SMT if vulnerable to L1TF/MDS
if grep -q "Vulnerable" /sys/devices/system/cpu/vulnerabilities/l1tf; then
    echo "Disabling SMT due to L1TF vulnerability"
    echo off > /sys/devices/system/cpu/smt/control
fi
```

---

## 5. Cryptography Implementation Security

### 5.1 Secure Random Number Generation

```c
// secure_random.c - Cryptographically secure RNG
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/random.h>
#include <errno.h>

// CPU hardware RNG support
#ifdef __x86_64__
#include <immintrin.h>

int has_rdrand() {
    uint32_t eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 30)) != 0;  // Check RDRAND bit
}

int hardware_random(void *buf, size_t len) {
    if (!has_rdrand()) return -1;
    
    uint64_t *p = (uint64_t *)buf;
    size_t count = len / sizeof(uint64_t);
    
    for (size_t i = 0; i < count; i++) {
        int retries = 10;
        while (retries-- > 0) {
            if (_rdrand64_step(&p[i])) break;
        }
        if (retries < 0) return -1;
    }
    
    // Handle remaining bytes
    if (len % sizeof(uint64_t)) {
        uint64_t tmp;
        if (!_rdrand64_step(&tmp)) return -1;
        memcpy((uint8_t *)buf + count * sizeof(uint64_t), 
               &tmp, len % sizeof(uint64_t));
    }
    
    return 0;
}
#endif

// Secure random with fallback chain
int secure_random(void *buf, size_t len) {
    // 1. Try getrandom() - blocks until entropy available
    ssize_t ret = getrandom(buf, len, 0);
    if (ret == len) return 0;
    
    // 2. Try /dev/urandom with O_CLOEXEC
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        ssize_t bytes_read = read(fd, buf, len);
        close(fd);
        if (bytes_read == len) return 0;
    }
    
    #ifdef __x86_64__
    // 3. Try hardware RNG
    if (hardware_random(buf, len) == 0) return 0;
    #endif
    
    // 4. Fatal error - no secure randomness available
    fprintf(stderr, "FATAL: No secure random source available\n");
    abort();
}

// Constant-time random integer in range [0, max)
uint64_t random_uniform(uint64_t max) {
    if (max <= 1) return 0;
    
    // Calculate mask for rejection sampling
    uint64_t mask = ~0ULL;
    uint64_t top = max - 1;
    while (top > 0) {
        top >>= 1;
        mask >>= 1;
    }
    mask = ~mask;
    
    // Rejection sampling for uniformity
    uint64_t val;
    do {
        secure_random(&val, sizeof(val));
        val &= mask;
    } while (val >= max);
    
    return val;
}
```

### 5.2 Cryptographic Library Usage

```python
#!/usr/bin/env python3
# crypto_operations.py - Secure cryptographic operations

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import secrets
import os

class SecureCrypto:
    """Production-ready cryptographic operations"""
    
    @staticmethod
    def generate_key(length=32):
        """Generate cryptographically secure random key"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def derive_key(password: bytes, salt: bytes = None, 
                   length: int = 32, n: int = 2**20):
        """Derive key from password using scrypt"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = Scrypt(
            salt=salt,
            length=length,
            n=n,        # CPU/memory cost (2^20 for high security)
            r=8,        # Block size
            p=1,        # Parallelization
            backend=default_backend()
        )
        
        return kdf.derive(password), salt
    
    @staticmethod
    def encrypt_aes_gcm(plaintext: bytes, key: bytes, 
                        associated_data: bytes = None):
        """AES-GCM authenticated encryption"""
        if len(key) not in [16, 24, 32]:
            raise ValueError("Invalid key size")
        
        # Generate random nonce (96 bits for GCM)
        nonce = secrets.token_bytes(12)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        # Add associated data if provided
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        # Encrypt
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Return nonce + ciphertext + tag
        return nonce + ciphertext + encryptor.tag
    
    @staticmethod
    def decrypt_aes_gcm(ciphertext: bytes, key: bytes,
                        associated_data: bytes = None):
        """Decrypt AES-GCM"""
        if len(ciphertext) < 28:  # 12 (nonce) + 16 (tag)
            raise ValueError("Invalid ciphertext")
        
        # Extract components
        nonce = ciphertext[:12]
        tag = ciphertext[-16:]
        actual_ciphertext = ciphertext[12:-16]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        
        # Add associated data if provided
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        # Decrypt and verify
        return decryptor.update(actual_ciphertext) + decryptor.finalize()
    
    @staticmethod
    def generate_rsa_keypair(key_size=4096):
        """Generate RSA keypair for signatures"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    @staticmethod
    def sign_data(data: bytes, private_key):
        """Create RSA-PSS signature"""
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    @staticmethod
    def verify_signature(data: bytes, signature: bytes, public_key):
        """Verify RSA-PSS signature"""
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False
    
    @staticmethod
    def constant_time_compare(a: bytes, b: bytes) -> bool:
        """Constant-time comparison"""
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= x ^ y
        
        return result == 0

# Example: Encrypted communication protocol
class SecureChannel:
    def __init__(self):
        # Generate ephemeral keys
        self.my_private_key, self.my_public_key = \
            SecureCrypto.generate_rsa_keypair()
        self.peer_public_key = None
        self.shared_key = None
    
    def handshake(self, peer_public_key_pem: bytes):
        """Establish secure channel"""
        # Load peer's public key
        self.peer_public_key = serialization.load_pem_public_key(
            peer_public_key_pem,
            backend=default_backend()
        )
        
        # Generate shared secret (simplified - use ECDH in production)
        self.shared_key = SecureCrypto.generate_key()
        
        # Return our public key
        return self.my_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def send_message(self, message: bytes) -> bytes:
        """Send authenticated encrypted message"""
        # Create message ID for replay protection
        msg_id = secrets.token_bytes(16)
        
        # Encrypt message
        ciphertext = SecureCrypto.encrypt_aes_gcm(
            plaintext=msg_id + message,
            key=self.shared_key,
            associated_data=b"secure_channel_v1"
        )
        
        # Sign the ciphertext
        signature = SecureCrypto.sign_data(ciphertext, self.my_private_key)
        
        # Return signed ciphertext
        return signature + ciphertext
```

---

## 6. Container and Cloud Security

### 6.1 Container Runtime Security with gVisor

```yaml
# runsc-config.yaml - gVisor configuration
# High-security container runtime

# Install gVisor
# wget https://storage.googleapis.com/gvisor/releases/release/latest/x86_64/runsc
# chmod +x runsc && sudo mv runsc /usr/local/bin

# Docker configuration for gVisor
runtime: runsc
runtime_root: /var/run/docker/runtime-runsc

# Security options
debug: false
debug-log: /dev/null
log-packets: false
platform: ptrace  # or kvm for better performance

# Syscall filtering
filters:
  - name: "strict"
    allow:
      # Minimal syscall set
      - sys_read
      - sys_write
      - sys_open
      - sys_close
      - sys_stat
      - sys_fstat
      - sys_lstat
      - sys_poll
      - sys_brk
      - sys_mmap
      - sys_mprotect
      - sys_munmap
      - sys_rt_sigaction
      - sys_rt_sigprocmask
      - sys_ioctl
      - sys_pread64
      - sys_pwrite64
      - sys_readv
      - sys_writev
      - sys_pipe
      - sys_select
      - sys_mremap
      - sys_madvise
      - sys_shmget
      - sys_shmat
      - sys_nanosleep
      - sys_getitimer
      - sys_setitimer
      - sys_getpid
      - sys_socket
      - sys_connect
      - sys_accept
      - sys_sendto
      - sys_recvfrom
      - sys_bind
      - sys_listen
      - sys_clone
      - sys_execve
      - sys_exit
      - sys_wait4
      - sys_fcntl
      - sys_flock
      - sys_fsync
      - sys_getcwd
      - sys_gettimeofday
      - sys_arch_prctl

# Network isolation
network: sandbox
```

**Dockerfile Security Best Practices**

```dockerfile
# secure.Dockerfile - Hardened container image

# Use distroless or minimal base
FROM gcr.io/distroless/static-debian12:nonroot AS runtime

# Multi-stage build for minimal attack surface
FROM golang:1.23-alpine AS builder

# Run as non-root during build
RUN adduser -D -g '' appuser

# Install dependencies with verification
RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Copy dependency files first (layer caching)
COPY go.mod go.sum ./
RUN go mod download

# Verify dependencies
RUN go mod verify

# Copy source
COPY . .

# Build with security flags
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s \
    -extldflags '-static' \
    -X main.version=$(git describe --tags) \
    -buildid=" \
    -trimpath \
    -tags netgo \
    -o app

# Final stage
FROM runtime

# Copy only the binary
COPY --from=builder /build/app /app

# Set up non-root user
USER nonroot:nonroot

# No shell, no package manager, minimal attack surface
ENTRYPOINT ["/app"]

# Security labels
LABEL security.scan="trivy scan --severity HIGH,CRITICAL"
LABEL security.updates="automatic"
```

### 6.2 Kubernetes Security

```yaml
# pod-security.yaml - Hardened Pod specification
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  annotations:
    # AppArmor profile
    container.apparmor.security.beta.kubernetes.io/app: localhost/k8s-apparmor-example
    # Seccomp profile
    seccomp.security.alpha.kubernetes.io/pod: localhost/profiles/audit.json
spec:
  # Pod Security Standards
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    fsGroup: 10001
    seccompProfile:
      type: RuntimeDefault
    seLinuxOptions:
      level: "s0:c123,c456"
  
  containers:
  - name: app
    image: myapp:latest
    
    # Container security context
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 10001
      capabilities:
        drop:
          - ALL
        add:
          - NET_BIND_SERVICE  # Only if needed
      seccompProfile:
        type: Localhost
        localhostProfile: profiles/fine-grained.json
    
    # Resource limits (prevent DoS)
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "256Mi"
        cpu: "500m"
    
    # Health checks
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
        scheme: HTTPS
      initialDelaySeconds: 10
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
        scheme: HTTPS
      initialDelaySeconds: 5
      periodSeconds: 5
    
    # Mount only necessary paths
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache
  
  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
  
  # Network policies
  dnsPolicy: ClusterFirst
  hostNetwork: false
  hostPID: false
  hostIPC: false

---
# Network Policy for zero-trust networking
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: secure-app-netpol
spec:
  podSelector:
    matchLabels:
      app: secure-app
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: production
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: production
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  # Allow DNS
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

**OPA (Open Policy Agent) for Policy Enforcement**

```rego
# kubernetes-policies.rego - Security policies
package kubernetes.admission

import data.kubernetes.namespaces

# Deny containers with latest tag
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container '%v' uses :latest tag", [container.name])
}

# Require non-root containers
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%v' must run as non-root", [container.name])
}

# Enforce resource limits
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf("Container '%v' must specify memory limits", [container.name])
}

# Require specific labels
required_labels := {"app", "version", "team", "security-scan"}

deny[msg] {
    input.request.kind.kind == "Pod"
    missing := required_labels - {label | input.request.object.metadata.labels[label]}
    count(missing) > 0
    msg := sprintf("Pod missing required labels: %v", [missing])
}

# Restrict privileged containers
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    container.securityContext.privileged
    msg := sprintf("Privileged container '%v' not allowed", [container.name])
}

# Enforce image pull policy
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    container.imagePullPolicy != "Always"
    msg := sprintf("Container '%v' must use imagePullPolicy: Always", [container.name])
}
```

### 6.3 eBPF-Based Container Security

```c
// container_monitor.bpf.c - Container runtime security monitoring
#include <linux/bpf.h>
#include <linux/sched.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Map to track container syscalls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, u32);    // PID
    __type(value, u64);  // Syscall bitmap
} container_syscalls SEC(".maps");

// Map for suspicious behavior alerts
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u32));
} alerts SEC(".maps");

struct alert_event {
    u32 pid;
    u32 uid;
    char comm[16];
    int syscall;
    char container_id[64];
};

// Helper to check if process is in container
static __always_inline int is_container_process() {
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    
    // Check for container indicators
    // This is simplified - in production, check cgroup path
    return 1;  // Placeholder
}

// Monitor execve for container escapes
SEC("tracepoint/syscalls/sys_enter_execve")
int trace_execve(struct trace_event_raw_sys_enter *ctx) {
    if (!is_container_process()) return 0;
    
    struct alert_event event = {};
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.uid = bpf_get_current_uid_gid() >> 32;
    event.syscall = __NR_execve;
    
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    // Check for suspicious executables
    char *filename = (char *)ctx->args[0];
    char buf[256];
    bpf_probe_read_user_str(buf, sizeof(buf), filename);
    
    // Alert on suspicious patterns
    if (strstr(buf, "nsenter") || 
        strstr(buf, "docker") ||
        strstr(buf, "kubectl")) {
        bpf_perf_event_output(ctx, &alerts, BPF_F_CURRENT_CPU,
                            &event, sizeof(event));
    }
    
    return 0;
}

// Monitor privilege escalation attempts
SEC("kprobe/commit_creds")
int BPF_KPROBE(commit_creds_hook, struct cred *new) {
    if (!is_container_process()) return 0;
    
    struct cred *old = (struct cred *)bpf_get_current_task()->cred;
    
    // Check for privilege escalation
    if (BPF_CORE_READ(new, uid.val) == 0 && 
        BPF_CORE_READ(old, uid.val) != 0) {
        
        struct alert_event event = {};
        event.pid = bpf_get_current_pid_tgid() >> 32;
        event.syscall = -1;  // Special marker for priv esc
        
        bpf_perf_event_output(ctx, &alerts, BPF_F_CURRENT_CPU,
                            &event, sizeof(event));
    }
    
    return 0;
}

// Monitor container breakout via /proc
SEC("kprobe/proc_pid_readlink")
int BPF_KPROBE(proc_readlink_hook) {
    if (!is_container_process()) return 0;
    
    // Alert on suspicious /proc access
    struct alert_event event = {};
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.syscall = -2;  // Marker for /proc access
    
    bpf_perf_event_output(ctx, &alerts, BPF_F_CURRENT_CPU,
                        &event, sizeof(event));
    
    return 0;
}
```

---

## 7. Supply Chain Security

### 7.1 SLSA Framework Implementation

```yaml
# .github/workflows/slsa-build.yml - SLSA Level 3 compliant build
name: SLSA Compliant Build

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: read
  id-token: write
  attestations: write

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digests: ${{ steps.hash.outputs.digests }}
      
    steps:
    - uses: actions/checkout@v4
      with:
        persist-credentials: false
        
    - name: Setup Build Environment
      run: |
        # Pin all tool versions
        echo "golang 1.23.1" > .tool-versions
        echo "cosign 2.5.0" >> .tool-versions
        
    - name: Build Artifacts
      id: build
      run: |
        # Hermetic build
        go build -trimpath -ldflags="-s -w -buildid=" -o app
        
    - name: Generate Hashes
      id: hash
      run: |
        set -euo pipefail
        sha256sum app > checksums.txt
        echo "digests=$(cat checksums.txt | base64 -w0)" >> $GITHUB_OUTPUT
        
    - name: Upload Artifacts
      uses: actions/upload-artifact@v4
      with:
        name: build-artifacts
        path: |
          app
          checksums.txt
        retention-days: 5

  provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.0.0
    with:
      base64-subjects: "${{ needs.build.outputs.digests }}"
      upload-assets: true
      
  verification:
    needs: [build, provenance]
    runs-on: ubuntu-latest
    steps:
    - name: Download artifacts
      uses: actions/download-artifact@v4
      
    - name: Verify SLSA Provenance
      run: |
        # Verify the provenance
        slsa-verifier verify-artifact \
          --provenance-path provenance.intoto.jsonl \
          --source-uri github.com/${{ github.repository }} \
          app
```

### 7.2 In-Toto Supply Chain Attestations

```python
#!/usr/bin/env python3
# supply_chain_attestation.py - Generate in-toto attestations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

class SupplyChainAttestor:
    """Generate and verify in-toto attestations"""
    
    def __init__(self, key_path: str = None):
        if key_path and Path(key_path).exists():
            with open(key_path, 'rb') as f:
                self.signing_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        else:
            # Generate new key pair
            self.signing_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
    
    def generate_link_metadata(self, step_name: str, 
                              materials: list, 
                              products: list,
                              command: list):
        """Generate in-toto link metadata"""
        
        link = {
            "_type": "link",
            "name": step_name,
            "materials": self._hash_artifacts(materials),
            "products": self._hash_artifacts(products),
            "command": command,
            "environment": self._capture_environment(),
            "byproducts": {
                "stdout": "",
                "stderr": "",
                "return-value": 0
            }
        }
        
        # Execute command and capture output
        if command:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            link["byproducts"]["stdout"] = result.stdout
            link["byproducts"]["stderr"] = result.stderr
            link["byproducts"]["return-value"] = result.returncode
        
        # Sign the link
        signed_link = self._sign_metadata(link)
        
        # Save link file
        filename = f"{step_name}.link"
        with open(filename, 'w') as f:
            json.dump(signed_link, f, indent=2)
        
        return signed_link
    
    def _hash_artifacts(self, artifacts: list) -> dict:
        """Generate hashes for artifacts"""
        hashed = {}
        
        for artifact in artifacts:
            if Path(artifact).exists():
                sha256_hash = hashlib.sha256()
                with open(artifact, 'rb') as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                
                hashed[artifact] = {
                    "sha256": sha256_hash.hexdigest(),
                    "size": Path(artifact).stat().st_size
                }
        
        return hashed
    
    def _capture_environment(self) -> dict:
        """Capture build environment for reproducibility"""
        import platform
        import os
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "env_vars": {
                k: v for k, v in os.environ.items()
                if k.startswith(('CI', 'GITHUB', 'BUILD'))
            }
        }
    
    def _sign_metadata(self, metadata: dict) -> dict:
        """Sign metadata with private key"""
        
        # Canonical JSON serialization
        canonical = json.dumps(metadata, sort_keys=True, 
                              separators=(',', ':'))
        
        # Sign
        signature = self.signing_key.sign(
            canonical.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Create signed wrapper
        public_key = self.signing_key.public_key()
        key_id = hashlib.sha256(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        ).hexdigest()[:16]
        
        return {
            "signatures": [{
                "keyid": key_id,
                "sig": signature.hex()
            }],
            "signed": metadata
        }
    
    def create_layout(self, steps: list, inspections: list):
        """Create in-toto layout policy"""
        
        layout = {
            "_type": "layout",
            "expires": "2030-01-01T00:00:00Z",
            "readme": "Supply chain policy for secure builds",
            "steps": steps,
            "inspections": inspections,
            "keys": {
                self._get_key_id(): self._get_public_key()
            }
        }
        
        return self._sign_metadata(layout)

# Example usage for build pipeline
if __name__ == "__main__":
    attestor = SupplyChainAttestor()
    
    # Step 1: Source code
    attestor.generate_link_metadata(
        step_name="clone",
        materials=[],
        products=["src/main.go", "go.mod", "go.sum"],
        command=["git", "clone", "https://github.com/example/app"]
    )
    
    # Step 2: Build
    attestor.generate_link_metadata(
        step_name="build", 
        materials=["src/main.go", "go.mod", "go.sum"],
        products=["app", "app.sha256"],
        command=["go", "build", "-o", "app", "./src"]
    )
    
    # Step 3: Test
    attestor.generate_link_metadata(
        step_name="test",
        materials=["app"],
        products=["test-results.xml"],
        command=["go", "test", "./...", "-v"]
    )
```

### 7.3 SigStore Integration

```bash
#!/bin/bash
# sigstore-sign.sh - Keyless signing with SigStore

set -euo pipefail

# Function to sign artifact with Cosign
sign_artifact() {
    local artifact=$1
    
    echo "Signing $artifact with Cosign..."
    
    # Sign with keyless mode (OIDC)
    cosign sign-blob \
        --yes \
        --oidc-issuer https://oauth2.sigstore.dev/auth \
        --fulcio-url https://fulcio.sigstore.dev \
        --rekor-url https://rekor.sigstore.dev \
        "$artifact" > "${artifact}.sig"
    
    # Upload signature to Rekor transparency log
    rekor-cli upload \
        --artifact "$artifact" \
        --signature "${artifact}.sig" \
        --public-key <(cosign public-key) \
        --rekor_server https://rekor.sigstore.dev
    
    # Generate SBOM
    syft packages "$artifact" -o spdx-json > "${artifact}.sbom.json"
    
    # Attest SBOM
    cosign attest \
        --yes \
        --predicate "${artifact}.sbom.json" \
        --type spdxjson \
        "$artifact"
}

# Function to verify artifact
verify_artifact() {
    local artifact=$1
    
    echo "Verifying $artifact..."
    
    # Verify signature
    cosign verify-blob \
        --certificate-identity-regexp ".*" \
        --certificate-oidc-issuer https://oauth2.sigstore.dev/auth \
        --signature "${artifact}.sig" \
        "$artifact"
    
    # Verify attestation
    cosign verify-attestation \
        --type spdxjson \
        --certificate-identity-regexp ".*" \
        --certificate-oidc-issuer https://oauth2.sigstore.dev/auth \
        "$artifact" | jq .
    
    # Check Rekor transparency log
    rekor-cli search --artifact "$artifact"
}

# Policy file for admission control
cat > cosign-policy.yaml << 'EOF'
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata:
  name: image-policy
spec:
  images:
  - glob: "**"
    authorities:
    - keyless:
        url: https://fulcio.sigstore.dev
        identities:
        - issuer: https://oauth2.sigstore.dev/auth
          subject: "user@example.com"
    attestations:
    - name: vuln-scan
      predicateType: cosign.sigstore.dev/attestation/vuln/v1
      policy:
        type: cue
        data: |
          import "time"
          before: time.Parse(time.RFC3339, time.Now())
          predicateType: "cosign.sigstore.dev/attestation/vuln/v1"
          predicate: {
            scanner: "trivy"
            result: {
              criticalCount: 0
              highCount: 0
            }
          }
EOF
```

---

## 8. Security Monitoring and Incident Response

### 8.1 SIEM with Wazuh

```xml
<!-- /var/ossec/etc/ossec.conf - Wazuh configuration -->
<ossec_config>
  <!-- Global settings -->
  <global>
    <jsonout_output>yes</jsonout_output>
    <alerts_log>yes</alerts_log>
    <logall>no</logall>
    <logall_json>no</logall_json>
    <email_notification>yes</email_notification>
    <smtp_server>smtp.example.com</smtp_server>
    <email_from>wazuh@example.com</email_from>
    <email_to>security@example.com</email_to>
    <email_maxperhour>12</email_maxperhour>
  </global>

  <!-- Security monitoring rules -->
  <ruleset>
    <!-- Detect privilege escalation -->
    <rule id="100001" level="15">
      <if_sid>5501</if_sid>
      <match>sudo.*COMMAND=/bin/bash|sudo.*COMMAND=/bin/sh</match>
      <description>Privilege escalation: sudo to shell detected</description>
      <group>privilege_escalation,</group>
    </rule>

    <!-- Detect suspicious network connections -->
    <rule id="100002" level="12">
      <if_group>syslog</if_group>
      <match>Connection to .* port (4444|1337|31337|6666)</match>
      <description>Connection to common backdoor port</description>
      <group>backdoor,network,</group>
    </rule>

    <!-- Container escape detection -->
    <rule id="100003" level="14">
      <decoded_as>json</decoded_as>
      <field name="container.name">\.+</field>
      <match>nsenter|docker.*exec.*privileged|kubectl.*exec</match>
      <description>Potential container escape attempt</description>
      <group>container,escape,</group>
    </rule>

    <!-- Detect cryptominer -->
    <rule id="100004" level="13">
      <if_sid>530</if_sid>
      <match>xmrig|minergate|cryptonight|stratum+tcp</match>
      <description>Cryptocurrency miner detected</description>
      <group>malware,miner,</group>
    </rule>
  </ruleset>

  <!-- File integrity monitoring -->
  <syscheck>
    <disabled>no</disabled>
    <frequency>43200</frequency>
    <scan_on_start>yes</scan_on_start>

    <!-- Critical system files -->
    <directories check_all="yes" report_changes="yes" realtime="yes">
      /etc,/usr/bin,/usr/sbin,/bin,/sbin
    </directories>
    
    <!-- Kubernetes configs -->
    <directories check_all="yes" report_changes="yes">
      /etc/kubernetes,/etc/cni,/opt/cni/bin
    </directories>

    <!-- Exclude noisy paths -->
    <ignore>/etc/mtab</ignore>
    <ignore>/etc/hosts.deny</ignore>
    <ignore>/etc/random-seed</ignore>
    <ignore>/etc/adjtime</ignore>

    <!-- Monitor Windows registry (if applicable) -->
    <windows_registry>HKEY_LOCAL_MACHINE\Software\Classes</windows_registry>
    <windows_registry>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion</windows_registry>
  </syscheck>

  <!-- Log analysis -->
  <localfile>
    <log_format>json</log_format>
    <location>/var/log/containers/*.log</location>
  </localfile>

  <localfile>
    <log_format>syslog</log_format>
    <location>/var/log/auth.log</location>
  </localfile>

  <localfile>
    <log_format>command</log_format>
    <command>df -P</command>
    <frequency>360</frequency>
  </localfile>

  <!-- Active response -->
  <active-response>
    <disabled>no</disabled>
    <command>firewall-drop</command>
    <location>local</location>
    <rules_id>100002,100003</rules_id>
    <timeout>600</timeout>
  </active-response>

  <!-- Integration with threat intelligence -->
  <integration>
    <name>virustotal</name>
    <api_key>YOUR_API_KEY</api_key>
    <rule_id>550,551,552</rule_id>
    <alert_format>json</alert_format>
  </integration>
</ossec_config>
```

### 8.2 Advanced Incident Response Automation

```python
#!/usr/bin/env python3
# incident_response.py - Automated incident response system

import asyncio
import json
import aiohttp
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import redis.asyncio as redis
from dataclasses import dataclass
from enum import Enum

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SecurityIncident:
    id: str
    timestamp: datetime
    severity: IncidentSeverity
    type: str
    source: str
    details: Dict
    affected_systems: List[str]

class IncidentResponseSystem:
    def __init__(self):
        self.redis = redis.Redis(decode_responses=True)
        self.playbooks = self._load_playbooks()
        self.active_incidents = {}
        
    def _load_playbooks(self) -> Dict:
        """Load response playbooks"""
        return {
            "malware_detected": self.respond_to_malware,
            "intrusion_attempt": self.respond_to_intrusion,
            "data_exfiltration": self.respond_to_exfiltration,
            "privilege_escalation": self.respond_to_privesc,
            "dos_attack": self.respond_to_dos,
        }
    
    async def process_alert(self, alert: Dict):
        """Process incoming security alert"""
        
        # Create incident
        incident = SecurityIncident(
            id=self._generate_incident_id(),
            timestamp=datetime.now(),
            severity=self._assess_severity(alert),
            type=alert.get('type', 'unknown'),
            source=alert.get('source', 'unknown'),
            details=alert,
            affected_systems=self._identify_affected_systems(alert)
        )
        
        # Store incident
        await self.redis.hset(
            f"incident:{incident.id}",
            mapping=incident.__dict__
        )
        
        # Execute response playbook
        if incident.type in self.playbooks:
            asyncio.create_task(
                self.playbooks[incident.type](incident)
            )
        
        # Notify team
        await self._notify_security_team(incident)
        
        return incident
    
    async def respond_to_malware(self, incident: SecurityIncident):
        """Automated malware response"""
        
        affected_host = incident.affected_systems[0]
        
        # 1. Isolate affected system
        await self._isolate_host(affected_host)
        
        # 2. Capture forensic data
        evidence_path = await self._capture_forensics(affected_host)
        
        # 3. Kill malicious processes
        if 'process_id' in incident.details:
            await self._kill_process(affected_host, 
                                   incident.details['process_id'])
        
        # 4. Remove malware artifacts
        if 'file_path' in incident.details:
            await self._quarantine_file(affected_host,
                                       incident.details['file_path'])
        
        # 5. Scan for additional infections
        await self._full_system_scan(affected_host)
        
        # 6. Update incident
        await self._update_incident_status(
            incident.id, 
            "contained",
            {
                "isolation_time": datetime.now().isoformat(),
                "evidence_path": evidence_path,
                "actions_taken": [
                    "host_isolated",
                    "forensics_captured", 
                    "malware_removed",
                    "system_scanned"
                ]
            }
        )
    
    async def respond_to_intrusion(self, incident: SecurityIncident):
        """Respond to intrusion attempts"""
        
        source_ip = incident.details.get('source_ip')
        target_port = incident.details.get('target_port')
        
        # 1. Block source IP at perimeter
        await self._block_ip_firewall(source_ip)
        
        # 2. Check for lateral movement
        compromised = await self._check_lateral_movement(source_ip)
        
        # 3. Reset credentials if needed
        if compromised:
            await self._force_password_reset(compromised)
        
        # 4. Enhanced monitoring
        await self._enable_enhanced_monitoring(incident.affected_systems)
        
        # 5. Threat intelligence lookup
        threat_info = await self._threat_intel_lookup(source_ip)
        
        await self._update_incident_status(
            incident.id,
            "mitigated",
            {
                "blocked_ip": source_ip,
                "threat_intelligence": threat_info,
                "compromised_accounts": compromised
            }
        )
    
    async def _isolate_host(self, hostname: str):
        """Network isolation using iptables"""
        
        # Allow only security team access
        rules = [
            f"iptables -I INPUT -s 10.0.100.0/24 -j ACCEPT",
            f"iptables -I OUTPUT -d 10.0.100.0/24 -j ACCEPT",
            f"iptables -P INPUT DROP",
            f"iptables -P OUTPUT DROP",
            f"iptables -P FORWARD DROP"
        ]
        
        for rule in rules:
            await self._remote_execute(hostname, rule)
    
    async def _capture_forensics(self, hostname: str) -> str:
        """Capture forensic data from compromised host"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evidence_dir = f"/forensics/{hostname}_{timestamp}"
        
        # Commands to capture forensic data
        commands = [
            f"mkdir -p {evidence_dir}",
            f"cp -r /var/log {evidence_dir}/",
            f"ps auxww > {evidence_dir}/processes.txt",
            f"netstat -plant > {evidence_dir}/network.txt",
            f"lsof -n > {evidence_dir}/open_files.txt",
            f"find / -mtime -1 -type f > {evidence_dir}/recent_files.txt",
            f"tar -czf {evidence_dir}.tar.gz {evidence_dir}"
        ]
        
        for cmd in commands:
            await self._remote_execute(hostname, cmd)
        
        return f"{evidence_dir}.tar.gz"
    
    async def _threat_intel_lookup(self, ioc: str) -> Dict:
        """Query threat intelligence sources"""
        
        results = {}
        
        # VirusTotal lookup
        async with aiohttp.ClientSession() as session:
            # Check IP reputation
            async with session.get(
                f"https://www.virustotal.com/api/v3/ip-addresses/{ioc}",
                headers={"x-apikey": "YOUR_VT_API_KEY"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results['virustotal'] = {
                        'malicious': data['data']['attributes']['last_analysis_stats']['malicious'],
                        'reputation': data['data']['attributes']['reputation']
                    }
        
        # Check against threat feeds
        if await self.redis.sismember("threat_intel:ips", ioc):
            results['known_threat'] = True
            
        return results
    
    async def _remote_execute(self, host: str, command: str):
        """Execute command on remote host"""
        
        # Use SSH with key-based auth
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-i", "/etc/security/incident_response_key",
            f"incident@{host}",
            command
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise Exception(f"Remote execution failed: {stderr.decode()}")
        
        return stdout.decode()

# Forensic artifact collection
class ForensicCollector:
    """Collect and preserve forensic evidence"""
    
    @staticmethod
    async def memory_dump(hostname: str) -> str:
        """Capture memory dump using LiME"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_file = f"/forensics/memory_{hostname}_{timestamp}.lime"
        
        # Load LiME kernel module
        commands = [
            "insmod /opt/lime/lime.ko path={dump_file} format=lime",
            f"sha256sum {dump_file} > {dump_file}.sha256"
        ]
        
        # Execute remotely
        # ... implementation ...
        
        return dump_file
    
    @staticmethod
    async def disk_image(hostname: str, device: str = "/dev/sda") -> str:
        """Create forensic disk image"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_file = f"/forensics/disk_{hostname}_{timestamp}.dd"
        
        # Use dd with proper block size
        dd_cmd = (
            f"dd if={device} of={image_file} "
            f"bs=4M conv=sync,noerror status=progress"
        )
        
        # Calculate hash while imaging
        hash_cmd = f"tee {image_file} | sha256sum > {image_file}.sha256"
        
        full_cmd = f"{dd_cmd} | {hash_cmd}"
        
        # Execute...
        
        return image_file

# Integration with SOAR platform
class SOARIntegration:
    """Security Orchestration, Automation and Response"""
    
    def __init__(self, api_endpoint: str):
        self.endpoint = api_endpoint
        self.session = aiohttp.ClientSession()
    
    async def create_case(self, incident: SecurityIncident) -> str:
        """Create case in SOAR platform"""
        
        case_data = {
            "title": f"Security Incident: {incident.type}",
            "severity": incident.severity.value,
            "description": json.dumps(incident.details),
            "observables": self._extract_observables(incident),
            "tasks": self._generate_tasks(incident)
        }
        
        async with self.session.post(
            f"{self.endpoint}/api/cases",
            json=case_data
        ) as resp:
            result = await resp.json()
            return result['id']
    
    def _extract_observables(self, incident: SecurityIncident) -> List[Dict]:
        """Extract IOCs from incident"""
        
        observables = []
        details = incident.details
        
        # Extract IPs
        for key in ['source_ip', 'dest_ip', 'attacker_ip']:
            if key in details:
                observables.append({
                    "type": "ip",
                    "value": details[key],
                    "tags": ["incident", incident.type]
                })
        
        # Extract file hashes
        for key in ['file_hash', 'md5', 'sha256']:
            if key in details:
                observables.append({
                    "type": "hash",
                    "value": details[key],
                    "tags": ["malware", "incident"]
                })
        
        return observables
```

---

## 9. Secure Development Workflow

### 9.1 Security-First CI/CD Pipeline

```yaml
# .gitlab-ci.yml - Comprehensive security pipeline
stages:
  - validate
  - build
  - test
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  SECURE_ANALYZERS_PREFIX: "registry.gitlab.com/security-products"
  DS_DEFAULT_ANALYZERS: "bandit,brakeman,gosec,nodejs-scan"

# Secret scanning
secret_detection:
  stage: validate
  image: trufflesecurity/trufflehog:latest
  script:
    - trufflehog git file://. --json --regex --entropy=True
    - gitleaks detect --source . --verbose --redact
  allow_failure: false

# Dependency scanning
dependency_scanning:
  stage: security
  image: $SECURE_ANALYZERS_PREFIX/dependency-scanning:latest
  script:
    - /analyzer run
  artifacts:
    reports:
      dependency_scanning: gl-dependency-scanning-report.json

# SAST with multiple tools
sast:semgrep:
  stage: security
  image: returntocorp/semgrep
  script:
    - semgrep --config=auto --json --output=semgrep-report.json .
    - semgrep --config=p/security-audit --config=p/owasp-top-ten .
  artifacts:
    reports:
      sast: semgrep-report.json

sast:codeql:
  stage: security
  image: ghcr.io/github/codeql-action:latest
  script:
    - codeql database create codeql-db --language=javascript
    - codeql database analyze codeql-db --format=sarif-latest --output=codeql-results.sarif

# License compliance
license_scanning:
  stage: security
  image: $SECURE_ANALYZERS_PREFIX/license-finder:latest
  script:
    - /analyzer run
  artifacts:
    reports:
      license_scanning: gl-license-scanning-report.json

# Container scanning
container_scanning:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL --format json --output trivy-report.json $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    # Fail on critical vulnerabilities
    - trivy image --exit-code 1 --severity CRITICAL $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  artifacts:
    reports:
      container_scanning: trivy-report.json

# DAST
dast:
  stage: security
  image: owasp/zap2docker-stable
  script:
    - |
      docker run -t owasp/zap2docker-stable zap-baseline.py \
        -t $DAST_WEBSITE \
        -J zap-report.json \
        -r zap-report.html \
        -c zap-rules.conf
  artifacts:
    reports:
      dast: zap-report.json

# Fuzzing
fuzz_testing:
  stage: test
  image: aflplusplus/aflplusplus
  script:
    - afl-fuzz -i fuzzing/input -o fuzzing/output -- ./build/target @@
  timeout: 1 hour
  allow_failure: true

# Infrastructure as Code scanning
iac_scanning:
  stage: validate
  image: bridgecrew/checkov
  script:
    - checkov -d . --framework kubernetes terraform --output json > checkov-report.json
    - tfsec . --format json > tfsec-report.json
  artifacts:
    reports:
      terraform: checkov-report.json

# Signed commits verification
verify_commits:
  stage: validate
  script:
    - |
      for commit in $(git rev-list origin/main..HEAD); do
        if ! git verify-commit $commit &>/dev/null; then
          echo "Unsigned commit found: $commit"
          exit 1
        fi
      done

# Deploy with security checks
deploy_production:
  stage: deploy
  script:
    # Verify all security scans passed
    - |
      if [ -f trivy-report.json ] && grep -q "CRITICAL" trivy-report.json; then
        echo "Critical vulnerabilities found, deployment blocked"
        exit 1
      fi
    
    # Sign deployment artifacts
    - cosign sign --key $COSIGN_KEY $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    
    # Deploy with verification
    - kubectl set image deployment/app app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/app
  only:
    - main
  when: manual
```

### 9.2 Secure Code Review Automation

```python
#!/usr/bin/env python3
# secure_code_review.py - Automated security code review

import ast
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import git
import requests

class SecurityCodeReviewer:
    """Automated security-focused code review"""
    
    def __init__(self, repo_path: str):
        self.repo = git.Repo(repo_path)
        self.issues = []
        self.severity_levels = {
            'CRITICAL': 4,
            'HIGH': 3,
            'MEDIUM': 2, 
            'LOW': 1
        }
    
    def review_pull_request(self, pr_branch: str) -> Dict:
        """Comprehensive security review of PR"""
        
        # Get changed files
        changed_files = self._get_changed_files(pr_branch)
        
        # Run security checks
        for file_path in changed_files:
            if file_path.suffix == '.py':
                self._review_python_file(file_path)
            elif file_path.suffix in ['.js', '.ts']:
                self._review_javascript_file(file_path)
            elif file_path.suffix == '.go':
                self._review_go_file(file_path)
            elif file_path.name == 'Dockerfile':
                self._review_dockerfile(file_path)
            elif file_path.suffix in ['.yml', '.yaml']:
                self._review_yaml_file(file_path)
        
        # Check for secrets
        self._scan_for_secrets(changed_files)
        
        # Check dependencies
        self._check_dependencies()
        
        # Generate report
        return self._generate_report()
    
    def _review_python_file(self, file_path: Path):
        """Python-specific security checks"""
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return
        
        # Check for dangerous functions
        for node in ast.walk(tree):
            # eval() and exec()
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        self.issues.append({
                            'file': str(file_path),
                            'line': node.lineno,
                            'severity': 'HIGH',
                            'issue': f'Dangerous function: {node.func.id}',
                            'recommendation': 'Avoid using eval/exec/\_\_import\_\_'
                        })
            
            # SQL injection
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                if isinstance(node.left, ast.Str):
                    if any(keyword in node.left.s.upper() 
                          for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                        self.issues.append({
                            'file': str(file_path),
                            'line': node.lineno,
                            'severity': 'CRITICAL',
                            'issue': 'Potential SQL injection',
                            'recommendation': 'Use parameterized queries'
                        })
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'["\']?[Aa][Pp][Ii]_?[Kk][Ee][Yy]["\']?\s*[:=]\s*["\'][^"\']+["\']', 'API key'),
            (r'["\']?[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]["\']?\s*[:=]\s*["\'][^"\']+["\']', 'Password'),
            (r'["\']?[Ss][Ee][Cc][Rr][Ee][Tt]["\']?\s*[:=]\s*["\'][^"\']+["\']', 'Secret'),
        ]
        
        for pattern, desc in secret_patterns:
            for match in re.finditer(pattern, content):
                self.issues.append({
                    'file': str(file_path),
                    'line': content[:match.start()].count('\n') + 1,
                    'severity': 'CRITICAL',
                    'issue': f'Hardcoded {desc}',
                    'recommendation': 'Use environment variables or secrets management'
                })
    
    def _review_dockerfile(self, file_path: Path):
        """Dockerfile security analysis"""
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            # Check for latest tags
            if re.match(r'^FROM\s+.*:latest', line):
                self.issues.append({
                    'file': str(file_path),
                    'line': i + 1,
                    'severity': 'MEDIUM',
                    'issue': 'Using :latest tag',
                    'recommendation': 'Pin to specific version'
                })
            
            # Check for running as root
            if re.match(r'^USER\s+root', line):
                self.issues.append({
                    'file': str(file_path),
                    'line': i + 1,
                    'severity': 'HIGH',
                    'issue': 'Running container as root',
                    'recommendation': 'Use non-root user'
                })
            
            # Check for sudo installation
            if 'sudo' in line and any(cmd in line for cmd in ['apt-get', 'yum', 'apk']):
                self.issues.append({
                    'file': str(file_path),
                    'line': i + 1,
                    'severity': 'MEDIUM',
                    'issue': 'Installing sudo in container',
                    'recommendation': 'Avoid sudo in containers'
                })
    
    def _scan_for_secrets(self, files: List[Path]):
        """Scan for leaked secrets using multiple tools"""
        
        # Run git-secrets
        try:
            result = subprocess.run(
                ['git', 'secrets', '--scan'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                for line in result.stderr.split('\n'):
                    if line and ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            self.issues.append({
                                'file': parts[0],
                                'line': int(parts[1]) if parts[1].isdigit() else 0,
                                'severity': 'CRITICAL',
                                'issue': 'Secret detected',
                                'recommendation': parts[2].strip()
                            })
        except subprocess.CalledProcessError:
            pass
        
        # Run detect-secrets
        try:
            result = subprocess.run(
                ['detect-secrets', 'scan', '--all-files'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                import json
                secrets_data = json.loads(result.stdout)
                for file_path, secrets in secrets_data.get('results', {}).items():
                    for secret in secrets:
                        self.issues.append({
                            'file': file_path,
                            'line': secret.get('line_number', 0),
                            'severity': 'CRITICAL',
                            'issue': f"Secret detected: {secret.get('type', 'Unknown')}",
                            'recommendation': 'Remove secret and rotate credentials'
                        })
        except:
            pass
    
    def _check_dependencies(self):
        """Check for vulnerable dependencies"""
        
        # Python dependencies
        if Path('requirements.txt').exists():
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    self.issues.append({
                        'file': 'requirements.txt',
                        'line': 0,
                        'severity': 'HIGH',
                        'issue': f"Vulnerable dependency: {vuln['package']} {vuln['installed_version']}",
                        'recommendation': f"Update to {vuln['closest_secure_version']}"
                    })
        
        # Node.js dependencies
        if Path('package-lock.json').exists():
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                audit_data = json.loads(result.stdout)
                for advisory_id, advisory in audit_data.get('advisories', {}).items():
                    self.issues.append({
                        'file': 'package.json',
                        'line': 0,
                        'severity': advisory['severity'].upper(),
                        'issue': f"Vulnerable dependency: {advisory['module_name']}",
                        'recommendation': advisory['recommendation']
                    })
    
    def _generate_report(self) -> Dict:
        """Generate security review report"""
        
        # Sort issues by severity
        self.issues.sort(
            key=lambda x: self.severity_levels.get(x['severity'], 0),
            reverse=True
        )
        
        # Calculate metrics
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }
        
        for issue in self.issues:
            severity_counts[issue['severity']] += 1
        
        # Risk score calculation
        risk_score = (
            severity_counts['CRITICAL'] * 10 +
            severity_counts['HIGH'] * 5 +
            severity_counts['MEDIUM'] * 2 +
            severity_counts['LOW'] * 1
        )
        
        return {
            'summary': {
                'total_issues': len(self.issues),
                'severity_counts': severity_counts,
                'risk_score': risk_score,
                'recommendation': 'BLOCK' if severity_counts['CRITICAL'] > 0 else 'REVIEW'
            },
            'issues': self.issues
        }

# GitHub PR integration
class GitHubSecurityBot:
    """Security bot for GitHub PRs"""
    
    def __init__(self, github_token: str):
        self.token = github_token
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def comment_on_pr(self, repo: str, pr_number: int, report: Dict):
        """Post security review as PR comment"""
        
        comment = self._format_report_comment(report)
        
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        response = requests.post(url, json={'body': comment}, headers=self.headers)
        
        return response.status_code == 201
    
    def _format_report_comment(self, report: Dict) -> str:
        """Format report as markdown comment"""
        
        summary = report['summary']
        
        comment = "## 🔒 Security Review Results\n\n"
        
        # Summary
        comment += f"**Risk Score:** {summary['risk_score']}/100\n"
        comment += f"**Recommendation:** {summary['recommendation']}\n\n"
        
        # Issue counts
        comment += "### Issue Summary\n"
        for severity, count in summary['severity_counts'].items():
            if count > 0:
                emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🔵'
                }.get(severity, '⚪')
                comment += f"- {emoji} {severity}: {count}\n"
        
        comment += "\n### Detailed Findings\n\n"
        
        # Group issues by file
        issues_by_file = {}
        for issue in report['issues']:
            file_name = issue['file']
            if file_name not in issues_by_file:
                issues_by_file[file_name] = []
            issues_by_file[file_name].append(issue)
        
        # Format issues
        for file_name, issues in issues_by_file.items():
            comment += f"#### `{file_name}`\n"
            for issue in issues:
                comment += f"- **Line {issue['line']}** [{issue['severity']}]: {issue['issue']}\n"
                comment += f"  - 💡 {issue['recommendation']}\n"
            comment += "\n"
        
        # Add automation notice
        comment += "\n---\n"
        comment += "_This automated security review is generated by SecurityBot. "
        comment += "Please address all CRITICAL and HIGH severity issues before merging._"
        
        return comment
```

### 9.3 Secure Development Environment

```bash
#!/bin/bash
# secure-dev-setup.sh - Set up secure development environment

set -euo pipefail

echo "🔒 Setting up secure development environment..."

# 1. Install security tools
install_security_tools() {
    echo "📦 Installing security tools..."
    
    # Python security tools
    pip install --user \
        bandit \
        safety \
        semgrep \
        detect-secrets \
        pip-audit
    
    # Node.js security tools
    npm install -g \
        snyk \
        retire \
        eslint-plugin-security \
        npm-audit-resolver
    
    # Go security tools
    go install github.com/securego/gosec/v2/cmd/gosec@latest
    go install golang.org/x/vuln/cmd/govulncheck@latest
    
    # Container security
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
    
    # Install Trivy
    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
}

# 2. Configure pre-commit hooks
setup_pre_commit() {
    echo "🪝 Setting up pre-commit hooks..."
    
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  # Security scanning
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        
  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-ll', '--skip', 'B101,B601']
        
  - repo: https://github.com/pre-commit/mirrors-semgrep
    rev: v1.45.0
    hooks:
      - id: semgrep
        args: ['--config=auto', '--error']
        
  # Code quality
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      
  # Dockerfile linting
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint
        
  # Kubernetes manifest validation
  - repo: https://github.com/syntaqx/kube-score
    rev: v1.16.1
    hooks:
      - id: kube-score
        files: \.(yaml|yml)$
        
  # Terraform security
  - repo: https://github.com/terraform-docs/terraform-docs
    rev: v0.16.0
    hooks:
      - id: terraform-docs-go
        args: ["markdown", "table", "--output-file", "README.md", "."]
        
  - repo: https://github.com/bridgecrewio/checkov.git
    rev: 3.1.0
    hooks:
      - id: checkov
        args: ['--framework', 'terraform,kubernetes']
EOF

    pre-commit install
    pre-commit install --hook-type commit-msg
    
    # Create secrets baseline
    detect-secrets scan > .secrets.baseline
}

# 3. Set up git security
configure_git_security() {
    echo "🔐 Configuring git security..."
    
    # Configure git-secrets
    git secrets --install
    git secrets --register-aws
    
    # Add custom patterns
    git secrets --add 'private_key'
    git secrets --add 'api[_-]?key'
    git secrets --add 'secret[_-]?key'
    git secrets --add 'access[_-]?token'
    
    # Enable commit signing
    git config --global commit.gpgsign true
    git config --global tag.gpgsign true
    
    # Set up SSH signing (Git 2.34+)
    git config --global gpg.format ssh
    git config --global user.signingkey ~/.ssh/id_ed25519.pub
}

# 4. IDE security plugins
setup_ide_security() {
    echo "💻 Setting up IDE security plugins..."
    
    # VS Code extensions
    if command -v code &> /dev/null; then
        code --install-extension ms-vscode.vscode-typescript-tslint-plugin
        code --install-extension dbaeumer.vscode-eslint
        code --install-extension ms-python.vscode-pylint
        code --install-extension golang.go
        code --install-extension redhat.vscode-yaml
        code --install-extension ms-azuretools.vscode-docker
        code --install-extension shardulm94.trailing-spaces
        code --install-extension timonwong.shellcheck
    fi
    
    # Create VS Code settings
    mkdir -p .vscode
    cat > .vscode/settings.json << 'EOF'
{
    "files.exclude": {
        "**/.git": true,
        "**/.DS_Store": true,
        "**/node_modules": true,
        "**/__pycache__": true
    },
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.banditEnabled": true,
    "go.lintTool": "golangci-lint",
    "go.lintFlags": [
        "--enable-all"
    ],
    "eslint.validate": [
        "javascript",
        "javascriptreact",
        "typescript",
        "typescriptreact"
    ]
}
EOF
}

# 5. Security aliases
setup_security_aliases() {
    echo "🚀 Setting up security aliases..."
    
    cat >> ~/.bashrc << 'EOF'

# Security aliases
alias security-scan='bandit -r . && safety check && npm audit'
alias docker-scan='trivy image'
alias k8s-scan='kube-score scan'
alias secrets-scan='detect-secrets scan'
alias vuln-check='grype dir:.'

# Security functions
security-review() {
    echo "🔍 Running comprehensive security review..."
    
    # Code scanning
    semgrep --config=auto .
    
    # Dependency scanning
    if [ -f "requirements.txt" ]; then
        safety check
        pip-audit
    fi
    
    if [ -f "package.json" ]; then
        npm audit
        retire --js
    fi
    
    if [ -f "go.mod" ]; then
        gosec ./...
        govulncheck ./...
    fi
    
    # Secret scanning
    detect-secrets scan
    
    # Container scanning
    if [ -f "Dockerfile" ]; then
        hadolint Dockerfile
    fi
}

# Git pre-push security check
git-security-push() {
    security-review
    if [ $? -eq 0 ]; then
        git push "$@"
    else
        echo "❌ Security issues detected. Fix before pushing."
        return 1
    fi
}
EOF
}

# Main execution
main() {
    install_security_tools
    setup_pre_commit
    configure_git_security
    setup_ide_security
    setup_security_aliases
    
    echo "✅ Secure development environment setup complete!"
    echo "🔒 Remember to:"
    echo "   - Run 'source ~/.bashrc' to load security aliases"
    echo "   - Configure your GPG/SSH key for commit signing"
    echo "   - Run 'security-review' before committing code"
}

main "$@"
```

---

## 10. Advanced Threat Hunting

### 10.1 Threat Hunting with YARA

```python
#!/usr/bin/env python3
# threat_hunter.py - Advanced threat hunting system

import yara
import os
import hashlib
import magic
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Set
import json
from datetime import datetime
import psutil
import struct

class ThreatHunter:
    """Advanced threat hunting using YARA and behavioral analysis"""
    
    def __init__(self, rules_dir: str):
        self.rules = self._load_yara_rules(rules_dir)
        self.file_magic = magic.Magic(mime=True)
        self.whitelist = self._load_whitelist()
        self.alerts = []
        
    def _load_yara_rules(self, rules_dir: str) -> yara.Rules:
        """Compile all YARA rules"""
        
        rule_files = {}
        for rule_file in Path(rules_dir).glob("*.yar"):
            namespace = rule_file.stem
            rule_files[namespace] = str(rule_file)
        
        return yara.compile(filepaths=rule_files)
    
    async def hunt_filesystem(self, target_dirs: List[str]):
        """Hunt for threats in filesystem"""
        
        tasks = []
        for target_dir in target_dirs:
            tasks.append(self._scan_directory(target_dir))
        
        await asyncio.gather(*tasks)
        
        return self.alerts
    
    async def _scan_directory(self, directory: str):
        """Recursively scan directory"""
        
        for root, dirs, files in os.walk(directory):
            # Skip system directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__']]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip if in whitelist
                if self._is_whitelisted(file_path):
                    continue
                
                await self._scan_file(file_path)
    
    async def _scan_file(self, file_path: Path):
        """Scan individual file"""
        
        try:
            # Check file size (skip very large files)
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                return
            
            # Get file metadata
            file_info = await self._get_file_info(file_path)
            
            # YARA scanning
            matches = self.rules.match(str(file_path))
            
            if matches:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'file': str(file_path),
                    'matches': [str(match) for match in matches],
                    'metadata': file_info,
                    'severity': self._calculate_severity(matches)
                }
                
                self.alerts.append(alert)
                
                # Additional analysis for high severity
                if alert['severity'] == 'CRITICAL':
                    alert['analysis'] = await self._deep_analysis(file_path)
        
        except Exception as e:
            # Log but continue scanning
            pass
    
    async def _get_file_info(self, file_path: Path) -> Dict:
        """Gather file metadata"""
        
        stat = file_path.stat()
        
        # Calculate hashes
        sha256_hash = await self._calculate_hash(file_path)
        
        return {
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'permissions': oct(stat.st_mode),
            'mime_type': self.file_magic.from_file(str(file_path)),
            'sha256': sha256_hash
        }
    
    async def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash asynchronously"""
        
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _deep_analysis(self, file_path: Path) -> Dict:
        """Perform deep analysis on suspicious files"""
        
        analysis = {
            'strings': await self._extract_strings(file_path),
            'imports': await self._analyze_imports(file_path),
            'entropy': await self._calculate_entropy(file_path),
            'packed': await self._check_packing(file_path)
        }
        
        return analysis
    
    async def _extract_strings(self, file_path: Path, min_length: int = 6) -> List[str]:
        """Extract ASCII and Unicode strings"""
        
        strings = []
        
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
        
        # ASCII strings
        ascii_pattern = b'[\x20-\x7e]{%d,}' % min_length
        strings.extend([s.decode('ascii') for s in re.findall(ascii_pattern, data)])
        
        # Unicode strings
        unicode_pattern = b'(?:[\x20-\x7e]\x00){%d,}' % min_length
        unicode_strings = re.findall(unicode_pattern, data)
        strings.extend([s.decode('utf-16-le', errors='ignore') for s in unicode_strings])
        
        # Filter interesting strings
        interesting = []
        suspicious_patterns = [
            'cmd.exe', 'powershell', 'http://', 'https://',
            'HKEY_', 'registry', 'password', 'admin',
            '.onion', 'bitcoin', 'ransom'
        ]
        
        for s in strings:
            if any(pattern in s.lower() for pattern in suspicious_patterns):
                interesting.append(s)
        
        return interesting[:50]  # Limit to 50 most interesting
    
    async def _calculate_entropy(self, file_path: Path) -> float:
        """Calculate file entropy (high entropy may indicate encryption/packing)"""
        
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
        
        if not data:
            return 0.0
        
        # Calculate frequency of each byte
        frequency = {}
        for byte in data:
            frequency[byte] = frequency.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in frequency.values():
            if count > 0:
                freq = float(count) / data_len
                entropy -= freq * math.log2(freq)
        
        return entropy

# YARA Rules for threat hunting
YARA_RULES = """
// APT Detection Rules

rule APT_Lazarus_Dropper {
    meta:
        description = "Detects Lazarus Group dropper"
        severity = "critical"
        
    strings:
        $magic = { 4D 5A }  // MZ header
        $s1 = "Lazarus" nocase
        $s2 = "GetProcAddress"
        $s3 = "LoadLibrary"
        $decode = { 8B 45 ?? 0F B6 00 34 ?? 88 45 ?? }
        
    condition:
        $magic at 0 and 2 of ($s*) and $decode
}

rule Ransomware_Generic {
    meta:
        description = "Generic ransomware indicators"
        severity = "critical"
        
    strings:
        $ransom1 = "Your files have been encrypted" nocase
        $ransom2 = "Bitcoin" nocase
        $ransom3 = "decrypt" nocase
        $ransom4 = ".locked" nocase
        $ransom5 = "readme.txt" nocase
        
        $crypto1 = "CryptEncrypt"
        $crypto2 = "AES_256"
        $crypto3 = "RSA"
        
        $api1 = "FindFirstFileW"
        $api2 = "FindNextFileW"
        $api3 = "CreateFileW"
        $api4 = "WriteFile"
        $api5 = "MoveFileW"
        
    condition:
        3 of ($ransom*) and 2 of ($crypto*) and 3 of ($api*)
}

rule Webshell_PHP {
    meta:
        description = "PHP webshell detection"
        severity = "high"
        
    strings:
        $php = "<?php"
        $s1 = "eval(" nocase
        $s2 = "base64_decode(" nocase
        $s3 = "system(" nocase
        $s4 = "shell_exec(" nocase
        $s5 = "passthru(" nocase
        $s6 = "exec(" nocase
        $s7 = "$_POST" nocase
        $s8 = "$_GET" nocase
        $s9 = "$_REQUEST" nocase
        
    condition:
        $php and (
            (any of ($s1, $s2) and any of ($s7, $s8, $s9)) or
            (2 of ($s3, $s4, $s5, $s6) and any of ($s7, $s8, $s9))
        )
}

rule Cryptominer {
    meta:
        description = "Cryptocurrency miner detection"
        severity = "medium"
        
    strings:
        $s1 = "stratum+tcp://" nocase
        $s2 = "monero" nocase
        $s3 = "xmrig" nocase
        $s4 = "minergate" nocase
        $s5 = "cpuminer" nocase
        $s6 = "--donate-level"
        $s7 = "--cpu-priority"
        $s8 = "cryptonight"
        
    condition:
        2 of them
}

rule Persistence_Registry {
    meta:
        description = "Registry persistence mechanism"
        severity = "high"
        
    strings:
        $reg1 = "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
        $reg2 = "Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce"
        $reg3 = "CurrentControlSet\\Services"
        $reg4 = "RegCreateKeyEx"
        $reg5 = "RegSetValueEx"
        
    condition:
        (any of ($reg1, $reg2, $reg3)) and (any of ($reg4, $reg5))
}
"""

# Memory hunting
class MemoryHunter:
    """Hunt for threats in process memory"""
    
    def __init__(self):
        self.yara_rules = yara.compile(source=YARA_RULES)
        
    def hunt_processes(self) -> List[Dict]:
        """Scan all running processes"""
        
        alerts = []
        
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                # Skip system processes
                if proc.info['pid'] in [0, 4]:
                    continue
                
                # Scan process memory
                matches = self._scan_process(proc.info['pid'])
                
                if matches:
                    alerts.append({
                        'process': proc.info,
                        'matches': matches,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return alerts
    
    def _scan_process(self, pid: int) -> List[str]:
        """Scan process memory with YARA"""
        
        try:
            # This is platform-specific (Linux example)
            with open(f'/proc/{pid}/maps', 'r') as maps_file:
                memory_maps = maps_file.readlines()
            
            matches = []
            
            for line in memory_maps:
                # Parse memory region
                parts = line.split()
                if len(parts) < 6:
                    continue
                
                # Only scan readable regions
                if 'r' not in parts[1]:
                    continue
                
                # Parse address range
                addr_range = parts[0].split('-')
                start_addr = int(addr_range[0], 16)
                end_addr = int(addr_range[1], 16)
                
                # Read memory region
                try:
                    with open(f'/proc/{pid}/mem', 'rb') as mem_file:
                        mem_file.seek(start_addr)
                        data = mem_file.read(end_addr - start_addr)
                    
                    # Scan with YARA
                    region_matches = self.yara_rules.match(data=data)
                    
                    if region_matches:
                        matches.extend([str(m) for m in region_matches])
                
                except:
                    continue
            
            return matches
            
        except:
            return []

# EDR-style behavioral detection
class BehavioralDetector:
    """Detect malicious behavior patterns"""
    
    def __init__(self):
        self.monitored_processes = {}
        self.alerts = []
        
    async def monitor_system(self):
        """Continuous behavioral monitoring"""
        
        while True:
            # Monitor process creation
            current_procs = {p.pid: p.info for p in psutil.process_iter(['pid', 'name', 'ppid', 'create_time'])}
            
            # Detect new processes
            new_procs = set(current_procs.keys()) - set(self.monitored_processes.keys())
            
            for pid in new_procs:
                proc_info = current_procs[pid]
                
                # Check for suspicious patterns
                if self._is_suspicious_process(proc_info):
                    self.alerts.append({
                        'type': 'suspicious_process',
                        'process': proc_info,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Check for process injection indicators
                if self._check_process_injection(pid):
                    self.alerts.append({
                        'type': 'process_injection',
                        'pid': pid,
                        'timestamp': datetime.now().isoformat()
                    })
            
            self.monitored_processes = current_procs
            
            # Check for other behavioral patterns
            await self._check_network_connections()
            await self._check_file_operations()
            
            await asyncio.sleep(1)  # Check every second
    
    def _is_suspicious_process(self, proc_info: Dict) -> bool:
        """Check for suspicious process characteristics"""
        
        suspicious_names = [
            'powershell.exe', 'cmd.exe', 'wscript.exe',
            'cscript.exe', 'mshta.exe', 'rundll32.exe'
        ]
        
        # Check process name
        if proc_info['name'].lower() in suspicious_names:
            # Check if spawned by unusual parent
            try:
                parent = psutil.Process(proc_info['ppid'])
                if parent.name().lower() in ['winword.exe', 'excel.exe', 'outlook.exe']:
                    return True
            except:
                pass
        
        return False
```

---

## 11. Security Automation and Orchestration

### 11.1 Security Automation Framework

```python
#!/usr/bin/env python3
# security_automation.py - Comprehensive security automation

import asyncio
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import httpx
import json
from datetime import datetime, timedelta
import hashlib

@dataclass
class SecurityEvent:
    id: str
    timestamp: datetime
    source: str
    type: str
    severity: str
    data: Dict[str, Any]
    
class SecurityAction(ABC):
    """Base class for security actions"""
    
    @abstractmethod
    async def execute(self, event: SecurityEvent, context: Dict) -> Dict:
        pass
    
    @abstractmethod
    def validate(self, event: SecurityEvent) -> bool:
        pass

class BlockIPAction(SecurityAction):
    """Block IP address at firewall"""
    
    def __init__(self, firewall_api: str, api_key: str):
        self.firewall_api = firewall_api
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    async def execute(self, event: SecurityEvent, context: Dict) -> Dict:
        ip_address = event.data.get('source_ip')
        if not ip_address:
            return {'status': 'failed', 'reason': 'No IP address in event'}
        
        # Add to firewall blocklist
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.firewall_api}/rules",
                json={
                    'action': 'block',
                    'source': ip_address,
                    'duration': 3600,  # 1 hour
                    'reason': f'Automated block: {event.type}'
                },
                headers=self.headers
            )
        
        return {
            'status': 'success' if response.status_code == 201 else 'failed',
            'ip_blocked': ip_address,
            'rule_id': response.json().get('id')
        }
    
    def validate(self, event: SecurityEvent) -> bool:
        return 'source_ip' in event.data

class IsolateHostAction(SecurityAction):
    """Isolate compromised host from network"""
    
    def __init__(self, orchestrator: 'SecurityOrchestrator'):
        self.orchestrator = orchestrator
    
    async def execute(self, event: SecurityEvent, context: Dict) -> Dict:
        hostname = event.data.get('hostname')
        
        # Multiple isolation methods
        tasks = [
            self._isolate_via_edr(hostname),
            self._isolate_via_network(hostname),
            self._disable_user_accounts(hostname)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'status': 'success',
            'hostname': hostname,
            'isolation_methods': [
                'edr' if not isinstance(results[0], Exception) else 'edr_failed',
                'network' if not isinstance(results[1], Exception) else 'network_failed',
                'accounts' if not isinstance(results[2], Exception) else 'accounts_failed'
            ]
        }
    
    async def _isolate_via_edr(self, hostname: str):
        """Isolate using EDR API"""
        # Implementation specific to your EDR solution
        pass
    
    async def _isolate_via_network(self, hostname: str):
        """Network isolation via switch/firewall"""
        # VLAN isolation or ACL updates
        pass
    
    async def _disable_user_accounts(self, hostname: str):
        """Disable associated user accounts"""
        # AD/LDAP account suspension
        pass
    
    def validate(self, event: SecurityEvent) -> bool:
        return 'hostname' in event.data

class SecurityPlaybook:
    """Automated security response playbook"""
    
    def __init__(self, name: str, triggers: List[Dict], actions: List[Dict]):
        self.name = name
        self.triggers = triggers
        self.actions = actions
        self.execution_history = []
    
    def should_trigger(self, event: SecurityEvent) -> bool:
        """Check if playbook should be triggered"""
        
        for trigger in self.triggers:
            if self._match_trigger(trigger, event):
                return True
        return False
    
    def _match_trigger(self, trigger: Dict, event: SecurityEvent) -> bool:
        """Match event against trigger conditions"""
        
        # Check event type
        if trigger.get('event_type') != event.type:
            return False
        
        # Check severity
        if trigger.get('min_severity'):
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            if severity_map.get(event.severity, 0) < severity_map.get(trigger['min_severity'], 0):
                return False
        
        # Check conditions
        for condition in trigger.get('conditions', []):
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            event_value = self._get_nested_value(event.data, field)
            
            if not self._evaluate_condition(event_value, operator, value):
                return False
        
        return True
    
    def _get_nested_value(self, data: Dict, field: str) -> Any:
        """Get nested dictionary value using dot notation"""
        
        parts = field.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        
        return value
    
    def _evaluate_condition(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate condition"""
        
        operators = {
            'equals': lambda a, e: a == e,
            'not_equals': lambda a, e: a != e,
            'contains': lambda a, e: e in str(a),
            'greater_than': lambda a, e: float(a) > float(e),
            'less_than': lambda a, e: float(a) < float(e),
            'in': lambda a, e: a in e,
            'regex': lambda a, e: bool(re.match(e, str(a)))
        }
        
        return operators.get(operator, lambda a, e: False)(actual, expected)

class SecurityOrchestrator:
    """Main orchestration engine"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.playbooks = self._load_playbooks()
        self.actions = self._initialize_actions()
        self.event_queue = asyncio.Queue()
        self.metrics = {
            'events_processed': 0,
            'playbooks_executed': 0,
            'actions_succeeded': 0,
            'actions_failed': 0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_playbooks(self) -> List[SecurityPlaybook]:
        """Load playbooks from configuration"""
        
        playbooks = []
        
        for pb_config in self.config.get('playbooks', []):
            playbook = SecurityPlaybook(
                name=pb_config['name'],
                triggers=pb_config['triggers'],
                actions=pb_config['actions']
            )
            playbooks.append(playbook)
        
        return playbooks
    
    def _initialize_actions(self) -> Dict[str, SecurityAction]:
        """Initialize available actions"""
        
        actions = {
            'block_ip': BlockIPAction(
                self.config['integrations']['firewall']['api'],
                self.config['integrations']['firewall']['api_key']
            ),
            'isolate_host': IsolateHostAction(self),
            # Add more actions...
        }
        
        return actions
    
    async def process_event(self, event: SecurityEvent):
        """Process security event through playbooks"""
        
        self.metrics['events_processed'] += 1
        
        # Check each playbook
        for playbook in self.playbooks:
            if playbook.should_trigger(event):
                await self.execute_playbook(playbook, event)
    
    async def execute_playbook(self, playbook: SecurityPlaybook, event: SecurityEvent):
        """Execute playbook actions"""
        
        self.metrics['playbooks_executed'] += 1
        
        context = {
            'playbook': playbook.name,
            'event': event,
            'results': []
        }
        
        # Execute actions in sequence
        for action_config in playbook.actions:
            action_name = action_config['action']
            action = self.actions.get(action_name)
            
            if not action:
                continue
            
            # Validate action
            if not action.validate(event):
                continue
            
            try:
                # Execute action
                result = await action.execute(event, context)
                context['results'].append({
                    'action': action_name,
                    'status': 'success',
                    'result': result
                })
                self.metrics['actions_succeeded'] += 1
                
            except Exception as e:
                context['results'].append({
                    'action': action_name,
                    'status': 'failed',
                    'error': str(e)
                })
                self.metrics['actions_failed'] += 1
                
                # Check if should continue on failure
                if not action_config.get('continue_on_failure', True):
                    break
        
        # Log execution
        await self._log_execution(playbook, event, context)
    
    async def _log_execution(self, playbook: SecurityPlaybook, 
                           event: SecurityEvent, context: Dict):
        """Log playbook execution"""
        
        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'playbook': playbook.name,
            'event_id': event.id,
            'event_type': event.type,
            'results': context['results'],
            'duration': (datetime.now() - event.timestamp).total_seconds()
        }
        
        # Store in database or SIEM
        # ...

# Configuration example
ORCHESTRATOR_CONFIG = """
integrations:
  firewall:
    api: https://firewall.internal/api/v1
    api_key: ${FIREWALL_API_KEY}
  
  edr:
    api: https://edr.internal/api/v2
    api_key: ${EDR_API_KEY}
  
  siem:
    api: https://siem.internal/api/v1
    api_key: ${SIEM_API_KEY}

playbooks:
  - name: respond_to_brute_force
    triggers:
      - event_type: authentication_failure
        min_severity: high
        conditions:
          - field: failure_count
            operator: greater_than
            value: 10
          - field: time_window
            operator: less_than
            value: 300  # 5 minutes
    
    actions:
      - action: block_ip
        duration: 3600
        continue_on_failure: true
      
      - action: create_incident
        severity: high
        assign_to: security_team
  
  - name: respond_to_malware
    triggers:
      - event_type: malware_detected
        min_severity: high
        conditions:
          - field: confidence
            operator: greater_than
            value: 0.8
    
    actions:
      - action: isolate_host
        continue_on_failure: false
      
      - action: collect_forensics
        tools: [memory_dump, process_list, network_connections]
      
      - action: create_incident
        severity: critical
        assign_to: incident_response_team
      
      - action: notify
        channels: [email, slack]
        recipients: [security-team@company.com]

  - name: respond_to_data_exfiltration
    triggers:
      - event_type: anomalous_upload
        conditions:
          - field: data_size_mb
            operator: greater_than
            value: 100
          - field: destination.reputation
            operator: equals
            value: malicious
    
    actions:
      - action: block_connection
        immediate: true
      
      - action: preserve_evidence
        types: [network_capture, file_hash]
      
      - action: isolate_user
        disable_accounts: true
      
      - action: legal_hold
        data_sources: [email, files]
"""

# Advanced automation rules engine
class RulesEngine:
    """Complex event processing and correlation"""
    
    def __init__(self):
        self.rules = []
        self.event_window = {}
        self.correlation_timeout = 300  # 5 minutes
    
    def add_rule(self, rule: 'CorrelationRule'):
        self.rules.append(rule)
    
    async def process_event(self, event: SecurityEvent):
        """Process event through correlation rules"""
        
        # Add to event window
        event_key = f"{event.source}:{event.type}"
        if event_key not in self.event_window:
            self.event_window[event_key] = []
        
        self.event_window[event_key].append(event)
        
        # Clean old events
        cutoff = datetime.now() - timedelta(seconds=self.correlation_timeout)
        self.event_window[event_key] = [
            e for e in self.event_window[event_key] 
            if e.timestamp > cutoff
        ]
        
        # Check correlation rules
        for rule in self.rules:
            if rule.check(self.event_window):
                await rule.trigger(self.event_window)

class CorrelationRule:
    """Multi-event correlation rule"""
    
    def __init__(self, name: str, conditions: List[Dict], 
                 time_window: int, threshold: int):
        self.name = name
        self.conditions = conditions
        self.time_window = time_window
        self.threshold = threshold
    
    def check(self, event_window: Dict[str, List[SecurityEvent]]) -> bool:
        """Check if correlation conditions are met"""
        
        matching_events = []
        
        for event_key, events in event_window.items():
            for event in events:
                if self._matches_conditions(event):
                    matching_events.append(event)
        
        # Check if we have enough events in time window
        if len(matching_events) >= self.threshold:
            # Check time window
            earliest = min(e.timestamp for e in matching_events)
            latest = max(e.timestamp for e in matching_events)
            
            if (latest - earliest).total_seconds() <= self.time_window:
                return True
        
        return False
    
    def _matches_conditions(self, event: SecurityEvent) -> bool:
        """Check if event matches rule conditions"""
        # Implementation similar to playbook triggers
        pass
    
    async def trigger(self, event_window: Dict[str, List[SecurityEvent]]):
        """Trigger actions when rule matches"""
        
        # Create composite event
        composite_event = SecurityEvent(
            id=hashlib.sha256(self.name.encode()).hexdigest()[:16],
            timestamp=datetime.now(),
            source='correlation_engine',
            type=f'correlated_{self.name}',
            severity='high',
            data={
                'rule': self.name,
                'event_count': sum(len(events) for events in event_window.values()),
                'sources': list(event_window.keys())
            }
        )
        
        # Send to orchestrator
        # ...

# Usage example
async def main():
    """Initialize and run security automation"""
    
    # Create orchestrator
    orchestrator = SecurityOrchestrator('config/orchestrator.yaml')
    
    # Create rules engine
    rules_engine = RulesEngine()
    
    # Add correlation rule for distributed brute force
    rules_engine.add_rule(CorrelationRule(
        name='distributed_brute_force',
        conditions=[
            {'field': 'type', 'operator': 'equals', 'value': 'authentication_failure'},
            {'field': 'target_user', 'operator': 'equals', 'value': 'admin'}
        ],
        time_window=600,  # 10 minutes
        threshold=50  # 50 failures from different sources
    ))
    
    # Start processing events
    # In production, this would integrate with your SIEM/log aggregation
    await orchestrator.start()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 12. Conclusion and Best Practices Summary

### Security Engineering Principles for 2025

1. **Defense in Depth**: Layer multiple security controls - no single point of failure
2. **Zero Trust Architecture**: Never trust, always verify - especially in containerized/cloud environments  
3. **Shift Left Security**: Integrate security from the first line of code
4. **Automation First**: Automate detection, response, and remediation
5. **Continuous Validation**: Security is not a checkbox - continuously test and improve

### Quick Reference Security Checklist

```bash
#!/bin/bash
# security-checklist.sh - Daily security validation

echo "🔒 Security Validation Checklist"
echo "================================"

# Code Security
echo "[ ] Code: No secrets in repository (git-secrets, detect-secrets)"
echo "[ ] Code: Dependencies scanned and updated (npm audit, safety, cargo audit)"
echo "[ ] Code: SAST scan passed (semgrep, codeql, gosec)"
echo "[ ] Code: License compliance verified"

# Runtime Security  
echo "[ ] Runtime: SELinux/AppArmor enforcing"
echo "[ ] Runtime: Seccomp profiles applied"
echo "[ ] Runtime: CPU mitigations enabled"
echo "[ ] Runtime: ASLR and DEP active"

# Network Security
echo "[ ] Network: Firewall rules reviewed"
echo "[ ] Network: IDS/IPS signatures updated"
echo "[ ] Network: VPN and zero-trust controls active"
echo "[ ] Network: Certificate expiration checked"

# Container/Cloud Security
echo "[ ] Containers: Images scanned for vulnerabilities"
echo "[ ] Containers: Running as non-root"
echo "[ ] Containers: Resource limits enforced"
echo "[ ] Containers: Network policies applied"

# Monitoring & Response
echo "[ ] Monitoring: SIEM collecting all logs"
echo "[ ] Monitoring: Alerts configured and tested"
echo "[ ] Response: Playbooks updated and tested"
echo "[ ] Response: Backup and recovery verified"

echo ""
echo "Run 'security-review' for detailed analysis"
```

### Final Recommendations

1. **Stay Updated**: Security is an arms race - continuously update tools and knowledge
2. **Practice Incident Response**: Regular drills prevent panic during real incidents
3. **Embrace Transparency**: Use open-source tools and contribute back to the community
4. **Measure Everything**: You can't improve what you don't measure
5. **Cultivate Paranoia**: In security, healthy paranoia is a professional requirement

Remember: Perfect security is impossible, but making attacks expensive and noisy is achievable. Focus on raising the cost for attackers while maintaining usability for legitimate users.

---

*This guide represents security best practices as of mid-2025. The security landscape evolves rapidly - always verify current tool versions and emerging threats.*            