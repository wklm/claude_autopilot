# The Definitive Guide to Hardware Development & Hacking (Mid-2025 Edition)

This guide synthesizes modern best practices for building production-grade hardware projects, from FPGA development to PCB design and mechanical integration. It covers the complete hardware development stack with an emphasis on professional workflows, testability, and manufacturability.

### Prerequisites & Tool Versions
- **KiCad 9.0+** for PCB design (the Altium importer is finally production-ready)
- **Vivado 2025.1+** or **Quartus Prime Pro 25.1** for FPGA development  
- **OnShape** with Enterprise features for mechanical CAD
- **Python 3.13+** for test automation and tooling
- **Rust 1.85+** for embedded firmware (where applicable)
- **VSCode** with hardware extensions or **Neovim** with coc-verilog

### Essential Hardware Lab Setup
```yaml
# .lab-config.yaml
instruments:
  scope: 
    model: "Rigol MSO5354"  # 350MHz, 4ch, built-in AWG
    calibration_due: "2025-07-15"
  logic_analyzer:
    model: "Saleae Logic Pro 16"
    channels: 16
  power_supply:
    model: "Rigol DP832"
    channels: 3
  
debug_tools:
  - "J-Link Pro"
  - "Black Magic Probe v2.3"
  - "Glasgow Interface Explorer"
  
environment:
  esd_mat: true
  temperature_controlled: true
  humidity_range: "30-50%"
```

---

## 1. Project Structure & Version Control

Hardware projects require special consideration for binary files, generated outputs, and manufacturing data. Use a consistent structure across all projects.

### ✅ DO: Use a Standardized Project Layout

```
/project-root
├── .gitignore              # Hardware-specific ignore patterns
├── .gitattributes          # LFS tracking for binaries
├── hardware/               # All hardware design files
│   ├── pcb/                # KiCad project files
│   │   ├── project.kicad_pro
│   │   ├── project.kicad_sch
│   │   ├── project.kicad_pcb
│   │   ├── libraries/      # Custom symbols/footprints
│   │   └── outputs/        # Generated files (Gerbers, etc.)
│   ├── mechanical/         # OnShape exports and specs
│   │   ├── enclosure/      # STEP files, drawings
│   │   └── assembly/       # Assembly instructions
│   └── simulation/         # LTspice, IBIS models
├── fpga/                   # FPGA/CPLD designs
│   ├── rtl/                # Verilog/VHDL source
│   ├── constraints/        # Pin assignments, timing
│   ├── testbench/          # Verification code
│   └── bitstreams/         # Generated bitfiles
├── firmware/               # Microcontroller code
│   ├── src/
│   ├── hal/                # Hardware abstraction layer
│   └── tests/
├── software/               # Host-side tools
│   ├── drivers/            # OS drivers if needed
│   ├── cli/                # Command-line tools
│   └── gui/                # Configuration GUIs
├── tests/                  # Hardware test procedures
│   ├── production/         # Manufacturing tests
│   ├── validation/         # Design validation
│   └── compliance/         # EMC, safety testing
├── docs/                   # Comprehensive documentation
│   ├── design/             # Architecture decisions
│   ├── datasheets/         # Component datasheets
│   └── manufacturing/      # Assembly documentation
└── tools/                  # Build scripts, automation
```

### ✅ DO: Configure Git LFS for Binary Files

Hardware projects generate large binary files. Use Git LFS to keep the repository size manageable.

```bash
# .gitattributes
# KiCad
*.kicad_pcb filter=kicad_pcb
*.kicad_sch filter=kicad_sch
*.lib filter=lfs diff=lfs merge=lfs -text
*.pretty filter=lfs diff=lfs merge=lfs -text

# Mechanical
*.step filter=lfs diff=lfs merge=lfs -text
*.stl filter=lfs diff=lfs merge=lfs -text
*.f3d filter=lfs diff=lfs merge=lfs -text

# FPGA
*.bit filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.mcs filter=lfs diff=lfs merge=lfs -text

# Documentation
*.pdf filter=lfs diff=lfs merge=lfs -text
datasheets/*.pdf filter=lfs diff=lfs merge=lfs -text

# Images
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
```

### ✅ DO: Use Semantic Versioning for Hardware

```python
# tools/version.py
"""
Hardware Semantic Versioning:
MAJOR.MINOR.PATCH-SPIN

MAJOR: Incompatible changes (different connectors, form factor)
MINOR: New features (added functionality, compatible changes)  
PATCH: Bug fixes (layout improvements, BOM optimizations)
SPIN: PCB revision (same schematic, layout changes only)

Example: 2.3.1-A (Version 2.3.1, PCB spin A)
"""
```

---

## 2. KiCad PCB Design Best Practices

KiCad 9.0 brings significant improvements including better Altium import, advanced DRC, and real-time 3D visualization.

### ✅ DO: Establish a Consistent Design Workflow

**1. Schematic Design Phase**

```python
# tools/schematic_checks.py
import subprocess
import json
from pathlib import Path

class SchematicValidator:
    """Automated schematic validation before PCB layout"""
    
    def __init__(self, project_path: Path):
        self.project = project_path
        self.errors = []
    
    def check_power_nets(self):
        """Ensure all power nets are properly connected"""
        # Use KiCad Python API
        import pcbnew
        board = pcbnew.LoadBoard(str(self.project))
        
        power_nets = ['VCC', 'VDD', '3V3', '5V', 'GND', 'AGND', 'DGND']
        for net_name in power_nets:
            net = board.GetNetsByName().get(net_name)
            if net and net.GetNodesCount() < 2:
                self.errors.append(f"Power net {net_name} has insufficient connections")
    
    def check_designators(self):
        """Verify all components have unique designators"""
        # Parse schematic JSON export
        sch_file = self.project.with_suffix('.kicad_sch')
        # ... validation logic
```

**2. Component Library Management**

```bash
# libraries/setup_libraries.sh
#!/bin/bash

# Clone official KiCad libraries
git clone https://github.com/KiCad/kicad-symbols.git libraries/kicad-symbols
git clone https://github.com/KiCad/kicad-footprints.git libraries/kicad-footprints

# Add organization-specific libraries
git submodule add git@company.com:hardware/company-symbols.git libraries/company-symbols
git submodule add git@company.com:hardware/verified-footprints.git libraries/verified-footprints

# Generate library tables
python tools/generate_lib_tables.py
```

### ✅ DO: Implement Design Rule Checks (DRC)

Create comprehensive DRC rules that go beyond KiCad defaults:

```python
# hardware/pcb/custom_rules.dru
(version 1)

# Differential pair rules
(rule "USB_Differential_Pairs"
    (condition "A.NetClass == 'USB_DIFF'")
    (constraint track_width (opt 0.2mm))
    (constraint diff_pair_gap (opt 0.15mm))
    (constraint diff_pair_uncoupled (max 5mm)))

# High-speed signal rules  
(rule "DRAM_Length_Match"
    (condition "A.NetClass == 'DRAM'")
    (constraint length (min 45mm) (max 55mm))
    (constraint via_count (max 2)))

# Thermal relief for power planes
(rule "Power_Thermal_Relief"
    (condition "A.Type == 'Pad' && A.NetName == 'GND'")
    (constraint thermal_spoke_width (min 0.4mm))
    (constraint thermal_relief_gap (min 0.2mm)))

# Courtyard overlap check
(rule "Component_Courtyard"
    (condition "A.Type == 'Footprint'")
    (constraint courtyard_clearance (min 0.5mm)))
```

### ✅ DO: Automate Manufacturing Outputs

```python
# tools/generate_production_files.py
#!/usr/bin/env python3
"""Generate all manufacturing files with one command"""

import pcbnew
import os
from pathlib import Path
from datetime import datetime

class ProductionFileGenerator:
    def __init__(self, board_file: str):
        self.board = pcbnew.LoadBoard(board_file)
        self.output_dir = Path("hardware/pcb/outputs")
        self.version = self._get_version()
        
    def generate_all(self):
        """Generate complete manufacturing package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.output_dir / f"fab_package_{self.version}_{timestamp}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        self.generate_gerbers(package_dir / "gerbers")
        self.generate_drill_files(package_dir / "drill")
        self.generate_pick_place(package_dir / "assembly")
        self.generate_3d_renders(package_dir / "3d")
        self.generate_bom(package_dir / "bom")
        self.create_readme(package_dir)
        
        # Create ZIP for upload
        self._create_zip(package_dir)
        
    def generate_gerbers(self, output_dir: Path):
        """Generate Gerbers with PCBWay/JLCPCB compatibility"""
        plot_controller = pcbnew.PLOT_CONTROLLER(self.board)
        plot_options = plot_controller.GetPlotOptions()
        
        # Configure for manufacturing
        plot_options.SetOutputDirectory(str(output_dir))
        plot_options.SetPlotFrameRef(False)
        plot_options.SetPlotValue(True)
        plot_options.SetPlotReference(True)
        plot_options.SetPlotInvisibleText(False)
        plot_options.SetPlotViaOnMaskLayer(False)
        plot_options.SetCreateGerberJobFile(True)
        plot_options.SetUseGerberProtelExtensions(False)
        plot_options.SetUseGerberX2format(True)
        plot_options.SetIncludeGerberNetlistInfo(True)
        plot_options.SetDisableGerberMacros(False)
        
        # Layer mapping for standard stackup
        layers = [
            ("F.Cu", pcbnew.F_Cu, "Top Copper"),
            ("B.Cu", pcbnew.B_Cu, "Bottom Copper"),
            ("F.Paste", pcbnew.F_Paste, "Top Paste"),
            ("B.Paste", pcbnew.B_Paste, "Bottom Paste"),
            ("F.SilkS", pcbnew.F_SilkS, "Top Silk"),
            ("B.SilkS", pcbnew.B_SilkS, "Bottom Silk"),
            ("F.Mask", pcbnew.F_Mask, "Top Mask"),
            ("B.Mask", pcbnew.B_Mask, "Bottom Mask"),
            ("Edge.Cuts", pcbnew.Edge_Cuts, "Board Outline"),
        ]
        
        # Add internal layers for 4+ layer boards
        for i in range(1, self.board.GetCopperLayerCount() - 1):
            layer_name = f"In{i}.Cu"
            layers.append((layer_name, (i-1)*2 + pcbnew.In1_Cu, f"Inner {i}"))
            
        for layer_name, layer_id, description in layers:
            plot_controller.SetLayer(layer_id)
            plot_controller.OpenPlotfile(layer_name, pcbnew.PLOT_FORMAT_GERBER, description)
            plot_controller.PlotLayer()
```

### ❌ DON'T: Ignore Stackup and Impedance Control

For any high-speed or RF design, always specify the PCB stackup explicitly:

```yaml
# hardware/pcb/stackup.yaml
stackup:
  layers: 4
  thickness: 1.6mm
  material: "FR-4 TG170"
  finish: "ENIG"
  
  dielectric:
    - layer: "1-2"
      thickness: 0.36mm
      er: 4.5
      loss_tangent: 0.02
    - layer: "2-3"  
      thickness: 0.71mm
      er: 4.5
      loss_tangent: 0.02
    - layer: "3-4"
      thickness: 0.36mm
      er: 4.5
      loss_tangent: 0.02
      
  copper:
    outer: "1oz (35um)"
    inner: "0.5oz (17um)"
    
impedance_control:
  - net_class: "USB_DIFF"
    target: "90 ohm differential"
    tolerance: "±10%"
    reference_plane: "GND"
  - net_class: "RGMII"
    target: "50 ohm single-ended"
    tolerance: "±10%"
    reference_plane: "GND"
```

---

## 3. FPGA Development with Modern Verilog

Modern FPGA development has moved beyond basic RTL to embrace verification-first design, formal methods, and high-level synthesis where appropriate.

### ✅ DO: Use SystemVerilog for New Designs

SystemVerilog (IEEE 1800-2023) provides powerful features for both design and verification:

```systemverilog
// rtl/axi4_stream_fifo.sv
`default_nettype none
`timescale 1ns/1ps

module axi4_stream_fifo #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH = 16,
    parameter int ID_WIDTH = 0,
    parameter int DEST_WIDTH = 0,
    parameter int USER_WIDTH = 1,
    parameter bit FALL_THROUGH = 1'b0,  // First-word fall-through
    parameter bit BACKPRESSURE_THRESHOLD = 0.75,
    // Derived parameters
    localparam int ADDR_WIDTH = $clog2(DEPTH)
) (
    // Clock and Reset
    input  wire                     clk,
    input  wire                     rst_n,
    
    // AXI4-Stream Slave Interface
    input  wire [DATA_WIDTH-1:0]    s_axis_tdata,
    input  wire [DATA_WIDTH/8-1:0]  s_axis_tkeep,
    input  wire                     s_axis_tlast,
    input  wire [ID_WIDTH-1:0]      s_axis_tid,
    input  wire [DEST_WIDTH-1:0]    s_axis_tdest,
    input  wire [USER_WIDTH-1:0]    s_axis_tuser,
    input  wire                     s_axis_tvalid,
    output logic                    s_axis_tready,
    
    // AXI4-Stream Master Interface  
    output logic [DATA_WIDTH-1:0]   m_axis_tdata,
    output logic [DATA_WIDTH/8-1:0] m_axis_tkeep,
    output logic                    m_axis_tlast,
    output logic [ID_WIDTH-1:0]     m_axis_tid,
    output logic [DEST_WIDTH-1:0]   m_axis_tdest,
    output logic [USER_WIDTH-1:0]   m_axis_tuser,
    output logic                    m_axis_tvalid,
    input  wire                     m_axis_tready,
    
    // Status Interface
    output logic [ADDR_WIDTH:0]     fifo_count,
    output logic                    fifo_empty,
    output logic                    fifo_full,
    output logic                    fifo_almost_full
);

    // Internal storage
    typedef struct packed {
        logic [DATA_WIDTH-1:0]   data;
        logic [DATA_WIDTH/8-1:0] keep;
        logic                    last;
        logic [ID_WIDTH-1:0]     id;
        logic [DEST_WIDTH-1:0]   dest;
        logic [USER_WIDTH-1:0]   user;
    } axis_payload_t;
    
    axis_payload_t mem [DEPTH];
    
    // Pointers and counters
    logic [ADDR_WIDTH-1:0] wr_ptr, rd_ptr;
    logic [ADDR_WIDTH:0]   count;
    logic                  empty, full;
    
    // Write logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            mem[wr_ptr] <= '{
                data: s_axis_tdata,
                keep: s_axis_tkeep,
                last: s_axis_tlast,
                id:   (ID_WIDTH > 0) ? s_axis_tid : '0,
                dest: (DEST_WIDTH > 0) ? s_axis_tdest : '0,
                user: s_axis_tuser
            };
            wr_ptr <= wr_ptr + 1'b1;
        end
    end
    
    // Read logic with optional fall-through
    generate
        if (FALL_THROUGH) begin : gen_fall_through
            // Combinational read for first-word fall-through
            always_comb begin
                if (!empty) begin
                    {m_axis_tdata, m_axis_tkeep, m_axis_tlast, 
                     m_axis_tid, m_axis_tdest, m_axis_tuser} = mem[rd_ptr];
                end else begin
                    {m_axis_tdata, m_axis_tkeep, m_axis_tlast,
                     m_axis_tid, m_axis_tdest, m_axis_tuser} = '0;
                end
            end
        end else begin : gen_registered
            // Registered output for better timing
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    {m_axis_tdata, m_axis_tkeep, m_axis_tlast,
                     m_axis_tid, m_axis_tdest, m_axis_tuser} <= '0;
                end else if (!empty && m_axis_tready) begin
                    {m_axis_tdata, m_axis_tkeep, m_axis_tlast,
                     m_axis_tid, m_axis_tdest, m_axis_tuser} <= mem[rd_ptr];
                end
            end
        end
    endgenerate
    
    // Pointer update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
        end else if (m_axis_tvalid && m_axis_tready) begin
            rd_ptr <= rd_ptr + 1'b1;
        end
    end
    
    // Count and flags
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
        end else begin
            case ({s_axis_tvalid && s_axis_tready, 
                   m_axis_tvalid && m_axis_tready})
                2'b10: count <= count + 1'b1;  // Write only
                2'b01: count <= count - 1'b1;  // Read only
                default: count <= count;       // Both or neither
            endcase
        end
    end
    
    // Status flags
    assign empty = (count == 0);
    assign full = (count == DEPTH);
    assign fifo_almost_full = (count >= DEPTH * BACKPRESSURE_THRESHOLD);
    
    // Ready/valid signals
    assign s_axis_tready = !full;
    assign m_axis_tvalid = !empty;
    
    // External status
    assign fifo_count = count;
    assign fifo_empty = empty;
    assign fifo_full = full;

    // Assertions for verification
    `ifdef FORMAL
        // Assume inputs are stable when not acknowledged
        if (s_axis_tvalid && !s_axis_tready) begin
            assume property (@(posedge clk) 
                s_axis_tdata == $past(s_axis_tdata));
        end
        
        // Assert FIFO never overflows or underflows
        assert property (@(posedge clk) 
            !(s_axis_tvalid && !s_axis_tready && full));
        assert property (@(posedge clk) 
            !(m_axis_tready && !m_axis_tvalid && empty));
            
        // Cover important scenarios
        cover property (@(posedge clk) 
            fifo_almost_full ##1 !fifo_almost_full);
    `endif

endmodule : axi4_stream_fifo
```

### ✅ DO: Implement Comprehensive Testbenches

Use SystemVerilog's verification features including constrained random testing:

```systemverilog
// testbench/axi4_stream_fifo_tb.sv
`timescale 1ns/1ps

class AXI4StreamTransaction #(parameter int DATA_WIDTH = 32);
    rand bit [DATA_WIDTH-1:0] data;
    rand bit [DATA_WIDTH/8-1:0] keep;
    rand bit last;
    rand int unsigned delay;
    
    constraint keep_valid {
        // Keep bits must be contiguous from LSB
        foreach (keep[i]) {
            if (i > 0 && keep[i] == 1'b1) {
                keep[i-1] == 1'b1;
            }
        }
    }
    
    constraint reasonable_delay {
        delay inside {[0:10]};
        delay dist {0:=50, [1:3]:=40, [4:10]:=10};
    }
endclass

module axi4_stream_fifo_tb;
    parameter int DATA_WIDTH = 32;
    parameter int DEPTH = 16;
    parameter int NUM_TESTS = 10000;
    
    logic clk = 0, rst_n = 0;
    logic [DATA_WIDTH-1:0] s_axis_tdata, m_axis_tdata;
    logic s_axis_tvalid, s_axis_tready, s_axis_tlast;
    logic m_axis_tvalid, m_axis_tready, m_axis_tlast;
    logic [DATA_WIDTH/8-1:0] s_axis_tkeep, m_axis_tkeep;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // DUT instantiation
    axi4_stream_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .FALL_THROUGH(1'b1)
    ) dut (.*);
    
    // Transaction queues
    AXI4StreamTransaction #(DATA_WIDTH) send_queue[$];
    AXI4StreamTransaction #(DATA_WIDTH) recv_queue[$];
    
    // Driver task
    task automatic drive_transaction(AXI4StreamTransaction #(DATA_WIDTH) tr);
        repeat (tr.delay) @(posedge clk);
        
        s_axis_tdata <= tr.data;
        s_axis_tkeep <= tr.keep;
        s_axis_tlast <= tr.last;
        s_axis_tvalid <= 1'b1;
        
        @(posedge clk);
        while (!s_axis_tready) @(posedge clk);
        
        s_axis_tvalid <= 1'b0;
    endtask
    
    // Monitor task
    task automatic monitor_output();
        AXI4StreamTransaction #(DATA_WIDTH) tr;
        
        forever begin
            @(posedge clk);
            if (m_axis_tvalid && m_axis_tready) begin
                tr = new();
                tr.data = m_axis_tdata;
                tr.keep = m_axis_tkeep;
                tr.last = m_axis_tlast;
                recv_queue.push_back(tr);
            end
        end
    endtask
    
    // Checker task
    task automatic check_results();
        AXI4StreamTransaction #(DATA_WIDTH) expected, actual;
        int errors = 0;
        
        while (send_queue.size() > 0) begin
            expected = send_queue.pop_front();
            
            wait (recv_queue.size() > 0);
            actual = recv_queue.pop_front();
            
            if (expected.data !== actual.data ||
                expected.keep !== actual.keep ||
                expected.last !== actual.last) begin
                $error("Mismatch at time %t", $time);
                errors++;
            end
        end
        
        if (errors == 0) begin
            $display("All %0d tests passed!", NUM_TESTS);
        end else begin
            $error("Found %0d errors", errors);
        end
    endtask
    
    // Main test
    initial begin
        AXI4StreamTransaction #(DATA_WIDTH) tr;
        
        // Generate waveforms
        $dumpfile("waves.vcd");
        $dumpvars(0, axi4_stream_fifo_tb);
        
        // Reset
        rst_n = 0;
        s_axis_tvalid = 0;
        m_axis_tready = 0;
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);
        
        // Start monitor
        fork
            monitor_output();
        join_none
        
        // Random testing
        repeat (NUM_TESTS) begin
            tr = new();
            assert(tr.randomize());
            send_queue.push_back(tr);
            
            fork
                drive_transaction(tr);
            join_none
            
            // Randomly toggle receiver ready
            if ($urandom_range(0, 100) < 80) begin
                m_axis_tready <= 1'b1;
            end else begin
                m_axis_tready <= 1'b0;
            end
        end
        
        // Drain FIFO
        m_axis_tready <= 1'b1;
        wait (dut.fifo_empty);
        repeat (10) @(posedge clk);
        
        // Check results
        check_results();
        
        $finish;
    end
    
    // Timeout
    initial begin
        #1ms;
        $error("Test timeout!");
        $finish;
    end
    
    // Coverage
    covergroup cg_fifo_states @(posedge clk);
        cp_fill_level: coverpoint dut.fifo_count {
            bins empty = {0};
            bins low = {[1:DEPTH/4]};
            bins mid = {[DEPTH/4+1:3*DEPTH/4]};
            bins high = {[3*DEPTH/4+1:DEPTH-1]};
            bins full = {DEPTH};
        }
        
        cp_transitions: coverpoint dut.fifo_count {
            bins empty_to_full = (0 => DEPTH);
            bins full_to_empty = (DEPTH => 0);
        }
    endgroup
    
    cg_fifo_states cg_inst = new();

endmodule
```

### ✅ DO: Use Formal Verification for Critical Modules

```systemverilog
// formal/formal_properties.sv
module formal_axi_stream_properties #(
    parameter DATA_WIDTH = 32
) (
    input clk,
    input rst_n,
    input s_axis_tvalid,
    input s_axis_tready,
    input m_axis_tvalid,
    input m_axis_tready,
    input [15:0] fifo_count,
    input fifo_empty,
    input fifo_full
);

    // Assume valid reset behavior
    assume property (@(posedge clk) 
        $fell(rst_n) |=> !s_axis_tvalid);
    
    // If FIFO is full, it cannot accept new data
    assert property (@(posedge clk) disable iff (!rst_n)
        fifo_full |-> !s_axis_tready);
    
    // If FIFO is empty, no data available
    assert property (@(posedge clk) disable iff (!rst_n)
        fifo_empty |-> !m_axis_tvalid);
    
    // Count consistency
    assert property (@(posedge clk) disable iff (!rst_n)
        (fifo_count == 0) <-> fifo_empty);
    
    assert property (@(posedge clk) disable iff (!rst_n)
        (fifo_count == 16) <-> fifo_full);
    
    // No lost transactions
    assert property (@(posedge clk) disable iff (!rst_n)
        (s_axis_tvalid && s_axis_tready) |=> 
        ##[1:$] (m_axis_tvalid && m_axis_tready));

endmodule
```

---

## 4. Mechanical Design Integration with OnShape

Modern hardware projects require tight integration between electrical and mechanical design.

### ✅ DO: Use OnShape's REST API for Automation

```python
# tools/onshape_integration.py
import requests
import json
from datetime import datetime
from pathlib import Path
import hashlib
import hmac
import base64

class OnShapeClient:
    """OnShape API client for automated CAD operations"""
    
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = "https://cad.onshape.com/api/v6"
        
    def _create_auth_headers(self, method: str, path: str, query: str = ""):
        """Generate HMAC-based auth headers"""
        date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        signature_string = (
            f"{method}\n"
            f"{date}\n"
            f"application/json\n"
            f"{path}{query}"
        )
        
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                signature_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            'Date': date,
            'Authorization': f'On {self.access_key}:HmacSHA256:{signature}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def export_step(self, document_id: str, workspace_id: str, element_id: str, 
                    output_path: Path):
        """Export assembly as STEP file"""
        path = f"/documents/d/{document_id}/w/{workspace_id}/e/{element_id}/export"
        
        params = {
            "format": "STEP",
            "version": "AP242",
            "configuration": "default"
        }
        
        headers = self._create_auth_headers("GET", path, 
                                          self._query_string(params))
        
        response = requests.get(
            f"{self.base_url}{path}",
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            print(f"Exported STEP file to {output_path}")
        else:
            raise Exception(f"Export failed: {response.text}")
    
    def get_mass_properties(self, document_id: str, workspace_id: str, 
                           element_id: str) -> dict:
        """Get mass properties for PCB clearance checks"""
        path = f"/documents/d/{document_id}/w/{workspace_id}/e/{element_id}/massproperties"
        
        headers = self._create_auth_headers("GET", path)
        
        response = requests.get(f"{self.base_url}{path}", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get mass properties: {response.text}")
```

### ✅ DO: Implement PCB-Mechanical Collision Detection

```python
# tools/mechanical_validation.py
import numpy as np
from pathlib import Path
import trimesh
import json

class MechanicalValidator:
    """Validate mechanical constraints for PCB assembly"""
    
    def __init__(self, pcb_3d_model: Path, enclosure_model: Path):
        self.pcb = trimesh.load(pcb_3d_model)
        self.enclosure = trimesh.load(enclosure_model)
        self.clearance_mm = 2.0  # Minimum clearance
        
    def check_fit(self) -> dict:
        """Verify PCB fits within enclosure with clearance"""
        # Get bounding boxes
        pcb_bounds = self.pcb.bounds
        enclosure_bounds = self.enclosure.bounds
        
        # Check basic dimensions
        pcb_size = pcb_bounds[1] - pcb_bounds[0]
        enclosure_size = enclosure_bounds[1] - enclosure_bounds[0]
        clearances = enclosure_size - pcb_size
        
        issues = []
        for axis, name in enumerate(['X', 'Y', 'Z']):
            if clearances[axis] < self.clearance_mm * 2:
                issues.append(
                    f"Insufficient clearance in {name}: "
                    f"{clearances[axis]:.1f}mm (need {self.clearance_mm * 2}mm)"
                )
        
        # Check for collisions
        collision = self.pcb.intersection(self.enclosure)
        if collision.is_valid and collision.area > 0:
            issues.append(f"Collision detected! Intersection volume: {collision.volume:.2f}mm³")
        
        return {
            'fits': len(issues) == 0,
            'issues': issues,
            'pcb_dimensions': pcb_size.tolist(),
            'enclosure_internal': enclosure_size.tolist(),
            'clearances': clearances.tolist()
        }
    
    def check_mounting_holes(self, hole_positions: list) -> dict:
        """Validate mounting hole alignment"""
        results = []
        
        for hole in hole_positions:
            # Cast ray from hole position to find enclosure mounting boss
            ray_origin = np.array(hole['position'])
            ray_direction = np.array([0, 0, -1])  # Downward
            
            locations, index_ray, index_tri = self.enclosure.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction]
            )
            
            if len(locations) > 0:
                distance = np.linalg.norm(locations[0] - ray_origin)
                results.append({
                    'hole_id': hole['id'],
                    'aligned': distance < 0.5,  # 0.5mm tolerance
                    'distance': float(distance)
                })
            else:
                results.append({
                    'hole_id': hole['id'],
                    'aligned': False,
                    'error': 'No mounting boss found'
                })
        
        return results
    
    def generate_assembly_report(self, output_path: Path):
        """Create comprehensive assembly validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'clearance_check': self.check_fit(),
            'thermal_analysis': self.analyze_thermal_paths(),
            'vibration_concerns': self.check_vibration_sensitive_components(),
            'assembly_sequence': self.suggest_assembly_order()
        }
        
        output_path.write_text(json.dumps(report, indent=2))
        
        # Generate visual report
        self._create_clearance_visualization(output_path.with_suffix('.html'))
```

---

## 5. Production Test Development

Manufacturing test fixtures are critical for quality and reliability. Modern test development emphasizes automation, coverage, and data collection.

### ✅ DO: Create Comprehensive Test Fixtures

```python
# tests/production/test_fixture.py
import asyncio
import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import pandas as pd

@dataclass
class TestLimit:
    """Define pass/fail criteria for measurements"""
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    
    def check(self, value: float) -> tuple[bool, str]:
        if self.min_value is not None and value < self.min_value:
            return False, f"{value}{self.unit} < {self.min_value}{self.unit}"
        if self.max_value is not None and value > self.max_value:
            return False, f"{value}{self.unit} > {self.max_value}{self.unit}"
        return True, f"{value}{self.unit} OK"

class ProductionTestFixture:
    """Automated production test system"""
    
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.instruments = {}
        self.test_results = []
        self.serial_number = None
        
    async def setup(self):
        """Initialize all test instruments"""
        # Connect to instruments
        self.instruments['dmm'] = await self._connect_dmm()
        self.instruments['scope'] = await self._connect_scope()
        self.instruments['power'] = await self._connect_power_supply()
        self.instruments['jtag'] = await self._connect_jtag()
        
        # Verify calibration dates
        for name, instrument in self.instruments.items():
            cal_date = await instrument.get_calibration_date()
            if (datetime.now() - cal_date).days > 365:
                raise Exception(f"{name} calibration expired!")
    
    async def run_test_sequence(self, serial_number: str) -> Dict:
        """Execute complete test sequence"""
        self.serial_number = serial_number
        self.test_results = []
        
        print(f"\n{'='*60}")
        print(f"Testing Unit: {serial_number}")
        print(f"{'='*60}\n")
        
        # Test groups in order
        test_groups = [
            ("Power Supply", self._test_power_rails),
            ("Clock Generation", self._test_clocks),
            ("Digital I/O", self._test_digital_io),
            ("Analog Circuits", self._test_analog),
            ("Communications", self._test_communications),
            ("FPGA Programming", self._test_fpga),
            ("Functional Tests", self._test_functional),
            ("Thermal Testing", self._test_thermal),
        ]
        
        overall_pass = True
        
        for group_name, test_func in test_groups:
            print(f"\n{group_name}:")
            print("-" * 40)
            
            try:
                group_results = await test_func()
                
                for result in group_results:
                    passed, details = result['limit'].check(result['measured'])
                    result['passed'] = passed
                    result['details'] = details
                    
                    # Display result
                    status = "PASS" if passed else "FAIL"
                    print(f"  {result['name']:<30} {status:>6}  {details}")
                    
                    if not passed:
                        overall_pass = False
                    
                    self.test_results.extend(group_results)
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                overall_pass = False
                self.test_results.append({
                    'name': f"{group_name} Error",
                    'measured': None,
                    'limit': TestLimit(group_name),
                    'passed': False,
                    'details': str(e)
                })
        
        # Generate report
        report = self._generate_report(overall_pass)
        
        # Save to database
        await self._save_to_database(report)
        
        return report
    
    async def _test_power_rails(self) -> List[Dict]:
        """Test all power supply rails"""
        results = []
        
        # Configure for current limit
        await self.instruments['power'].set_output(3.3, current_limit=0.1)
        await self.instruments['power'].enable_output(True)
        await asyncio.sleep(0.1)  # Allow settling
        
        # Test each rail
        rails = [
            ('3V3', 3.3, TestLimit('3V3', 3.2, 3.4, 'V')),
            ('1V8', 1.8, TestLimit('1V8', 1.75, 1.85, 'V')),
            ('1V2_CORE', 1.2, TestLimit('1V2', 1.15, 1.25, 'V')),
            ('VREF', 2.5, TestLimit('VREF', 2.495, 2.505, 'V')),
        ]
        
        for rail_name, nominal, limit in rails:
            # Measure voltage
            voltage = await self.instruments['dmm'].measure_voltage(
                test_point=f"TP_{rail_name}"
            )
            
            results.append({
                'name': f"{rail_name} Voltage",
                'measured': voltage,
                'limit': limit,
                'timestamp': datetime.now()
            })
            
            # Measure ripple
            ripple = await self.instruments['scope'].measure_ripple(
                channel=1,
                coupling='AC',
                bandwidth_limit='20MHz'
            )
            
            results.append({
                'name': f"{rail_name} Ripple",
                'measured': ripple * 1000,  # Convert to mV
                'limit': TestLimit(f'{rail_name}_ripple', max_value=50, unit='mV'),
                'timestamp': datetime.now()
            })
        
        return results
    
    async def _test_clocks(self) -> List[Dict]:
        """Verify all clock signals"""
        results = []
        
        clocks = [
            ('XTAL_25MHz', 25e6, 50),      # 25MHz ±50ppm
            ('CLK_100MHz', 100e6, 100),    # 100MHz ±100ppm  
            ('CLK_125MHz', 125e6, 50),     # 125MHz ±50ppm for Ethernet
        ]
        
        for clock_name, nominal_freq, ppm_tolerance in clocks:
            # Configure scope for frequency measurement
            await self.instruments['scope'].auto_setup(
                channel=1,
                signal_type='clock'
            )
            
            # Measure frequency
            freq = await self.instruments['scope'].measure_frequency()
            
            # Calculate PPM error
            ppm_error = ((freq - nominal_freq) / nominal_freq) * 1e6
            
            results.append({
                'name': f"{clock_name} Frequency",
                'measured': freq / 1e6,  # Convert to MHz
                'limit': TestLimit(
                    clock_name,
                    min_value=nominal_freq * (1 - ppm_tolerance/1e6) / 1e6,
                    max_value=nominal_freq * (1 + ppm_tolerance/1e6) / 1e6,
                    unit='MHz'
                ),
                'timestamp': datetime.now()
            })
            
            # Measure jitter
            jitter_ps = await self.instruments['scope'].measure_jitter(
                measurement_time=1.0  # 1 second measurement
            )
            
            results.append({
                'name': f"{clock_name} Jitter",
                'measured': jitter_ps,
                'limit': TestLimit(f'{clock_name}_jitter', max_value=100, unit='ps'),
                'timestamp': datetime.now()
            })
        
        return results
    
    async def _test_fpga(self) -> List[Dict]:
        """Program and verify FPGA"""
        results = []
        
        # Check JTAG chain
        devices = await self.instruments['jtag'].scan_chain()
        
        expected_id = 0x6BA00477  # Example: Artix-7
        if len(devices) > 0 and devices[0]['idcode'] == expected_id:
            results.append({
                'name': 'FPGA JTAG ID',
                'measured': 1,
                'limit': TestLimit('jtag_detect', min_value=1),
                'timestamp': datetime.now()
            })
        else:
            results.append({
                'name': 'FPGA JTAG ID',
                'measured': 0,
                'limit': TestLimit('jtag_detect', min_value=1),
                'timestamp': datetime.now()
            })
            return results  # Can't continue without JTAG
        
        # Program FPGA
        bitstream_path = self.config['fpga']['test_bitstream']
        success = await self.instruments['jtag'].program_fpga(
            bitstream_path,
            verify=True
        )
        
        results.append({
            'name': 'FPGA Programming',
            'measured': 1 if success else 0,
            'limit': TestLimit('program_success', min_value=1),
            'timestamp': datetime.now()
        })
        
        # Run BIST (Built-In Self Test)
        if success:
            bist_result = await self._run_fpga_bist()
            results.extend(bist_result)
        
        return results
    
    def _generate_report(self, overall_pass: bool) -> Dict:
        """Generate test report with all results"""
        df = pd.DataFrame(self.test_results)
        
        report = {
            'serial_number': self.serial_number,
            'test_date': datetime.now().isoformat(),
            'overall_result': 'PASS' if overall_pass else 'FAIL',
            'test_station': self.config['station']['id'],
            'operator': self.config['station']['operator'],
            'firmware_version': self.config['dut']['firmware_version'],
            'test_duration': sum(r.get('duration', 0) for r in self.test_results),
            'results': self.test_results,
            'statistics': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['passed']]),
                'failed': len([r for r in self.test_results if not r['passed']]),
                'yield_rate': overall_pass
            }
        }
        
        # Generate PDF report
        self._create_pdf_report(report)
        
        # Save CSV for analysis
        df.to_csv(f"test_results_{self.serial_number}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        
        return report
```

### ✅ DO: Implement Boundary Scan Testing

```systemverilog
// fpga/rtl/boundary_scan_test.sv
module boundary_scan_test #(
    parameter int NUM_IO = 64
) (
    input  logic clk,
    input  logic rst_n,
    
    // JTAG Interface
    input  logic tck,
    input  logic tms,
    input  logic tdi,
    output logic tdo,
    
    // Test Control
    input  logic test_enable,
    output logic test_done,
    output logic test_pass,
    
    // I/O under test
    inout  wire [NUM_IO-1:0] io_pins
);

    // Boundary scan chain registers
    logic [NUM_IO-1:0] boundary_scan_out;
    logic [NUM_IO-1:0] boundary_scan_in;
    logic [NUM_IO-1:0] boundary_scan_oe;
    
    // Test pattern generator
    logic [31:0] lfsr;
    logic [5:0] test_phase;
    
    // LFSR for pseudo-random patterns
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= 32'hDEADBEEF;  // Non-zero seed
        end else if (test_enable && !test_done) begin
            // Galois LFSR with taps at 32,22,2,1
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        end
    end
    
    // Test sequencer
    typedef enum logic [2:0] {
        TEST_IDLE,
        TEST_WALKING_1,
        TEST_WALKING_0,
        TEST_RANDOM,
        TEST_INTERCONNECT,
        TEST_COMPLETE
    } test_state_t;
    
    test_state_t test_state;
    logic [6:0] bit_counter;
    logic [NUM_IO-1:0] expected_pattern;
    logic [NUM_IO-1:0] captured_pattern;
    logic pattern_error;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            test_state <= TEST_IDLE;
            test_phase <= 0;
            test_done <= 0;
            test_pass <= 1;
            bit_counter <= 0;
        end else begin
            case (test_state)
                TEST_IDLE: begin
                    if (test_enable) begin
                        test_state <= TEST_WALKING_1;
                        test_done <= 0;
                        test_pass <= 1;
                    end
                end
                
                TEST_WALKING_1: begin
                    // Drive one bit high at a time
                    boundary_scan_out <= 1 << bit_counter;
                    boundary_scan_oe <= '1;
                    
                    if (bit_counter == NUM_IO-1) begin
                        bit_counter <= 0;
                        test_state <= TEST_WALKING_0;
                    end else begin
                        bit_counter <= bit_counter + 1;
                    end
                end
                
                TEST_WALKING_0: begin
                    // Drive one bit low at a time  
                    boundary_scan_out <= ~(1 << bit_counter);
                    boundary_scan_oe <= '1;
                    
                    if (bit_counter == NUM_IO-1) begin
                        bit_counter <= 0;
                        test_state <= TEST_RANDOM;
                    end else begin
                        bit_counter <= bit_counter + 1;
                    end
                end
                
                TEST_RANDOM: begin
                    // Random patterns
                    boundary_scan_out <= lfsr[NUM_IO-1:0];
                    boundary_scan_oe <= '1;
                    
                    if (test_phase == 63) begin
                        test_state <= TEST_INTERCONNECT;
                        test_phase <= 0;
                    end else begin
                        test_phase <= test_phase + 1;
                    end
                end
                
                TEST_INTERCONNECT: begin
                    // Test board-level interconnect
                    // Even pins drive, odd pins receive
                    boundary_scan_oe <= {NUM_IO/2{2'b10}};
                    boundary_scan_out <= {NUM_IO/2{2'b10}};
                    
                    // Check if odd pins see what even pins drive
                    captured_pattern <= boundary_scan_in;
                    expected_pattern <= {NUM_IO/2{2'b10}};
                    
                    if (captured_pattern != expected_pattern) begin
                        test_pass <= 0;
                    end
                    
                    test_state <= TEST_COMPLETE;
                end
                
                TEST_COMPLETE: begin
                    test_done <= 1;
                    boundary_scan_oe <= '0;  // Release all pins
                    if (!test_enable) begin
                        test_state <= TEST_IDLE;
                    end
                end
            endcase
        end
    end
    
    // Bidirectional I/O control
    genvar i;
    generate
        for (i = 0; i < NUM_IO; i++) begin : io_control
            assign io_pins[i] = boundary_scan_oe[i] ? boundary_scan_out[i] : 1'bz;
            assign boundary_scan_in[i] = io_pins[i];
        end
    endgenerate

endmodule
```

---

## 6. EMC and Compliance Testing

Electromagnetic compatibility is critical for commercial products. Design for EMC from the start.

### ✅ DO: Implement Pre-Compliance Testing

```python
# tests/compliance/emc_pre_compliance.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import asyncio
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EMCLimit:
    """Define regulatory limits for emissions"""
    frequency_mhz: List[float]
    limit_dbuv: List[float]
    name: str
    
    def get_limit_at_frequency(self, freq_mhz: float) -> float:
        """Interpolate limit at specific frequency"""
        return np.interp(freq_mhz, self.frequency_mhz, self.limit_dbuv)

class EMCPreCompliance:
    """Pre-compliance EMC testing using spectrum analyzer"""
    
    # FCC Part 15 Class B limits
    FCC_CLASS_B = EMCLimit(
        frequency_mhz=[30, 88, 216, 960, 1000],
        limit_dbuv=[40, 43.5, 46, 54, 54],
        name="FCC Part 15 Class B"
    )
    
    # CISPR 32 Class B limits
    CISPR32_CLASS_B = EMCLimit(
        frequency_mhz=[30, 230, 1000],
        limit_dbuv=[40, 47, 47],
        name="CISPR 32 Class B"
    )
    
    def __init__(self, spectrum_analyzer):
        self.sa = spectrum_analyzer
        self.results = []
        
    async def run_conducted_emissions(self, 
                                    start_freq: float = 150e3,
                                    stop_freq: float = 30e6) -> Dict:
        """Test conducted emissions on power lines"""
        
        # Configure spectrum analyzer
        await self.sa.reset()
        await self.sa.set_frequency_range(start_freq, stop_freq)
        await self.sa.set_rbw(9e3)  # 9kHz for conducted
        await self.sa.set_vbw(30e3)  # VBW = 3x RBW
        await self.sa.set_detector('quasi_peak')
        await self.sa.set_sweep_time(0)  # Auto
        
        # Perform measurement
        print("Measuring conducted emissions...")
        await self.sa.single_sweep()
        
        frequencies = await self.sa.get_frequency_points()
        amplitudes = await self.sa.get_trace_data()
        
        # Find peaks above limit
        violations = []
        limit = self.FCC_CLASS_B
        
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            freq_mhz = freq / 1e6
            limit_value = limit.get_limit_at_frequency(freq_mhz)
            
            if amp > limit_value:
                margin = amp - limit_value
                violations.append({
                    'frequency': freq,
                    'amplitude': amp,
                    'limit': limit_value,
                    'margin': margin,
                    'type': 'Conducted'
                })
        
        return {
            'pass': len(violations) == 0,
            'violations': violations,
            'worst_margin': max([v['margin'] for v in violations]) if violations else 0,
            'trace': {
                'frequency': frequencies,
                'amplitude': amplitudes
            }
        }
    
    async def run_radiated_emissions(self,
                                   distance_m: float = 3.0,
                                   antenna: str = 'bilog') -> Dict:
        """Test radiated emissions in semi-anechoic chamber"""
        
        # Configure for radiated measurements
        await self.sa.reset()
        await self.sa.set_frequency_range(30e6, 1e9)
        await self.sa.set_rbw(120e3)  # 120kHz for radiated
        await self.sa.set_detector('peak')  # Peak first, then QP on violations
        
        results = {
            'horizontal': {},
            'vertical': {}
        }
        
        # Test both polarizations
        for polarization in ['horizontal', 'vertical']:
            print(f"\nTesting {polarization} polarization...")
            
            # Rotate turntable through 360 degrees
            max_emissions = []
            
            for angle in range(0, 360, 15):
                await self._rotate_turntable(angle)
                await asyncio.sleep(2)  # Settle time
                
                await self.sa.single_sweep()
                trace = await self.sa.get_trace_data()
                max_emissions.append(np.max(trace))
            
            # Find worst-case angle
            worst_angle = np.argmax(max_emissions) * 15
            await self._rotate_turntable(worst_angle)
            
            # Detailed measurement at worst angle
            await self.sa.set_detector('quasi_peak')
            await self.sa.single_sweep()
            
            frequencies = await self.sa.get_frequency_points()
            amplitudes = await self.sa.get_trace_data()
            
            # Apply antenna factor and cable loss
            af = self._get_antenna_factor(antenna, frequencies)
            cable_loss = 2.0  # dB typical
            
            field_strength = amplitudes + af + cable_loss - 107  # Convert to dBuV/m
            
            # Check against limits
            violations = []
            limit = self.CISPR32_CLASS_B
            
            for freq, fs in zip(frequencies, field_strength):
                freq_mhz = freq / 1e6
                limit_value = limit.get_limit_at_frequency(freq_mhz)
                
                if fs > limit_value:
                    violations.append({
                        'frequency': freq,
                        'field_strength': fs,
                        'limit': limit_value,
                        'margin': fs - limit_value,
                        'angle': worst_angle,
                        'polarization': polarization
                    })
            
            results[polarization] = {
                'violations': violations,
                'worst_angle': worst_angle,
                'trace': {
                    'frequency': frequencies,
                    'field_strength': field_strength
                }
            }
        
        return results
    
    def generate_report(self, output_path: str):
        """Create EMC pre-compliance report with plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot conducted emissions
        ax = axes[0, 0]
        # ... plotting code
        
        # Plot radiated emissions  
        ax = axes[0, 1]
        # ... plotting code
        
        plt.tight_layout()
        plt.savefig(output_path)
        
    async def suggest_fixes(self, violations: List[Dict]) -> List[Dict]:
        """Suggest EMC fixes based on violation frequencies"""
        suggestions = []
        
        for violation in violations:
            freq = violation['frequency']
            
            if freq < 1e6:  # Low frequency
                suggestions.append({
                    'issue': f"Conducted emission at {freq/1e3:.1f}kHz",
                    'suggestions': [
                        "Add common-mode choke on power input",
                        "Increase Y-capacitor value",
                        "Add ferrite bead on DC lines"
                    ]
                })
            elif freq < 100e6:  # VHF
                suggestions.append({
                    'issue': f"Radiated emission at {freq/1e6:.1f}MHz",
                    'suggestions': [
                        "Check for resonances in power planes",
                        "Add more bypass capacitors",
                        "Improve ground plane stitching"
                    ]
                })
            else:  # UHF and up
                suggestions.append({
                    'issue': f"High-frequency emission at {freq/1e6:.0f}MHz",
                    'suggestions': [
                        "Shield high-speed signals",
                        "Add guard vias around clock traces",
                        "Use spread-spectrum clocking"
                    ]
                })
        
        return suggestions
```

---

## 7. Supply Chain and Manufacturing

Managing the transition from prototype to production requires careful attention to component sourcing and manufacturing processes.

### ✅ DO: Implement Smart BOM Management

```python
# tools/bom_manager.py
import pandas as pd
import requests
from typing import List, Dict, Optional
import asyncio
import aiohttp
from datetime import datetime, timedelta

class BOMManager:
    """Intelligent BOM management with availability checking"""
    
    def __init__(self, octopart_api_key: str):
        self.api_key = octopart_api_key
        self.preferred_distributors = ['Digi-Key', 'Mouser', 'Arrow']
        
    async def analyze_bom(self, bom_file: str) -> Dict:
        """Complete BOM analysis including availability and alternates"""
        
        # Load BOM
        df = pd.read_csv(bom_file)
        
        # Standardize columns
        df.columns = [col.lower().strip() for col in df.columns]
        
        results = {
            'total_components': len(df),
            'total_line_items': df['quantity'].sum(),
            'availability_issues': [],
            'cost_analysis': {},
            'alternates_found': {},
            'obsolescence_risks': []
        }
        
        # Check each component
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _, row in df.iterrows():
                task = self._check_component(session, row)
                tasks.append(task)
            
            component_results = await asyncio.gather(*tasks)
        
        # Analyze results
        total_cost = 0
        for comp_result in component_results:
            if comp_result['availability'] < comp_result['required_qty']:
                results['availability_issues'].append(comp_result)
            
            if comp_result['lifecycle_status'] in ['Obsolete', 'NRND']:
                results['obsolescence_risks'].append(comp_result)
            
            total_cost += comp_result['extended_price']
            
            if comp_result['alternates']:
                results['alternates_found'][comp_result['mpn']] = comp_result['alternates']
        
        results['cost_analysis'] = {
            'unit_cost': total_cost,
            'cost_per_1k': total_cost * 0.85,  # Assume 15% discount at 1k
            'cost_per_10k': total_cost * 0.70  # Assume 30% discount at 10k
        }
        
        return results
    
    async def _check_component(self, session: aiohttp.ClientSession, 
                              component: pd.Series) -> Dict:
        """Check single component availability and pricing"""
        
        mpn = component.get('mpn', '')
        manufacturer = component.get('manufacturer', '')
        quantity = int(component.get('quantity', 1))
        
        # Query Octopart API
        url = "https://octopart.com/api/v4/search"
        params = {
            'apikey': self.api_key,
            'q': mpn,
            'limit': 10
        }
        
        async with session.get(url, params=params) as response:
            data = await response.json()
        
        # Parse results
        best_result = None
        alternates = []
        
        for result in data.get('results', []):
            part = result['part']
            
            # Check manufacturer match
            if part['manufacturer']['name'].lower() == manufacturer.lower():
                best_result = part
            else:
                # Potential alternate
                alternates.append({
                    'mpn': part['mpn'],
                    'manufacturer': part['manufacturer']['name'],
                    'description': part['short_description']
                })
        
        if not best_result:
            return {
                'mpn': mpn,
                'status': 'Not Found',
                'availability': 0,
                'required_qty': quantity,
                'unit_price': 0,
                'extended_price': 0,
                'lifecycle_status': 'Unknown',
                'alternates': alternates
            }
        
        # Get availability and pricing
        total_stock = 0
        best_price = float('inf')
        lead_time = 0
        
        for offer in best_result.get('offers', []):
            if offer['seller']['name'] in self.preferred_distributors:
                total_stock += offer.get('in_stock_quantity', 0)
                
                # Find price break
                for price_break in offer.get('prices', {}).get('USD', []):
                    if price_break[0] <= quantity:
                        price = float(price_break[1])
                        if price < best_price:
                            best_price = price
                
                # Track lead time
                if offer.get('factory_lead_days'):
                    lead_time = max(lead_time, offer['factory_lead_days'])
        
        return {
            'mpn': mpn,
            'manufacturer': manufacturer,
            'status': 'OK',
            'availability': total_stock,
            'required_qty': quantity,
            'unit_price': best_price,
            'extended_price': best_price * quantity,
            'lead_time_days': lead_time,
            'lifecycle_status': best_result.get('lifecycle_status', 'Active'),
            'alternates': alternates[:3]  # Top 3 alternates
        }
    
    def generate_procurement_report(self, analysis: Dict, output_file: str):
        """Generate detailed procurement report"""
        
        with open(output_file, 'w') as f:
            f.write("# BOM Procurement Analysis Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n")
            f.write(f"- Total Components: {analysis['total_components']}\n")
            f.write(f"- Total Quantity: {analysis['total_line_items']}\n")
            f.write(f"- Unit Cost: ${analysis['cost_analysis']['unit_cost']:.2f}\n")
            f.write(f"- Availability Issues: {len(analysis['availability_issues'])}\n")
            f.write(f"- Obsolescence Risks: {len(analysis['obsolescence_risks'])}\n\n")
            
            # Critical Issues
            if analysis['availability_issues']:
                f.write("## ⚠️ Availability Issues\n")
                for issue in analysis['availability_issues']:
                    f.write(f"- **{issue['mpn']}**: Need {issue['required_qty']}, "
                           f"available {issue['availability']}\n")
                    if issue['alternates']:
                        f.write("  Alternates:\n")
                        for alt in issue['alternates']:
                            f.write(f"  - {alt['mpn']} ({alt['manufacturer']})\n")
                f.write("\n")
            
            # Obsolescence warnings
            if analysis['obsolescence_risks']:
                f.write("## ⚠️ Obsolescence Risks\n")
                for risk in analysis['obsolescence_risks']:
                    f.write(f"- **{risk['mpn']}**: {risk['lifecycle_status']}\n")
                f.write("\n")
            
            # Cost breakdown
            f.write("## Cost Analysis\n")
            f.write(f"| Quantity | Unit Cost | Total |\n")
            f.write(f"|----------|-----------|-------|\n")
            f.write(f"| 1 | ${analysis['cost_analysis']['unit_cost']:.2f} | "
                   f"${analysis['cost_analysis']['unit_cost']:.2f} |\n")
            f.write(f"| 1,000 | ${analysis['cost_analysis']['cost_per_1k']:.2f} | "
                   f"${analysis['cost_analysis']['cost_per_1k'] * 1000:.2f} |\n")
            f.write(f"| 10,000 | ${analysis['cost_analysis']['cost_per_10k']:.2f} | "
                   f"${analysis['cost_analysis']['cost_per_10k'] * 10000:.2f} |\n")
```

### ✅ DO: Automate PCB Ordering

```python
# tools/pcb_ordering.py
import json
import zipfile
from pathlib import Path
import requests
from typing import Dict, List

class PCBOrderAutomation:
    """Automate PCB ordering with various fabs"""
    
    def __init__(self):
        self.fabricators = {
            'pcbway': PCBWayAPI(),
            'jlcpcb': JLCPCBAPI(),
            'oshpark': OSHParkAPI()
        }
        
    def prepare_fabrication_package(self, 
                                   project_dir: Path,
                                   fab_name: str) -> Path:
        """Prepare fab-specific package"""
        
        fab = self.fabricators.get(fab_name)
        if not fab:
            raise ValueError(f"Unknown fabricator: {fab_name}")
        
        # Create output directory
        output_dir = project_dir / "outputs" / f"{fab_name}_package"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename files per fab requirements
        gerber_map = fab.get_gerber_mapping()
        
        gerber_dir = project_dir / "outputs" / "gerbers"
        for std_name, fab_name in gerber_map.items():
            src = gerber_dir / f"{project_dir.name}.{std_name}"
            if src.exists():
                dst = output_dir / fab_name
                dst.write_bytes(src.read_bytes())
        
        # Add fab-specific files
        if fab_name == 'jlcpcb':
            # Generate pick-and-place in JLCPCB format
            self._generate_jlc_pnp(project_dir, output_dir)
            # Generate BOM in JLCPCB format
            self._generate_jlc_bom(project_dir, output_dir)
        
        # Create ZIP
        zip_path = output_dir.parent / f"{fab_name}_{datetime.now():%Y%m%d}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in output_dir.iterdir():
                zf.write(file, file.name)
        
        return zip_path
    
    async def get_instant_quote(self, 
                               fab_name: str,
                               pcb_specs: Dict) -> Dict:
        """Get instant quote from fabricator"""
        
        fab = self.fabricators.get(fab_name)
        if not fab:
            raise ValueError(f"Unknown fabricator: {fab_name}")
        
        return await fab.get_quote(pcb_specs)
    
    async def place_order(self,
                         fab_name: str,
                         package_path: Path,
                         specifications: Dict,
                         shipping_info: Dict) -> str:
        """Place order with fabricator"""
        
        fab = self.fabricators.get(fab_name)
        
        # Upload package
        upload_result = await fab.upload_gerbers(package_path)
        
        # Configure specifications
        order_config = {
            'board_id': upload_result['board_id'],
            'quantity': specifications['quantity'],
            'layers': specifications['layers'],
            'thickness': specifications.get('thickness', 1.6),
            'surface_finish': specifications.get('finish', 'HASL'),
            'solder_mask': specifications.get('mask_color', 'Green'),
            'silkscreen': specifications.get('silk_color', 'White'),
            'copper_weight': specifications.get('copper_weight', '1oz'),
            'via_process': specifications.get('via_process', 'Tenting'),
            'shipping': shipping_info
        }
        
        # Place order
        order_result = await fab.place_order(order_config)
        
        return order_result['order_number']


class PCBWayAPI:
    """PCBWay specific API implementation"""
    
    def __init__(self):
        self.api_key = os.getenv('PCBWAY_API_KEY')
        self.base_url = "https://www.pcbway.com/api/v1"
        
    def get_gerber_mapping(self) -> Dict[str, str]:
        """PCBWay expects specific file names"""
        return {
            'F.Cu.gbr': 'TopLayer.GTL',
            'B.Cu.gbr': 'BottomLayer.GBL',
            'F.Mask.gbr': 'TopSolderMask.GTS',
            'B.Mask.gbr': 'BottomSolderMask.GBS',
            'F.SilkS.gbr': 'TopSilkscreen.GTO',
            'B.SilkS.gbr': 'BottomSilkscreen.GBO',
            'Edge.Cuts.gbr': 'BoardOutline.GKO',
            'PTH.drl': 'Drill.TXT',
            'NPTH.drl': 'DrillNPTH.TXT'
        }
    
    async def get_quote(self, specs: Dict) -> Dict:
        """Get instant quote from PCBWay"""
        
        params = {
            'api_key': self.api_key,
            'length': specs['size_x'],
            'width': specs['size_y'],
            'layers': specs['layers'],
            'quantity': specs['quantity'],
            'thickness': specs.get('thickness', 1.6),
            'finish': specs.get('finish', 'HASL'),
            'soldermask': specs.get('mask_color', 'Green'),
            'silkscreen': specs.get('silk_color', 'White')
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/quote", params=params) as resp:
                data = await resp.json()
                
        return {
            'fabricator': 'PCBWay',
            'unit_price': data['unit_price'],
            'setup_fee': data['setup_fee'],
            'shipping': data['shipping_options'],
            'lead_time': data['lead_time_days'],
            'total': data['total_price']
        }
```

---

## 8. Version Control and Collaboration

Hardware projects require special considerations for version control due to binary files and multi-tool workflows.

### ✅ DO: Implement Proper Git Workflows

```bash
# .gitmessage
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type: feat, fix, docs, style, refactor, test, chore
# Scope: pcb, fpga, firmware, mechanical, docs
# Subject: imperative mood, max 50 chars
# Body: explain what and why, not how
# Footer: reference issues, breaking changes

# Example:
# feat(pcb): add USB-C power delivery support
#
# - Added TPS65987 PD controller
# - Supports up to 100W (20V/5A)
# - Updated power supply section for higher current
#
# Closes #123
```

### ✅ DO: Use CI/CD for Hardware

```yaml
# .github/workflows/hardware-ci.yml
name: Hardware CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  electrical-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true
          
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'
          
      - name: Install KiCad CLI
        run: |
          sudo add-apt-repository ppa:kicad/kicad-9.0-releases
          sudo apt update
          sudo apt install -y kicad kicad-cli
          
      - name: Run ERC
        run: |
          kicad-cli sch erc hardware/pcb/project.kicad_sch
          
      - name: Run DRC  
        run: |
          kicad-cli pcb drc hardware/pcb/project.kicad_pcb
          
      - name: Generate Production Files
        run: |
          python tools/generate_production_files.py
          
      - name: Check BOM
        run: |
          python tools/bom_manager.py analyze hardware/pcb/bom.csv
          
      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: production-files
          path: hardware/pcb/outputs/
          
  fpga-synthesis:
    runs-on: ubuntu-latest
    container:
      image: hdlc/vivado:2025.1  # Custom Docker image with Vivado
    steps:
      - uses: actions/checkout@v3
        
      - name: Run Synthesis
        run: |
          cd fpga
          vivado -mode batch -source scripts/synthesize.tcl
          
      - name: Check Timing
        run: |
          python scripts/check_timing.py reports/timing_summary.rpt
          
      - name: Resource Utilization
        run: |
          python scripts/check_utilization.py reports/utilization.rpt
          
  mechanical-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true
          
      - name: Setup Tools
        run: |
          pip install trimesh numpy
          
      - name: Validate Fit
        run: |
          python tools/mechanical_validation.py \
            --pcb hardware/pcb/outputs/3d/board.step \
            --enclosure hardware/mechanical/enclosure/case.step
            
      - name: Generate Assembly Drawings
        run: |
          python tools/generate_assembly_drawings.py
```

---

## 9. Advanced Debugging Techniques

Professional hardware debugging requires systematic approaches and proper tooling.

### ✅ DO: Implement In-System Debugging

```systemverilog
// fpga/rtl/debug/integrated_logic_analyzer.sv
module integrated_logic_analyzer #(
    parameter int DEPTH = 4096,
    parameter int WIDTH = 128,
    parameter int TRIGGER_STAGES = 4
) (
    input  logic clk,
    input  logic rst_n,
    
    // Signals to monitor
    input  logic [WIDTH-1:0] probe_signals,
    
    // Control interface (connected to UART/USB)
    input  logic        cmd_valid,
    input  logic [7:0]  cmd_data,
    output logic        cmd_ready,
    
    output logic        data_valid,
    output logic [7:0]  data_out,
    input  logic        data_ready,
    
    // Status
    output logic        armed,
    output logic        triggered,
    output logic        complete
);

    // Command definitions
    typedef enum logic [7:0] {
        CMD_ARM = 8'h01,
        CMD_DISARM = 8'h02,
        CMD_SET_TRIGGER = 8'h10,
        CMD_SET_MASK = 8'h11,
        CMD_SET_VALUE = 8'h12,
        CMD_SET_EDGE = 8'h13,
        CMD_READ_DATA = 8'h20,
        CMD_STATUS = 8'h30
    } command_t;
    
    // Trigger configuration
    logic [WIDTH-1:0] trigger_mask [TRIGGER_STAGES];
    logic [WIDTH-1:0] trigger_value [TRIGGER_STAGES];
    logic [WIDTH-1:0] trigger_edge [TRIGGER_STAGES];
    logic [TRIGGER_STAGES-1:0] trigger_match;
    
    // Sample buffer
    logic [WIDTH-1:0] sample_buffer [DEPTH];
    logic [$clog2(DEPTH)-1:0] write_addr, read_addr;
    logic [$clog2(DEPTH)-1:0] trigger_addr;
    logic [$clog2(DEPTH)-1:0] post_trigger_samples;
    
    // State machine
    typedef enum logic [2:0] {
        IDLE,
        ARMED,
        TRIGGERED,
        POST_TRIGGER,
        COMPLETE,
        READOUT
    } state_t;
    
    state_t state, next_state;
    
    // Edge detection
    logic [WIDTH-1:0] probe_signals_d;
    logic [WIDTH-1:0] probe_edges;
    
    always_ff @(posedge clk) begin
        probe_signals_d <= probe_signals;
        probe_edges <= probe_signals ^ probe_signals_d;
    end
    
    // Trigger detection
    genvar i;
    generate
        for (i = 0; i < TRIGGER_STAGES; i++) begin : trigger_stage
            always_comb begin
                trigger_match[i] = 
                    ((probe_signals & trigger_mask[i]) == 
                     (trigger_value[i] & trigger_mask[i])) &&
                    ((probe_edges & trigger_edge[i]) == trigger_edge[i]);
            end
        end
    endgenerate
    
    // Sequential trigger evaluation
    logic trigger_fire;
    always_comb begin
        trigger_fire = &trigger_match;  // All stages must match
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (cmd_valid && cmd_data == CMD_ARM) begin
                    next_state = ARMED;
                end
            end
            
            ARMED: begin
                if (trigger_fire) begin
                    next_state = TRIGGERED;
                end else if (cmd_valid && cmd_data == CMD_DISARM) begin
                    next_state = IDLE;
                end
            end
            
            TRIGGERED: begin
                next_state = POST_TRIGGER;
            end
            
            POST_TRIGGER: begin
                if (write_addr == trigger_addr + post_trigger_samples) begin
                    next_state = COMPLETE;
                end
            end
            
            COMPLETE: begin
                if (cmd_valid && cmd_data == CMD_READ_DATA) begin
                    next_state = READOUT;
                end else if (cmd_valid && cmd_data == CMD_DISARM) begin
                    next_state = IDLE;
                end
            end
            
            READOUT: begin
                if (read_addr == write_addr) begin
                    next_state = COMPLETE;
                end
            end
        endcase
    end
    
    // Sample capture
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_addr <= '0;
            trigger_addr <= '0;
        end else begin
            case (state)
                ARMED, POST_TRIGGER: begin
                    sample_buffer[write_addr] <= probe_signals;
                    write_addr <= write_addr + 1;
                end
                
                TRIGGERED: begin
                    trigger_addr <= write_addr;
                end
                
                IDLE: begin
                    write_addr <= '0;
                end
            endcase
        end
    end
    
    // Command processor
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset trigger configuration
            for (int i = 0; i < TRIGGER_STAGES; i++) begin
                trigger_mask[i] <= '1;  // All bits active
                trigger_value[i] <= '0;
                trigger_edge[i] <= '0;  // Level trigger
            end
            post_trigger_samples <= DEPTH / 2;  // 50% post-trigger
        end else if (cmd_valid && cmd_ready) begin
            // Process commands
            // ... (command handling logic)
        end
    end
    
    // Data readout
    logic [3:0] byte_counter;
    logic [WIDTH-1:0] current_sample;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_addr <= '0;
            byte_counter <= '0;
            data_valid <= '0;
        end else begin
            if (state == READOUT && data_ready) begin
                if (byte_counter == 0) begin
                    current_sample <= sample_buffer[read_addr];
                end
                
                // Output bytes MSB first
                data_out <= current_sample[WIDTH-1-byte_counter*8 -: 8];
                data_valid <= 1'b1;
                
                if (byte_counter == (WIDTH/8 - 1)) begin
                    byte_counter <= '0;
                    read_addr <= read_addr + 1;
                end else begin
                    byte_counter <= byte_counter + 1;
                end
            end else begin
                data_valid <= '0;
            end
        end
    end
    
    // Status outputs
    assign armed = (state == ARMED);
    assign triggered = (state != IDLE && state != ARMED);
    assign complete = (state == COMPLETE);
    assign cmd_ready = (state != READOUT);  // Can accept commands except during readout

endmodule
```

### ✅ DO: Create Hardware-in-the-Loop Testing

```python
# tests/hil/hardware_in_loop_test.py
import asyncio
import numpy as np
from typing import List, Dict, Callable
import serial
import struct

class HILTestFramework:
    """Hardware-in-the-Loop test framework"""
    
    def __init__(self, dut_port: str, instrument_config: Dict):
        self.dut = serial.Serial(dut_port, 115200, timeout=1)
        self.instruments = self._setup_instruments(instrument_config)
        self.test_results = []
        
    async def run_test_sequence(self, test_cases: List[Dict]) -> Dict:
        """Execute HIL test sequence"""
        
        print("Starting Hardware-in-the-Loop Testing")
        print("=" * 60)
        
        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            print("-" * 40)
            
            # Setup test conditions
            await self._setup_test_environment(test_case['setup'])
            
            # Apply stimulus
            await self._apply_stimulus(test_case['stimulus'])
            
            # Measure response
            response = await self._measure_response(test_case['measurements'])
            
            # Validate results
            passed = self._validate_response(response, test_case['expected'])
            
            # Store results
            self.test_results.append({
                'name': test_case['name'],
                'passed': passed,
                'response': response,
                'expected': test_case['expected']
            })
            
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            
        return self._generate_report()
    
    async def test_power_sequencing(self) -> bool:
        """Test power supply sequencing"""
        
        # Monitor all power rails
        rail_monitors = {
            '3V3': self.instruments['scope'].channel(1),
            '1V8': self.instruments['scope'].channel(2),
            '1V2': self.instruments['scope'].channel(3),
            'RESET': self.instruments['scope'].channel(4)
        }
        
        # Configure scope for single sequence capture
        await self.instruments['scope'].set_timebase(1e-3)  # 1ms/div
        await self.instruments['scope'].set_trigger(
            source='EXT',
            level=1.5,
            mode='NORMAL'
        )
        
        # Arm scope
        await self.instruments['scope'].single()
        
        # Apply power
        await self.instruments['power'].enable_output(True)
        
        # Wait for acquisition
        await asyncio.sleep(0.5)
        
        # Get waveforms
        waveforms = {}
        for rail, channel in rail_monitors.items():
            waveforms[rail] = await channel.get_waveform()
        
        # Analyze sequencing
        sequence_ok = True
        
        # Find rise times
        rise_times = {}
        for rail, waveform in waveforms.items():
            threshold = 0.9 * waveform['data'].max()
            rise_idx = np.where(waveform['data'] > threshold)[0][0]
            rise_times[rail] = waveform['time'][rise_idx]
        
        # Check sequence order: 3V3 -> 1V8 -> 1V2
        if not (rise_times['3V3'] < rise_times['1V8'] < rise_times['1V2']):
            print("ERROR: Power sequence order incorrect")
            sequence_ok = False
        
        # Check timing constraints
        if rise_times['1V8'] - rise_times['3V3'] < 100e-6:  # Min 100us
            print("ERROR: 3V3 to 1V8 delay too short")
            sequence_ok = False
            
        # Check reset releases after all rails stable
        reset_release = rise_times['RESET']
        if reset_release < max(rise_times.values()) + 10e-3:  # 10ms margin
            print("ERROR: Reset released too early")
            sequence_ok = False
        
        return sequence_ok
    
    async def test_high_speed_interfaces(self) -> Dict:
        """Test high-speed serial interfaces"""
        
        interfaces = {
            'USB3': self._test_usb3_compliance,
            'PCIe': self._test_pcie_compliance,
            'RGMII': self._test_rgmii_timing,
            'DDR3': self._test_ddr3_timing
        }
        
        results = {}
        
        for interface, test_func in interfaces.items():
            print(f"\nTesting {interface}...")
            results[interface] = await test_func()
        
        return results
    
    async def _test_usb3_compliance(self) -> Dict:
        """USB 3.0 compliance testing"""
        
        # Configure for USB3 eye diagram
        await self.instruments['scope'].set_channel(
            channel=1,
            coupling='DC',
            bandwidth='2GHz',
            attenuation=10
        )
        
        # Enable test pattern from DUT
        self._send_command('USB3_TEST_PATTERN', 'CP0')
        
        # Capture eye diagram
        await self.instruments['scope'].set_eye_diagram_mode(
            data_rate=5e9,  # 5 Gbps
            trigger_pattern='01010101'
        )
        
        # Accumulate waveforms
        await asyncio.sleep(2.0)
        
        # Measure eye parameters
        eye_height = await self.instruments['scope'].measure('EYE_HEIGHT')
        eye_width = await self.instruments['scope'].measure('EYE_WIDTH')
        jitter_rms = await self.instruments['scope'].measure('TIE_RMS')
        
        # Check against USB3 specifications
        passed = (
            eye_height > 100e-3 and  # 100mV minimum
            eye_width > 0.65 and      # 65% UI minimum
            jitter_rms < 3.0e-12      # 3ps RMS max
        )
        
        return {
            'passed': passed,
            'eye_height_mv': eye_height * 1000,
            'eye_width_ui': eye_width,
            'jitter_ps': jitter_rms * 1e12
        }
    
    def _send_command(self, command: str, *args):
        """Send command to DUT via serial"""
        
        cmd_packet = {
            'USB3_TEST_PATTERN': 0x10,
            'DDR3_TRAINING': 0x20,
            'BIST_START': 0x30,
            'REG_WRITE': 0x40,
            'REG_READ': 0x41
        }
        
        packet = bytearray()
        packet.append(cmd_packet[command])
        packet.append(len(args))
        
        for arg in args:
            if isinstance(arg, int):
                packet.extend(struct.pack('<I', arg))
            elif isinstance(arg, str):
                packet.extend(arg.encode())
                packet.append(0)  # Null terminator
        
        # Add checksum
        checksum = sum(packet) & 0xFF
        packet.append(checksum)
        
        self.dut.write(packet)
        
        # Wait for acknowledgment
        ack = self.dut.read(1)
        if ack != b'\x06':  # ACK
            raise Exception(f"Command {command} not acknowledged")
```

---

## 10. Documentation and Knowledge Management

Comprehensive documentation is critical for hardware projects due to their complexity and long lifecycles.

### ✅ DO: Maintain Living Documentation

```python
# tools/documentation_generator.py
import json
import subprocess
from pathlib import Path
from datetime import datetime
import markdown
import pdfkit
from jinja2 import Environment, FileSystemLoader

class HardwareDocumentationGenerator:
    """Generate comprehensive hardware documentation"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.template_env = Environment(
            loader=FileSystemLoader(project_root / 'docs' / 'templates')
        )
        
    def generate_all_documentation(self):
        """Generate complete documentation package"""
        
        docs = [
            self.generate_design_guide(),
            self.generate_assembly_manual(),
            self.generate_test_procedures(),
            self.generate_datasheet(),
            self.generate_user_manual(),
            self.generate_service_manual()
        ]
        
        # Combine into master PDF
        self._combine_pdfs(docs, self.project_root / 'docs' / 'complete_documentation.pdf')
        
    def generate_datasheet(self) -> Path:
        """Generate product datasheet"""
        
        # Collect specifications
        specs = self._extract_specifications()
        
        # Load template
        template = self.template_env.get_template('datasheet.md.j2')
        
        # Render
        content = template.render(
            product_name=specs['product_name'],
            version=specs['version'],
            features=specs['features'],
            specifications=specs['specifications'],
            pinout=self._generate_pinout_table(),
            block_diagram=self._generate_block_diagram(),
            typical_application=self._generate_app_circuit(),
            absolute_maximum=specs['absolute_maximum'],
            operating_conditions=specs['operating_conditions'],
            electrical_characteristics=specs['electrical'],
            timing_diagrams=self._generate_timing_diagrams(),
            package_info=specs['package'],
            ordering_info=specs['ordering']
        )
        
        # Convert to PDF
        output_path = self.project_root / 'docs' / 'datasheet.pdf'
        self._markdown_to_pdf(content, output_path)
        
        return output_path
    
    def generate_assembly_manual(self) -> Path:
        """Generate assembly instructions"""
        
        template = self.template_env.get_template('assembly_manual.md.j2')
        
        # Extract assembly data
        bom = self._load_bom()
        assembly_steps = self._generate_assembly_sequence(bom)
        
        content = template.render(
            bom=bom,
            tools_required=[
                "Soldering iron (350°C)",
                "Flux pen",
                "Tweezers (ESD-safe)",
                "Microscope or magnifier",
                "Hot air station (optional)",
                "Multimeter"
            ],
            assembly_steps=assembly_steps,
            test_points=self._get_test_points(),
            troubleshooting=self._generate_troubleshooting_guide()
        )
        
        output_path = self.project_root / 'docs' / 'assembly_manual.pdf'
        self._markdown_to_pdf(content, output_path)
        
        return output_path
    
    def _generate_assembly_sequence(self, bom: List[Dict]) -> List[Dict]:
        """Generate optimal assembly order"""
        
        # Sort by component height and thermal mass
        components = []
        
        for item in bom:
            height = self._get_component_height(item['footprint'])
            thermal_mass = self._estimate_thermal_mass(item)
            
            components.append({
                'refdes': item['references'],
                'value': item['value'],
                'footprint': item['footprint'],
                'height': height,
                'thermal_mass': thermal_mass,
                'special_notes': self._get_assembly_notes(item)
            })
        
        # Sort: lowest to highest, light to heavy
        components.sort(key=lambda x: (x['height'], x['thermal_mass']))
        
        # Group into assembly stages
        stages = [
            {
                'name': 'Passive Components',
                'components': [c for c in components if c['footprint'].startswith(('R', 'C', 'L'))],
                'temperature': '350°C',
                'notes': 'Start with smallest components'
            },
            {
                'name': 'ICs and Semiconductors',
                'components': [c for c in components if c['footprint'].startswith(('SOT', 'SO', 'QFN', 'TQFP'))],
                'temperature': '340°C',
                'notes': 'Use flux for fine-pitch components'
            },
            {
                'name': 'Connectors and Mechanical',
                'components': [c for c in components if c['footprint'].startswith(('USB', 'CONN', 'SW'))],
                'temperature': '360°C',
                'notes': 'Ensure proper alignment before soldering'
            }
        ]
        
        return stages
    
    def _generate_timing_diagrams(self) -> List[str]:
        """Generate timing diagrams from simulation"""
        
        diagrams = []
        
        # Run Verilog simulation to capture timing
        sim_result = subprocess.run([
            'iverilog',
            '-o', 'timing_sim',
            'fpga/testbench/timing_extraction_tb.sv',
            'fpga/rtl/*.sv'
        ], capture_output=True)
        
        if sim_result.returncode == 0:
            # Run simulation
            subprocess.run(['vvp', 'timing_sim'])
            
            # Convert VCD to timing diagram
            # Using wavedrom format
            with open('timing.json', 'r') as f:
                timing_data = json.load(f)
            
            for signal_group in timing_data:
                svg = self._wavedrom_to_svg(signal_group)
                diagrams.append(svg)
        
        return diagrams
```

### ✅ DO: Create Interactive Documentation

```html
<!-- docs/interactive/board_viewer.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Interactive PCB Viewer</title>
    <script src="https://unpkg.com/three@0.150.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.150.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://unpkg.com/three@0.150.0/examples/js/loaders/STLLoader.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #viewer { width: 100%; height: 600px; }
        #info { position: absolute; top: 10px; left: 10px; background: white; padding: 10px; }
        .component { cursor: pointer; }
        .component:hover { color: blue; }
        #details { position: absolute; top: 10px; right: 10px; background: white; 
                   padding: 10px; width: 300px; display: none; }
    </style>
</head>
<body>
    <div id="viewer"></div>
    <div id="info">
        <h3>Interactive PCB Viewer</h3>
        <p>Click on components for details</p>
        <p>Mouse: Rotate | Scroll: Zoom | Right-click: Pan</p>
    </div>
    <div id="details">
        <h4 id="comp-name"></h4>
        <p id="comp-desc"></p>
        <p id="comp-value"></p>
        <p id="comp-datasheet"></p>
    </div>
    
    <script>
        // Component database
        const components = {
            'U1': {
                name: 'Main Processor',
                description: 'STM32H743 ARM Cortex-M7',
                value: 'STM32H743ZIT6',
                datasheet: 'https://www.st.com/resource/en/datasheet/stm32h743zi.pdf',
                position: { x: 50, y: 30, z: 2 }
            },
            'U2': {
                name: 'FPGA',
                description: 'Xilinx Artix-7',
                value: 'XC7A35T-1CSG324I',
                datasheet: 'https://www.xilinx.com/support/documentation/data_sheets/ds180_7Series_Overview.pdf',
                position: { x: 80, y: 50, z: 2 }
            }
            // ... more components
        };
        
        // Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 1000
        );
        camera.position.set(100, 100, 100);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('viewer').appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(50, 50, 50);
        scene.add(directionalLight);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        // Load PCB model
        const loader = new THREE.STLLoader();
        loader.load('assets/pcb_model.stl', function(geometry) {
            const material = new THREE.MeshPhongMaterial({ 
                color: 0x0a7e0a,
                specular: 0x111111,
                shininess: 100
            });
            const pcb = new THREE.Mesh(geometry, material);
            scene.add(pcb);
        });
        
        // Add component markers
        const markerGeometry = new THREE.SphereGeometry(2, 32, 32);
        const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        
        const componentMeshes = {};
        
        for (const [refdes, component] of Object.entries(components)) {
            const marker = new THREE.Mesh(markerGeometry, markerMaterial);
            marker.position.set(
                component.position.x,
                component.position.y,
                component.position.z
            );
            marker.userData = { refdes, component };
            scene.add(marker);
            componentMeshes[refdes] = marker;
        }
        
        // Raycaster for mouse interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        function onMouseMove(event) {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        }
        
        function onMouseClick(event) {
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(Object.values(componentMeshes));
            
            if (intersects.length > 0) {
                const selected = intersects[0].object;
                const { refdes, component } = selected.userData;
                
                // Show component details
                document.getElementById('details').style.display = 'block';
                document.getElementById('comp-name').textContent = `${refdes}: ${component.name}`;
                document.getElementById('comp-desc').textContent = component.description;
                document.getElementById('comp-value').textContent = `Value: ${component.value}`;
                document.getElementById('comp-datasheet').innerHTML = 
                    `<a href="${component.datasheet}" target="_blank">View Datasheet</a>`;
                
                // Highlight selected component
                selected.material.color.setHex(0x00ff00);
                setTimeout(() => {
                    selected.material.color.setHex(0xff0000);
                }, 1000);
            }
        }
        
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('click', onMouseClick);
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
```

---

## 11. Advanced Topics

### Heterogeneous Computing with FPGA+CPU

```systemverilog
// fpga/rtl/cpu_accelerator_interface.sv
module cpu_accelerator_interface #(
    parameter AXI_DATA_WIDTH = 64,
    parameter AXI_ADDR_WIDTH = 32,
    parameter NUM_ACCELERATORS = 4
) (
    // AXI4 Slave Interface from CPU
    input  logic                        s_axi_aclk,
    input  logic                        s_axi_aresetn,
    // Write address channel
    input  logic [AXI_ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  logic [7:0]                  s_axi_awlen,
    input  logic [2:0]                  s_axi_awsize,
    input  logic [1:0]                  s_axi_awburst,
    input  logic                        s_axi_awvalid,
    output logic                        s_axi_awready,
    // Write data channel
    input  logic [AXI_DATA_WIDTH-1:0]   s_axi_wdata,
    input  logic [AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  logic                        s_axi_wlast,
    input  logic                        s_axi_wvalid,
    output logic                        s_axi_wready,
    // Write response channel
    output logic [1:0]                  s_axi_bresp,
    output logic                        s_axi_bvalid,
    input  logic                        s_axi_bready,
    
    // Accelerator interfaces
    output logic [NUM_ACCELERATORS-1:0] accel_start,
    input  logic [NUM_ACCELERATORS-1:0] accel_done,
    input  logic [NUM_ACCELERATORS-1:0] accel_error,
    output logic [31:0]                 accel_config [NUM_ACCELERATORS],
    input  logic [31:0]                 accel_status [NUM_ACCELERATORS],
    
    // DMA interfaces to accelerators
    output logic                        dma_req_valid,
    output logic [AXI_ADDR_WIDTH-1:0]   dma_req_addr,
    output logic [31:0]                 dma_req_len,
    input  logic                        dma_req_ready,
    
    // Interrupt to CPU
    output logic                        irq
);

    // Register map
    localparam ACCEL_CTRL_OFFSET   = 'h0000;
    localparam ACCEL_STATUS_OFFSET = 'h0100;
    localparam ACCEL_CONFIG_OFFSET = 'h0200;
    localparam DMA_ADDR_OFFSET     = 'h1000;
    localparam DMA_LEN_OFFSET      = 'h1004;
    localparam DMA_CTRL_OFFSET     = 'h1008;
    
    // Control registers
    logic [NUM_ACCELERATORS-1:0] accel_enable;
    logic [NUM_ACCELERATORS-1:0] accel_irq_enable;
    logic [NUM_ACCELERATORS-1:0] accel_done_latch;
    
    // AXI write handling
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            accel_enable <= '0;
            accel_irq_enable <= '0;
            accel_start <= '0;
            dma_req_valid <= '0;
        end else begin
            // Clear start pulses
            accel_start <= '0;
            
            if (s_axi_awvalid && s_axi_awready && s_axi_wvalid && s_axi_wready) begin
                case (s_axi_awaddr[11:0])
                    ACCEL_CTRL_OFFSET: begin
                        accel_enable <= s_axi_wdata[NUM_ACCELERATORS-1:0];
                        accel_start <= s_axi_wdata[NUM_ACCELERATORS-1:0] & ~accel_enable;
                    end
                    
                    ACCEL_CTRL_OFFSET + 4: begin
                        accel_irq_enable <= s_axi_wdata[NUM_ACCELERATORS-1:0];
                    end
                    
                    DMA_CTRL_OFFSET: begin
                        if (s_axi_wdata[0]) begin
                            dma_req_valid <= 1'b1;
                        end
                    end
                endcase
                
                // Accelerator configuration registers
                for (int i = 0; i < NUM_ACCELERATORS; i++) begin
                    if (s_axi_awaddr[11:0] == ACCEL_CONFIG_OFFSET + i*4) begin
                        accel_config[i] <= s_axi_wdata;
                    end
                end
            end
            
            // Clear DMA request when accepted
            if (dma_req_valid && dma_req_ready) begin
                dma_req_valid <= 1'b0;
            end
        end
    end
    
    // Completion detection and interrupt generation
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            accel_done_latch <= '0;
            irq <= 1'b0;
        end else begin
            // Latch completion status
            accel_done_latch <= accel_done_latch | accel_done;
            
            // Generate interrupt
            irq <= |(accel_done_latch & accel_irq_enable);
        end
    end

endmodule
```

### Machine Learning Inference Acceleration

```python
# tools/ml_accelerator_generator.py
"""Generate optimized FPGA accelerators for ML inference"""

import numpy as np
from typing import Dict, List, Tuple
import onnx
import torch

class MLAcceleratorGenerator:
    """Convert ONNX models to optimized Verilog"""
    
    def __init__(self, target_device: str = "xc7a100t"):
        self.device = target_device
        self.resource_budget = self._get_device_resources()
        
    def generate_accelerator(self, 
                           onnx_model: str,
                           optimization_level: int = 2) -> Dict:
        """Generate Verilog accelerator from ONNX model"""
        
        # Load and analyze model
        model = onnx.load(onnx_model)
        graph = model.graph
        
        # Extract layers and parameters
        layers = self._extract_layers(graph)
        
        # Optimize for FPGA
        optimized_layers = self._optimize_for_fpga(layers, optimization_level)
        
        # Generate Verilog
        verilog_modules = []
        for layer in optimized_layers:
            if layer['type'] == 'Conv2D':
                module = self._generate_conv2d_module(layer)
            elif layer['type'] == 'Dense':
                module = self._generate_dense_module(layer)
            elif layer['type'] == 'MaxPool2D':
                module = self._generate_maxpool_module(layer)
            else:
                continue
                
            verilog_modules.append(module)
        
        # Generate top-level module
        top_module = self._generate_top_module(optimized_layers, verilog_modules)
        
        # Resource estimation
        resources = self._estimate_resources(optimized_layers)
        
        return {
            'verilog': top_module,
            'modules': verilog_modules,
            'resources': resources,
            'latency': self._estimate_latency(optimized_layers),
            'throughput': self._estimate_throughput(optimized_layers)
        }
    
    def _generate_conv2d_module(self, layer: Dict) -> str:
        """Generate optimized 2D convolution module"""
        
        # Extract parameters
        in_channels = layer['in_channels']
        out_channels = layer['out_channels']
        kernel_size = layer['kernel_size']
        stride = layer['stride']
        weights = layer['weights']
        
        # Determine parallelism based on resources
        parallel_mult = min(
            self.resource_budget['dsp'] // (kernel_size * kernel_size),
            out_channels
        )
        
        verilog = f"""
module conv2d_{layer['name']} #(
    parameter IN_WIDTH = 8,
    parameter OUT_WIDTH = 16,
    parameter WEIGHT_WIDTH = 8
) (
    input  wire clk,
    input  wire rst_n,
    
    // Input feature map streaming interface
    input  wire [IN_WIDTH-1:0] in_data,
    input  wire in_valid,
    output wire in_ready,
    
    // Output feature map streaming interface  
    output reg  [OUT_WIDTH-1:0] out_data,
    output reg  out_valid,
    input  wire out_ready
);

    // Convolution engine with {parallel_mult} parallel multipliers
    localparam KERNEL_SIZE = {kernel_size};
    localparam IN_CHANNELS = {in_channels};
    localparam OUT_CHANNELS = {out_channels};
    localparam STRIDE = {stride};
    
    // Line buffers for sliding window
    reg [IN_WIDTH-1:0] line_buffer [KERNEL_SIZE-1:0][INPUT_WIDTH-1:0];
    
    // Kernel weights (from quantized model)
    reg [WEIGHT_WIDTH-1:0] weights [OUT_CHANNELS-1:0][IN_CHANNELS-1:0][KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    
    initial begin
        // Initialize weights from model
        {self._generate_weight_init(weights)}
    end
    
    // Systolic array for convolution
    wire [OUT_WIDTH-1:0] partial_sums [OUT_CHANNELS-1:0];
    
    genvar i, j;
    generate
        for (i = 0; i < OUT_CHANNELS; i = i + 1) begin : out_ch
            // Instantiate MAC units
            mac_array #(
                .SIZE(KERNEL_SIZE * KERNEL_SIZE * IN_CHANNELS),
                .DATA_WIDTH(IN_WIDTH),
                .WEIGHT_WIDTH(WEIGHT_WIDTH),
                .OUT_WIDTH(OUT_WIDTH)
            ) mac_inst (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(/* window data */),
                .weights(weights[i]),
                .result(partial_sums[i])
            );
        end
    endgenerate
    
    // Output accumulation and activation
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
        end else begin
            // ReLU activation
            if (/* convolution complete */) begin
                out_data <= (partial_sums[/* current channel */] > 0) ? 
                           partial_sums[/* current channel */] : 0;
                out_valid <= 1'b1;
            end
        end
    end

endmodule
"""
        return verilog
```

### Power Analysis and Optimization

```python
# tools/power_analysis.py
"""Analyze and optimize power consumption"""

import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class PowerAnalyzer:
    """Comprehensive power analysis for hardware designs"""
    
    def __init__(self, design_files: Dict):
        self.schematic = design_files['schematic']
        self.pcb_layout = design_files['layout']
        self.fpga_power = design_files.get('fpga_power_report')
        
    def analyze_total_power(self) -> Dict:
        """Complete system power analysis"""
        
        # Static power analysis
        static_power = self._analyze_static_power()
        
        # Dynamic power analysis
        dynamic_power = self._analyze_dynamic_power()
        
        # Thermal analysis
        thermal = self._thermal_analysis(
            static_power['total'] + dynamic_power['total']
        )
        
        return {
            'static': static_power,
            'dynamic': dynamic_power,
            'total': static_power['total'] + dynamic_power['total'],
            'thermal': thermal,
            'efficiency': self._calculate_efficiency(),
            'battery_life': self._estimate_battery_life(),
            'optimization_suggestions': self._suggest_optimizations()
        }
    
    def _analyze_static_power(self) -> Dict:
        """Analyze quiescent power consumption"""
        
        components = self._extract_components()
        
        static_power = {
            'regulators': 0,
            'always_on': 0,
            'leakage': 0,
            'total': 0
        }
        
        # Voltage regulators quiescent current
        for reg in components['regulators']:
            iq = reg.get('quiescent_current_ma', 0)
            vin = reg.get('input_voltage', 0)
            static_power['regulators'] += iq * vin / 1000  # Convert to watts
        
        # Always-on circuits
        for comp in components['always_on']:
            static_power['always_on'] += comp['power_mw'] / 1000
        
        # FPGA static power
        if self.fpga_power:
            fpga_static = self._parse_fpga_power_report()
            static_power['leakage'] = fpga_static['static_power_w']
        
        static_power['total'] = sum(v for k, v in static_power.items() if k != 'total')
        
        return static_power
    
    def _analyze_dynamic_power(self) -> Dict:
        """Analyze active power consumption"""
        
        # Parse switching activity
        activity = self._parse_switching_activity()
        
        dynamic_power = {
            'digital_switching': 0,
            'analog_circuits': 0,
            'io_power': 0,
            'clock_tree': 0,
            'total': 0
        }
        
        # Digital switching power: P = α * C * V² * f
        for net in activity['nets']:
            capacitance = self._extract_net_capacitance(net['name'])
            voltage = net['voltage']
            frequency = net['switching_frequency']
            activity_factor = net['activity_factor']
            
            power = activity_factor * capacitance * voltage**2 * frequency
            dynamic_power['digital_switching'] += power
        
        # Clock tree power (typically 30-40% of dynamic power)
        clock_nets = [n for n in activity['nets'] if 'clk' in n['name'].lower()]
        for clock in clock_nets:
            fanout = self._get_clock_fanout(clock['name'])
            clock_cap = fanout * 15e-15  # ~15fF per gate input
            dynamic_power['clock_tree'] += clock_cap * clock['voltage']**2 * clock['frequency']
        
        # I/O power
        io_standards = {
            'LVCMOS33': {'voltage': 3.3, 'current': 8e-3},
            'LVCMOS18': {'voltage': 1.8, 'current': 4e-3},
            'LVDS': {'voltage': 1.2, 'current': 3.5e-3}
        }
        
        for io in activity['io_pins']:
            std = io_standards.get(io['standard'])
            if std:
                dynamic_power['io_power'] += std['voltage'] * std['current'] * io['toggle_rate']
        
        dynamic_power['total'] = sum(v for k, v in dynamic_power.items() if k != 'total')
        
        return dynamic_power
    
    def _thermal_analysis(self, total_power: float) -> Dict:
        """Thermal modeling and analysis"""
        
        # PCB thermal model
        pcb_area = self._calculate_pcb_area()  # cm²
        copper_coverage = self._estimate_copper_coverage()  # percentage
        
        # Thermal resistance calculation
        # Empirical formula for FR4 with copper planes
        theta_ja = 50 - (copper_coverage * 0.3) - (pcb_area * 0.1)  # °C/W
        
        # Component-specific thermal analysis
        components = []
        
        # Main heat sources
        heat_sources = [
            {'name': 'FPGA', 'power': total_power * 0.4, 'theta_jc': 2.5},
            {'name': 'CPU', 'power': total_power * 0.3, 'theta_jc': 3.0},
            {'name': 'Power Supply', 'power': total_power * 0.15, 'theta_jc': 5.0}
        ]
        
        ambient_temp = 25  # °C
        
        for source in heat_sources:
            junction_temp = ambient_temp + source['power'] * (source['theta_jc'] + theta_ja)
            components.append({
                'component': source['name'],
                'power_dissipation': source['power'],
                'junction_temperature': junction_temp,
                'margin': 85 - junction_temp,  # Assuming 85°C max junction temp
                'heatsink_required': junction_temp > 70
            })
        
        return {
            'ambient_temperature': ambient_temp,
            'board_thermal_resistance': theta_ja,
            'components': components,
            'total_board_dissipation': total_power,
            'cooling_requirement': 'passive' if max(c['junction_temperature'] for c in components) < 70 else 'active'
        }
    
    def _suggest_optimizations(self) -> List[Dict]:
        """Suggest power optimization strategies"""
        
        suggestions = []
        
        # Analyze voltage rails
        rails = self._get_voltage_rails()
        for rail in rails:
            if rail['efficiency'] < 0.85:
                suggestions.append({
                    'category': 'Power Supply',
                    'issue': f"{rail['name']} efficiency is {rail['efficiency']*100:.1f}%",
                    'suggestion': 'Consider switching to a more efficient regulator',
                    'potential_savings': f"{rail['power_loss']*0.3:.1f}W"
                })
        
        # Clock gating opportunities
        clocks = self._analyze_clock_domains()
        for clock in clocks:
            if clock['always_on'] and clock['average_activity'] < 0.5:
                suggestions.append({
                    'category': 'Clock Gating',
                    'issue': f"{clock['name']} is always on but only {clock['average_activity']*100:.1f}% active",
                    'suggestion': 'Implement clock gating',
                    'potential_savings': f"{clock['power']*0.4:.1f}W"
                })
        
        # I/O optimization
        ios = self._analyze_io_power()
        for io_bank in ios:
            if io_bank['voltage'] > io_bank['required_voltage']:
                suggestions.append({
                    'category': 'I/O Voltage',
                    'issue': f"{io_bank['name']} uses {io_bank['voltage']}V but only needs {io_bank['required_voltage']}V",
                    'suggestion': 'Lower I/O voltage to reduce power',
                    'potential_savings': f"{io_bank['power']*0.3:.1f}W"
                })
        
        return suggestions
```

### Safety and Reliability

```python
# tools/safety_analysis.py
"""Safety and reliability analysis for hardware designs"""

import numpy as np
from scipy import stats
import pandas as pd

class SafetyAnalyzer:
    """Comprehensive safety and reliability analysis"""
    
    def __init__(self, design_files: Dict):
        self.components = self._load_component_database()
        self.operating_conditions = design_files.get('operating_conditions')
        
    def perform_fmea(self) -> pd.DataFrame:
        """Failure Mode and Effects Analysis"""
        
        fmea_data = []
        
        # Analyze each component
        for component in self.components:
            failure_modes = self._get_failure_modes(component)
            
            for mode in failure_modes:
                severity = self._assess_severity(component, mode)
                occurrence = self._assess_occurrence(component, mode)
                detection = self._assess_detection(component, mode)
                
                rpn = severity * occurrence * detection  # Risk Priority Number
                
                fmea_data.append({
                    'Component': component['reference'],
                    'Function': component['function'],
                    'Failure Mode': mode['description'],
                    'Effect': mode['effect'],
                    'Cause': mode['cause'],
                    'Severity': severity,
                    'Occurrence': occurrence,
                    'Detection': detection,
                    'RPN': rpn,
                    'Mitigation': self._suggest_mitigation(component, mode, rpn)
                })
        
        # Create DataFrame and sort by RPN
        fmea_df = pd.DataFrame(fmea_data)
        fmea_df = fmea_df.sort_values('RPN', ascending=False)
        
        return fmea_df
    
    def calculate_mtbf(self) -> Dict:
        """Calculate Mean Time Between Failures"""
        
        # MIL-HDBK-217F based calculation
        total_failure_rate = 0
        
        component_failures = []
        
        for component in self.components:
            # Base failure rate
            lambda_b = self._get_base_failure_rate(component)
            
            # Environmental factor
            pi_e = self._get_environment_factor()
            
            # Temperature factor
            temp = self.operating_conditions.get('temperature', 25)
            pi_t = self._calculate_temperature_factor(component, temp)
            
            # Quality factor
            pi_q = self._get_quality_factor(component)
            
            # Component failure rate
            lambda_p = lambda_b * pi_e * pi_t * pi_q
            
            total_failure_rate += lambda_p
            
            component_failures.append({
                'component': component['reference'],
                'type': component['type'],
                'failure_rate': lambda_p,
                'mtbf_hours': 1 / lambda_p if lambda_p > 0 else float('inf')
            })
        
        # System MTBF
        system_mtbf = 1 / total_failure_rate if total_failure_rate > 0 else float('inf')
        
        return {
            'system_mtbf_hours': system_mtbf,
            'system_mtbf_years': system_mtbf / (24 * 365),
            'reliability_1_year': np.exp(-total_failure_rate * 24 * 365),
            'component_failures': sorted(component_failures, 
                                       key=lambda x: x['failure_rate'], 
                                       reverse=True)[:10],
            'critical_components': [c for c in component_failures 
                                  if c['mtbf_hours'] < system_mtbf * 10]
        }
    
    def thermal_derating_analysis(self) -> Dict:
        """Analyze component derating for reliability"""
        
        derating_issues = []
        
        for component in self.components:
            ratings = component.get('ratings', {})
            operating = component.get('operating_point', {})
            
            # Voltage derating (typically 80% for ceramics, 50% for electrolytics)
            if 'voltage' in ratings and 'voltage' in operating:
                derating = operating['voltage'] / ratings['voltage']
                recommended = 0.5 if component['type'] == 'electrolytic_cap' else 0.8
                
                if derating > recommended:
                    derating_issues.append({
                        'component': component['reference'],
                        'parameter': 'voltage',
                        'current_derating': f"{derating*100:.0f}%",
                        'recommended': f"{recommended*100:.0f}%",
                        'action': 'Increase voltage rating or reduce operating voltage'
                    })
            
            # Power derating (typically 50% at max ambient)
            if 'power' in ratings and 'power' in operating:
                temp_rise = operating.get('temperature_rise', 40)
                power_derating = operating['power'] / ratings['power']
                
                # Adjust for temperature
                max_temp = self.operating_conditions.get('max_temperature', 70)
                temp_factor = 1 - (max_temp - 25) / (ratings.get('max_temp', 125) - 25)
                effective_derating = power_derating / temp_factor
                
                if effective_derating > 0.5:
                    derating_issues.append({
                        'component': component['reference'],
                        'parameter': 'power',
                        'current_derating': f"{effective_derating*100:.0f}%",
                        'recommended': '50%',
                        'action': 'Increase power rating or improve cooling'
                    })
        
        return {
            'issues': derating_issues,
            'overall_reliability': 'Good' if len(derating_issues) == 0 else 'Needs attention',
            'recommendations': self._generate_derating_recommendations(derating_issues)
        }
```

### Automated Testing Infrastructure

```python
# tests/automated_test_framework.py
"""Comprehensive automated testing framework for hardware"""

import asyncio
import pytest
from typing import Dict, List, Callable
import yaml
import time

class HardwareTestOrchestrator:
    """Orchestrate complex hardware test sequences"""
    
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.test_stations = {}
        self.results_database = TestResultsDatabase()
        
    async def run_production_test(self, serial_number: str) -> Dict:
        """Run complete production test sequence"""
        
        start_time = time.time()
        test_record = {
            'serial_number': serial_number,
            'start_time': start_time,
            'test_steps': [],
            'overall_result': 'PASS'
        }
        
        # Initialize test fixture
        fixture = await self._setup_test_fixture(serial_number)
        
        try:
            # 1. Boundary scan test
            bscan_result = await self.run_boundary_scan_test(fixture)
            test_record['test_steps'].append(bscan_result)
            
            if not bscan_result['passed']:
                test_record['overall_result'] = 'FAIL'
                return test_record
            
            # 2. Power-on test
            power_result = await self.run_power_on_test(fixture)
            test_record['test_steps'].append(power_result)
            
            # 3. Programming
            prog_result = await self.program_device(fixture)
            test_record['test_steps'].append(prog_result)
            
            # 4. Functional tests
            func_result = await self.run_functional_tests(fixture)
            test_record['test_steps'].append(func_result)
            
            # 5. Performance characterization
            perf_result = await self.characterize_performance(fixture)
            test_record['test_steps'].append(perf_result)
            
            # 6. Environmental stress screening
            if self.config.get('enable_ess', False):
                ess_result = await self.run_ess_test(fixture)
                test_record['test_steps'].append(ess_result)
            
            # 7. Final QC
            qc_result = await self.final_qc_check(fixture)
            test_record['test_steps'].append(qc_result)
            
        except Exception as e:
            test_record['overall_result'] = 'ERROR'
            test_record['error'] = str(e)
            
        finally:
            # Clean up
            await fixture.power_down()
            test_record['end_time'] = time.time()
            test_record['duration'] = test_record['end_time'] - start_time
            
            # Save results
            await self.results_database.save(test_record)
            
            # Generate labels if passed
            if test_record['overall_result'] == 'PASS':
                await self.generate_qc_label(serial_number, test_record)
        
        return test_record
    
    async def run_boundary_scan_test(self, fixture) -> Dict:
        """JTAG boundary scan testing"""
        
        print("\n[BOUNDARY SCAN TEST]")
        print("-" * 50)
        
        result = {
            'test_name': 'Boundary Scan',
            'start_time': time.time(),
            'subtests': []
        }
        
        # Initialize JTAG
        jtag = fixture.get_jtag_interface()
        
        # Test: Scan chain integrity
        print("Checking JTAG chain integrity...")
        devices = await jtag.scan_chain()
        
        expected_devices = self.config['jtag']['expected_devices']
        chain_ok = len(devices) == len(expected_devices)
        
        for i, (found, expected) in enumerate(zip(devices, expected_devices)):
            if found['idcode'] != expected['idcode']:
                chain_ok = False
                print(f"  Device {i}: FAIL - Expected {expected['idcode']:08X}, "
                      f"found {found['idcode']:08X}")
            else:
                print(f"  Device {i}: PASS - {expected['name']}")
        
        result['subtests'].append({
            'name': 'Chain Integrity',
            'passed': chain_ok
        })
        
        # Test: Interconnect test
        if chain_ok:
            print("\nRunning interconnect test...")
            interconnect_result = await jtag.run_interconnect_test(
                self.config['jtag']['bsdl_files']
            )
            
            print(f"  Nets tested: {interconnect_result['nets_tested']}")
            print(f"  Shorts found: {interconnect_result['shorts_found']}")
            print(f"  Opens found: {interconnect_result['opens_found']}")
            
            result['subtests'].append({
                'name': 'Interconnect',
                'passed': interconnect_result['shorts_found'] == 0 and 
                         interconnect_result['opens_found'] == 0,
                'details': interconnect_result
            })
        
        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']
        result['passed'] = all(t['passed'] for t in result['subtests'])
        
        return result
    
    async def characterize_performance(self, fixture) -> Dict:
        """Detailed performance characterization"""
        
        print("\n[PERFORMANCE CHARACTERIZATION]")
        print("-" * 50)
        
        result = {
            'test_name': 'Performance Characterization',
            'start_time': time.time(),
            'measurements': {}
        }
        
        # 1. Frequency response
        print("Measuring frequency response...")
        freq_response = await self._measure_frequency_response(fixture)
        result['measurements']['frequency_response'] = freq_response
        
        # 2. Power consumption vs load
        print("Measuring power efficiency...")
        power_curve = await self._measure_power_curve(fixture)
        result['measurements']['power_efficiency'] = power_curve
        
        # 3. Thermal characterization
        print("Thermal characterization...")
        thermal = await self._thermal_characterization(fixture)
        result['measurements']['thermal'] = thermal
        
        # 4. EMC pre-scan
        print("EMC pre-compliance scan...")
        emc = await self._emc_prescan(fixture)
        result['measurements']['emc'] = emc
        
        # Generate performance certificate
        self._generate_performance_certificate(
            fixture.serial_number,
            result['measurements']
        )
        
        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']
        result['passed'] = self._evaluate_performance(result['measurements'])
        
        return result
    
    async def _measure_frequency_response(self, fixture) -> Dict:
        """Measure frequency response characteristics"""
        
        network_analyzer = fixture.get_network_analyzer()
        
        measurements = {}
        
        # Configure for S-parameter measurement
        await network_analyzer.configure(
            start_freq=10e3,    # 10 kHz
            stop_freq=6e9,      # 6 GHz
            points=1601,
            power=-10           # dBm
        )
        
        # Measure critical paths
        for path in self.config['performance']['rf_paths']:
            print(f"  Measuring {path['name']}...")
            
            # Set up path
            await fixture.configure_rf_path(path['input'], path['output'])
            
            # Measure S-parameters
            s_params = await network_analyzer.measure_s_parameters()
            
            # Extract key metrics
            measurements[path['name']] = {
                's21_magnitude': s_params['S21']['magnitude'],
                's21_phase': s_params['S21']['phase'],
                's11_magnitude': s_params['S11']['magnitude'],
                'bandwidth_3db': self._calculate_bandwidth(s_params['S21']['magnitude']),
                'insertion_loss': -max(s_params['S21']['magnitude']),
                'return_loss': -min(s_params['S11']['magnitude']),
                'group_delay': self._calculate_group_delay(s_params['S21']['phase'])
            }
        
        return measurements
```

### Design for Manufacturing (DFM) Automation

```python
# tools/dfm_checker.py
"""Automated Design for Manufacturing checks"""

import pcbnew
import numpy as np
from shapely.geometry import Polygon, Point
import json

class DFMChecker:
    """Comprehensive DFM rule checking"""
    
    def __init__(self, board_file: str, assembly_side: str = 'top'):
        self.board = pcbnew.LoadBoard(board_file)
        self.assembly_side = assembly_side
        self.violations = []
        
    def run_all_checks(self) -> Dict:
        """Run comprehensive DFM analysis"""
        
        print("Running Design for Manufacturing Analysis...")
        print("=" * 60)
        
        # Component placement checks
        placement_issues = self.check_component_placement()
        
        # Solder paste checks
        paste_issues = self.check_solder_paste()
        
        # Assembly checks
        assembly_issues = self.check_assembly_constraints()
        
        # Testability checks
        test_issues = self.check_testability()
        
        # Generate report
        report = {
            'summary': {
                'total_issues': len(self.violations),
                'critical': len([v for v in self.violations if v['severity'] == 'critical']),
                'warnings': len([v for v in self.violations if v['severity'] == 'warning']),
                'info': len([v for v in self.violations if v['severity'] == 'info'])
            },
            'placement': placement_issues,
            'solder_paste': paste_issues,
            'assembly': assembly_issues,
            'testability': test_issues,
            'violations': self.violations
        }
        
        return report
    
    def check_component_placement(self) -> Dict:
        """Check component placement rules"""
        
        issues = {
            'courtyard_violations': [],
            'placement_grid': [],
            'orientation': [],
            'keepout_violations': []
        }
        
        # Get all footprints
        footprints = list(self.board.GetFootprints())
        
        # Check courtyard overlaps
        for i, fp1 in enumerate(footprints):
            if fp1.GetSide() != self._get_side_layer():
                continue
                
            courtyard1 = self._get_courtyard_polygon(fp1)
            if not courtyard1:
                continue
                
            for fp2 in footprints[i+1:]:
                if fp2.GetSide() != self._get_side_layer():
                    continue
                    
                courtyard2 = self._get_courtyard_polygon(fp2)
                if not courtyard2:
                    continue
                
                if courtyard1.intersects(courtyard2):
                    self._add_violation(
                        'critical',
                        f"Courtyard overlap: {fp1.GetReference()} and {fp2.GetReference()}",
                        fp1.GetPosition()
                    )
                    issues['courtyard_violations'].append({
                        'ref1': fp1.GetReference(),
                        'ref2': fp2.GetReference()
                    })
        
        # Check placement grid (typically 0.5mm for pick&place)
        grid_size = 0.5 * 1e6  # Convert to nm
        
        for fp in footprints:
            pos = fp.GetPosition()
            if pos.x % grid_size != 0 or pos.y % grid_size != 0:
                self._add_violation(
                    'warning',
                    f"{fp.GetReference()} not on {grid_size/1e6}mm grid",
                    pos
                )
                issues['placement_grid'].append(fp.GetReference())
        
        # Check rotation (0, 90, 180, 270 degrees preferred)
        for fp in footprints:
            rotation = fp.GetOrientation().AsDegrees()
            if rotation % 90 != 0:
                self._add_violation(
                    'info',
                    f"{fp.GetReference()} rotation {rotation}° not multiple of 90°",
                    fp.GetPosition()
                )
                issues['orientation'].append({
                    'ref': fp.GetReference(),
                    'rotation': rotation
                })
        
        return issues
    
    def check_solder_paste(self) -> Dict:
        """Check solder paste stencil design rules"""
        
        issues = {
            'aspect_ratio': [],
            'min_aperture': [],
            'paste_reduction': []
        }
        
        # Typical stencil thickness
        stencil_thickness = 0.127  # mm (5 mil)
        
        for footprint in self.board.GetFootprints():
            if footprint.GetSide() != self._get_side_layer():
                continue
                
            for pad in footprint.Pads():
                # Get paste aperture size
                paste_size = pad.GetSolderPasteMargin()
                pad_size = pad.GetSize()
                
                # Calculate actual aperture
                aperture_x = pad_size.x + 2 * paste_size
                aperture_y = pad_size.y + 2 * paste_size
                
                # Convert from nm to mm
                aperture_x_mm = aperture_x / 1e6
                aperture_y_mm = aperture_y / 1e6
                
                # Check minimum aperture (typically 0.2mm)
                min_dimension = min(aperture_x_mm, aperture_y_mm)
                if min_dimension < 0.2:
                    self._add_violation(
                        'critical',
                        f"Paste aperture too small on {footprint.GetReference()} "
                        f"pad {pad.GetNumber()}: {min_dimension:.3f}mm",
                        pad.GetPosition()
                    )
                    issues['min_aperture'].append({
                        'ref': footprint.GetReference(),
                        'pad': pad.GetNumber(),
                        'size': min_dimension
                    })
                
                # Check aspect ratio (width:thickness should be > 1.5)
                aspect_ratio = min_dimension / stencil_thickness
                if aspect_ratio < 1.5:
                    self._add_violation(
                        'warning',
                        f"Poor paste release aspect ratio on {footprint.GetReference()} "
                        f"pad {pad.GetNumber()}: {aspect_ratio:.2f}",
                        pad.GetPosition()
                    )
                    issues['aspect_ratio'].append({
                        'ref': footprint.GetReference(),
                        'pad': pad.GetNumber(),
                        'ratio': aspect_ratio
                    })
        
        return issues
    
    def check_testability(self) -> Dict:
        """Check in-circuit test requirements"""
        
        issues = {
            'missing_testpoints': [],
            'testpoint_access': [],
            'probe_clearance': []
        }
        
        # Minimum test probe spacing (typically 1.27mm/50mil)
        min_probe_spacing = 1.27 * 1e6  # nm
        
        # Find all test points
        testpoints = []
        for footprint in self.board.GetFootprints():
            if 'TP' in footprint.GetReference() or 'TEST' in footprint.GetValue().upper():
                testpoints.append(footprint)
        
        # Check critical nets have test points
        critical_nets = ['3V3', '5V', 'GND', 'RESET', 'CLK']
        board_nets = self.board.GetNetsByName()
        
        for net_name in critical_nets:
            if net_name in board_nets:
                net = board_nets[net_name]
                has_testpoint = False
                
                for pad in net.Pads():
                    fp = pad.GetParent()
                    if fp in testpoints:
                        has_testpoint = True
                        break
                
                if not has_testpoint:
                    self._add_violation(
                        'warning',
                        f"Critical net '{net_name}' has no test point",
                        None
                    )
                    issues['missing_testpoints'].append(net_name)
        
        # Check test point spacing
        for i, tp1 in enumerate(testpoints):
            pos1 = tp1.GetPosition()
            
            for tp2 in testpoints[i+1:]:
                pos2 = tp2.GetPosition()
                distance = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
                
                if distance < min_probe_spacing:
                    self._add_violation(
                        'critical',
                        f"Test points {tp1.GetReference()} and {tp2.GetReference()} "
                        f"too close: {distance/1e6:.2f}mm",
                        pos1
                    )
                    issues['probe_clearance'].append({
                        'tp1': tp1.GetReference(),
                        'tp2': tp2.GetReference(),
                        'distance': distance/1e6
                    })
        
        return issues
    
    def generate_assembly_drawing(self, output_file: str):
        """Generate assembly drawing with DFM annotations"""
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Rectangle, Circle
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        # Draw board outline
        edge_cuts = self.board.GetBoardEdgesBoundingBox()
        board_rect = Rectangle(
            (edge_cuts.GetX()/1e6, edge_cuts.GetY()/1e6),
            edge_cuts.GetWidth()/1e6,
            edge_cuts.GetHeight()/1e6,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(board_rect)
        
        # Draw components
        for footprint in self.board.GetFootprints():
            if footprint.GetSide() != self._get_side_layer():
                continue
                
            pos = footprint.GetPosition()
            x, y = pos.x/1e6, pos.y/1e6
            
            # Draw component body
            bbox = footprint.GetBoundingBox()
            comp_rect = Rectangle(
                (bbox.GetX()/1e6, bbox.GetY()/1e6),
                bbox.GetWidth()/1e6,
                bbox.GetHeight()/1e6,
                fill=True,
                facecolor='lightgray',
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(comp_rect)
            
            # Add reference designator
            ax.text(x, y, footprint.GetReference(),
                   ha='center', va='center', fontsize=6)
        
        # Highlight violations
        for violation in self.violations:
            if violation['position']:
                x, y = violation['position'].x/1e6, violation['position'].y/1e6
                
                if violation['severity'] == 'critical':
                    color = 'red'
                elif violation['severity'] == 'warning':
                    color = 'orange'
                else:
                    color = 'yellow'
                
                circle = Circle((x, y), 2, fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(circle)
        
        # Set aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(edge_cuts.GetX()/1e6 - 10, 
                   (edge_cuts.GetX() + edge_cuts.GetWidth())/1e6 + 10)
        ax.set_ylim(edge_cuts.GetY()/1e6 - 10,
                   (edge_cuts.GetY() + edge_cuts.GetHeight())/1e6 + 10)
        
        # Add title and grid
        ax.set_title(f'Assembly Drawing - {self.assembly_side.upper()} Side\n'
                    f'DFM Issues: {len(self.violations)}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
    
    def _add_violation(self, severity: str, message: str, position):
        """Add a DFM violation"""
        self.violations.append({
            'severity': severity,
            'message': message,
            'position': position,
            'timestamp': datetime.now().isoformat()
        })
    
    def _get_side_layer(self):
        """Get PCB layer for assembly side"""
        return pcbnew.F_Cu if self.assembly_side == 'top' else pcbnew.B_Cu
```

---

## 12. Conclusion and Best Practices Summary

### Key Takeaways

1. **Design for Test from Day One**
   - Include test points on all critical signals
   - Design in boundary scan capability
   - Plan for automated testing

2. **Version Control Everything**
   - Use Git LFS for binary files
   - Maintain clear commit messages
   - Tag releases properly

3. **Document Obsessively**
   - Keep documentation in sync with design
   - Use interactive documentation where possible
   - Include assembly and service information

4. **Automate Repetitive Tasks**
   - Generate production files automatically
   - Automate BOM management
   - Use CI/CD for validation

5. **Think About Manufacturing Early**
   - Run DFM checks regularly
   - Consider assembly constraints
   - Plan for testing and debugging

6. **Prioritize Safety and Reliability**
   - Perform FMEA analysis
   - Derate components appropriately
   - Plan for thermal management

### Continuous Learning Resources

- **Forums**: EEVblog, Reddit r/PrintedCircuitBoard, Hackaday
- **Training**: IPC certification, Altium/KiCad courses
- **Standards**: IPC-2221 (PCB design), IPC-A-610 (acceptability)
- **Books**: "High Speed Digital Design" by Howard Johnson
- **YouTube**: Phil's Lab, Robert Feranec, EEVblog

### Future Trends (2025-2026)

- **AI-Assisted Design**: Automated routing and optimization
- **Digital Twins**: Real-time simulation of physical hardware
- **Chiplet Integration**: Modular chip architectures
- **Sustainable Electronics**: Design for recycling and repair
- **6G Hardware**: THz frequencies and new materials

Remember: hardware development is iterative. Each project teaches valuable lessons. Document your failures as thoroughly as your successes—they're often more instructive.

Happy hacking! 🛠️⚡🔬