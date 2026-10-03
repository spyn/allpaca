# Product Requirements Document (PRD)
## Project Name: Allpaca — Distributed Local LLM Inference Cluster & Web Console

---

### 1. Executive Summary & Vision

**Allpaca** is a high-performance, developer-grade distributed inference server and cluster telemetry console tailored for local home-lab and small-team compute clusters. Its core objective is to expose an OpenAI/Ollama-compatible local inference API across a local area network (LAN) by aggregating heterogeneous local hardware (such as ROCm/HIP AMD GPUs, Vulkan acceleration, and remote LAN worker nodes) into a unified compute pipeline.

The web console provides a mission-control interface giving engineers full visibility into hardware utilization, dynamic KV cache allocation, multi-GPU layer distribution, real-time token generation telemetry, model catalogue management, and a rich developer playground with transparent reasoning traces.

---

### 2. Problem Statement & User Needs

#### 2.1 Problems
1. **Opaque Multi-Device Memory Orchestration**: Modern high-parameter models (e.g., 27B–70B+ weights or MoEs) exceed single-GPU VRAM limits. Quantized weights, KV caches, and multimodal vision projectors are fragmented across multiple devices without clear visual feedback.
2. **Hidden Speculative Decoding Efficiency**: Multi-Token Prediction (MTP) and speculative draft models can boost throughput significantly, but developers lack real-time visibility into acceptance rates, latency gains, and overhead.
3. **Troubleshooting Distributed Nodes**: When requests fail or degrade over LAN RPC connections (10GbE), developers are left digging through raw terminal stdout streams without structured log filtering or subsystem correlation.

#### 2.2 User Personas
- **Home-Lab AI Engineers & Enthusiasts**: Building multi-GPU workstations (e.g., RX 7900 XTX + RX 9070/9060 XT setups) desiring maximum tok/s, GGUF/EXL2 format support, and fine-grained VRAM management.
- **Local Application Developers**: Engineers connecting local IDEs, agents, and web applications to local ports (`:11434`, `:8080`) requiring stable uptime, low TTFT (Time-to-First-Token), and OpenAI-compatible API schemas.
- **System Administrators / Hardware Tinkers**: Managing mixed-hardware nodes over high-speed LAN, monitoring thermal/wattage/PCIe limits and network socket health.

---

### 3. Core Architecture & System Overview

```
                          [ Client Applications / LAN Devices ]
                                        │
                               (HTTP Tower :11434 / :8080)
                                        │
                               ┌────────▼────────┐
                               │  Allpaca Engine │
                               │  (Orchestrator) │
                               └────────┬────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │ (PCIe Local Bus)           │ (PCIe Local Bus)           │ (10GbE LAN / RPC)
    ┌──────▼──────┐              ┌──────▼──────┐              ┌──────▼──────┐
    │ GPU 0 (HIP) │              │ GPU 1 (HIP) │              │ Remote Node │
    │ RX 9070     │              │ RX 9060 XT  │              │ alpha-rig   │
    │ Layers 0-32 │              │ Layers 33-64│              │ Standby/MTP │
    └─────────────┘              └─────────────┘              └─────────────┘
```

- **Backend Runtime**: C++/HIP engine supporting ROCm 6.1+, Vulkan RPC sharding, FlashAttention-2, and Paged Attention.
- **Speculative Decoding**: Support for internal/external MTP draft models with verifiable draft token validation.
- **Network Interface**: FastAPI / HTTP Tower server with SSE (Server-Sent Events) and WebSocket pipelines for telemetry streams (`/api/v1/cluster/traces`).

---

### 4. Detailed Feature & Functional Requirements

#### 4.1 Global App Shell & Navigation Chrome
- **Cluster Status Indicator**: Live socket connection status, active build version (`v0.9.4-rc`), active cluster profile (`HOME-LAB`), and active ports (`:11434, :8080`).
- **Persistent Header Navigation**: Instant switching between **Dashboard**, **Chat Playground**, **Models & VRAM**, and **Logs & Traces**.
- **Global Actions**: API documentation drawer modal, connection settings, authentication lock, and user/profile toggle.
- **Global Health Footer**: Sticky bottom summary displaying active master node hostname, backend drivers (ROCm / Vulkan), aggregate memory consumption (e.g., `68.2 GiB / 96.0 GiB`), and total connected cluster nodes.

---

#### 4.2 Module 1: Cluster Telemetry Dashboard (`Cluster Telemetry Dashboard`)
- **Active Model Target Card**:
  - Displays loaded model name, parameter count, quantization scheme, and engine backend status.
  - Active request counter, cumulative tokens processed per session, and base API endpoint URL.
  - Quick action controls: **Switch Model** and **Run Benchmark**.
- **Live Throughput Telemetry**:
  - Real-time rolling generation speed (tokens per second) with a 60-second historical SVG sparkline.
  - Rolling target baseline indicator.
- **Global Cluster Memory Distribution Bar**:
  - Segmented horizontal visual bar showing:
    - Model weights (GiB)
    - Active KV cache footprint (MiB/GiB)
    - System buffers, desktop compositor overhead & OS memory
    - Available headroom
- **Modular Hardware & Engine Metric Cards**:
  - **Host RAM**: Used vs total with % bar, memory type (e.g., DDR5-6000), dual-channel indicators.
  - **GPU VRAM per Device**: Individual cards with real-time temperature (°C), wattage (W), and PCIe link speed.
  - **Weights on Device**: Exact on-device footprint with Paged Attention status.
  - **Engine Latency**: Inter-Token Latency (ITL in ms), Time to First Token (TTFT in ms), and p95 latency.
  - **Host CPU Load**: Utilization percentage, thread count (e.g., 16c/32t Ryzen 9 9950X), clock speed.
  - **KV Pages**: Total pages allocated, DRAM spillover counters, and on-chip hit rates.
  - **Speculative Decoding**: Draft acceptance rate percentage, speedup factor (e.g., `+1.98x`), gamma factor.
  - **RPC & Network**: Round-trip time (RTT in ms), activation payload sizes, and remote node status.
- **Compute Hardware Topology Matrix**:
  - Inventory table of all devices (Host CPU, Local HipGPUs, Remote RPC Workers over LAN).
  - Columns: Device ID, Driver Type, Hardware Identifier, Bus Address, Temperature & Power, Visual Memory Allocation Bar, Compute Utilization, and Engine Layer Role (e.g., *Layers 0-32 Primary*).
  - Interactive button: **Attach Remote RPC Worker**.
- **Live Event Stream**: Monospace real-time feed logging inference initiation, prompt token evaluation, and draft verification rates.

---

#### 4.3 Module 2: Model Management & Dynamic VRAM Estimator (`Models & VRAM`)
- **Loaded Model Inspection Hero**:
  - Filepath reference, architecture family (`qwen35`, `llama`, `nemotron`), parameter count, quantization algorithm (`UD-Q5_K_XL`, `FP16`, `ROCmFP4`), trained context window, layer/attention head breakdown, embedding dimension, and quantization author.
  - **Local VRAM Fit Badge**: Dynamic computation calculating whether weights + KV cache fit safely into local VRAM or require RPC offloading.
- **Speculative & Projector Configuration Bar**:
  - Selectors for Multimodal Vision Projectors (`mmproj-F16.gguf`).
  - Speculation Mode options (MTP Internal Head vs External Draft Weights).
  - One-click hot-reload button (**Reload with these options**).
- **Interactive Context Size & Dynamic KV Cache Allocation Slider**:
  - Continuous slider from 2,048 tokens up to 131,072+ tokens with milestone snap points (Optimal, High VRAM).
  - Real-time computation of KV Cache Footprint (GiB) based on context length, per-token memory cost (KiB), and active attention layers.
  - Context save & allocate CTA button.
- **Model Storage Directory Scanner & Catalog Table**:
  - Directory path tracker with active search/regex filter and directory **Rescan** trigger.
  - Filterable inventory table listing: Model ID & Filepath, Architecture, Size/Quantization, Native Context, Sharding Topology (dense vs MoE active experts), Disk Footprint, and Action button (**Select** / **Details**).
  - Visual status pill for currently active loaded models.

---

#### 4.4 Module 3: Chat Playground & Developer Debugger (`Chat Playground`)
- **Top Inference Telemetry Ribbon**:
  - Active model tag, live context consumption meter (`4,120 / 32,768 tokens · 12.5%`), last TTFT timer, generation speed (tok/s), and active MTP gamma factor.
  - Controls for **System Prompt**, **Export JSON**, and toggle for **Sampling & Cluster Drawer**.
- **Interactive Message Flow**:
  - Clean conversational bubbles with clear visual contrast between User and Assistant.
  - **Collapsible `<think>` Reasoning Blocks**: Dedicated styled container for deep thinking models displaying total thinking tokens, elapsed time, and collapsible disclosure toggle.
  - **Per-Message Telemetry Metadata**: Token count, TTFT, average generation tok/s, RPC worker verification tag, one-click copy, and re-generate actions.
  - **Inference Status Callout**: Live streaming indicator showing thinking state, prompt evaluation phase, and speculative draft acceptance hit rate.
- **Right Slide-out Drawer: Sampling & Nodes**:
  - Real-time cluster memory gauges per GPU and RPC node.
  - Hyperparameter sliders: Temperature, Top-P (Nucleus), Repetition Penalty, Max Generation Cap.
  - Speculative decoding parameters & Advanced Flash-Attention / RoPE configuration.
- **Input Terminal & Multimodal Dock**:
  - Auto-expanding textarea with support for multimodal attachments (`mmproj CLIP-ViT`).
  - Shortcut prompts (`Shift+Enter` for newline, `Ctrl+Enter` to send).

---

#### 4.5 Module 4: Real-time Logs & Cluster Tracing (`Logs & Traces`)
- **Subsystem Connection Banner**: Engine pipe state, WebSocket stream endpoint, ROCm device sync count, pause stream, clear terminal, and `.ndjson` dump export.
- **Multi-Level Filter Toolbar**:
  - Level dropdown (`ALL`, `INFO & Above`, `WARN`, `ERROR`).
  - Monospace regex and text filter input with keyboard reset.
  - Subsystem quick-toggle chips: `All Subsystems`, `HTTP Tower (:11434)`, `Engine Core (ROCm/HIP)`, `RPC Cluster LAN`, `Speculative MTP`, `API Token Stream`.
  - Terminal view toggles: **Follow Output (Auto-Scroll)** and **Soft Wrap**.
- **Monospace Terminal Screen**:
  - High-density virtualized terminal feed with line numbering, timestamp, colored log level pill, subsystem source, and formatted output.
- **Cluster Socket Health Footer**:
  - Live node count, aggregate VRAM allocation, throughput counter, and HTTP response code monitor (`200 OK`, error traces count).

---

### 5. Design System & UI Specifications

The interface is built upon the **Obsidian Telemetry** design system:
- **Base Canvas & Surfaces**: Deep graphite & dark obsidian palette (`#0a0e13`, `#101419`, `#181c21`, `#36393f`).
- **Accent & Status Colors**:
  - Primary Accent & Success: Emerald green (`#10b981`, `#34d399`) indicating nominal operation and live pipelines.
  - High Performance / Active Stream: Neon Cyan (`#06b6d4`, `#22d3ee`).
  - Memory & Quantization Categories: Electric Indigo / Purple (`#a855f7`, `#818cf8`).
  - Warning / Alert: Amber (`#f59e0b`).
  - Error / OOM Guard: Crimson (`#ef4444`).
- **Typography**: `Geist` and `Geist Mono` for razor-sharp telemetry readouts, numeric alignment, and terminal logs.
- **Corner Radii & Borders**: Compact `rounded-sm` (4px–6px) aesthetic conveying industrial, developer-focused tooling.

---

### 6. Non-Functional Requirements & Performance Targets

1. **Telemetry Streaming Latency**: Dashboard metrics and sparklines must refresh at `<500ms` intervals via WebSockets without causing UI thread jank.
2. **Terminal Virtualization**: The log viewer must support buffer depths of >10,000 lines without frame drops using virtual DOM rendering.
3. **Local Responsiveness**: Zero external internet dependencies; all fonts, icons, and assets are hosted locally on the Allpaca node.
4. **Safety & Guardrails**: The VRAM allocator must actively warn and prevent users from allocating context sizes exceeding available GPU capacity.

---

### 7. Future Roadmap & Milestones
- **Phase 1 (Completed)**: Core UI Design System, Telemetry Dashboard, Chat Playground with `<think>` blocks, Models & VRAM Estimator, Logs & Traces Console.
- **Phase 2 (Immediate Next)**: Dynamic model benchmarking suite, automated quantization tester (PPL / perplexity evaluation on local datasets).
- **Phase 3**: LoRA adapter hot-swapping and automated multi-node load balancing across LAN instances.
