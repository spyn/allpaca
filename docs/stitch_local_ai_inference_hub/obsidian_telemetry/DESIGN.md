---
name: Obsidian Telemetry
colors:
  surface: '#101419'
  surface-dim: '#101419'
  surface-bright: '#36393f'
  surface-container-lowest: '#0a0e13'
  surface-container-low: '#181c21'
  surface-container: '#1c2025'
  surface-container-high: '#262a30'
  surface-container-highest: '#31353b'
  on-surface: '#e0e2ea'
  on-surface-variant: '#bbcabf'
  inverse-surface: '#e0e2ea'
  inverse-on-surface: '#2d3136'
  outline: '#86948a'
  outline-variant: '#3c4a42'
  surface-tint: '#4edea3'
  primary: '#4edea3'
  on-primary: '#003824'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#006c49'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#d0bcff'
  on-tertiary: '#3c0091'
  tertiary-container: '#b090ff'
  on-tertiary-container: '#4600a7'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#e9ddff'
  tertiary-fixed-dim: '#d0bcff'
  on-tertiary-fixed: '#23005c'
  on-tertiary-fixed-variant: '#5516be'
  background: '#101419'
  on-background: '#e0e2ea'
  surface-variant: '#31353b'
typography:
  headline-xl:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  metric-display:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 28px
    letterSpacing: -0.03em
  metric-unit:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
  label-section:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.08em
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-code:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  margin: 1.25rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system targets machine learning engineers, systems architects, and homelab hobbyists orchestrating distributed local LLM inference engines and GPU clusters. The core aesthetic merges **minimalist technical dashboard precision** with **developer-centric high-density tooling**.

The emotional tone balances absolute mechanical reliability, quiet low-light efficiency, and instantaneous observability. Visually, the experience is anchored in ultra-deep obsidian and slate backdrops punctuated by razor-thin dark borders, high-legibility numerical metrics, and deliberate, luminescent accents: energetic emerald green indicating operational health and model readiness, with sharp neon cyan delineating remote RPC cluster nodes.

## Colors

The palette is tuned specifically for pitch-black server rooms and multi-monitor developer environments:

- **Primary (`#10B981`)**: Electric Emerald. Applied to healthy telemetry states, active throughput indicators, loaded model indicators, and primary action triggers.
- **Secondary (`#06B6D4`)**: Neon Cyan. Dedicated to RPC remote nodes, network transport statistics, distributed workers, and packet sync indicators.
- **Tertiary (`#8B5CF6`)**: Deep Violet. Used for memory allocation categorizations (weights on device, static tensor memory) to clearly contrast against dynamic KV cache footprints.
- **Neutral (`#0B0F14`)**: Pitch Obsidian Canvas. Forms the foundational floor, layering upward into `#111827` (card panels) and `#1F2937` (subtle separating strokes and input fields).

Warning and critical states map strictly to `#F59E0B` (VRAM pressure / memory spillover) and `#EF4444` (node disconnect, inference panic, dropped connection).

## Typography

Typography leverages a tripartite hierarchy:
1. **Geist** delivers crisp geometric forms for screen titles, telemetry card titles, and high-impact numerical metric displays.
2. **Inter** handles explanatory documentation, subtexts, and configuration descriptions where sustained reading clarity is required.
3. **JetBrains Mono** powers technical metadata: parameter counts, quantization strings (`UD-Q5_K_XL`), IP sockets, context token numbers, and section category banners.

All micro-labels are rendered uppercase with intentional tracking (`0.08em`) to guarantee quick recognition on dense displays.

## Layout & Spacing

The layout is built for high information density across a standard 12-column fluid grid, optimized for wide and ultrawide technical displays.
- Section boundaries and card groupings adhere to a strict 4px/8px micro-grid.
- Telemetry grids leverage responsive CSS subgrids: 4 columns across standard desktop (`>= 1280px`), wrapping to 2 columns on tablet (`>= 768px`), and a stacked 1-column layout on mobile (`< 768px`).
- Section gaps are compressed down to `1.25rem` to keep core execution statistics (throughput, VRAM consumption, latency, and RPC health) above the visual fold.

## Elevation & Depth

Depth is established primarily through **tonal separation** and **hairline stroke stratification**, rather than diffuse drop shadows:

- **Base Canvas**: Surface-0 (`#0B0F14`) - zero elevation.
- **Card Containers**: Surface-1 (`#111827`) bordered with a crisp `1px` continuous border in `#1F2937`.
- **Nested Controls & Tables**: Surface-2 (`#151E2E`) with outline tint `#374151`.
- **Glow Accents**: Active telemetry curves, active nodes, and online indicators emit a subtle `0 0 12px rgba(16, 185, 129, 0.15)` optical bloom, ensuring telemetry traces feel alive without introducing visual noise.

## Shapes

Shapes reflect an industrial, precision-tooled aesthetic. A roundedness factor of `1` provides controlled micro-radii:
- Container cards, metric tiles, and inputs use `0.375rem` (6px) corners.
- Badges, status pills, and small buttons use `0.25rem` (4px) corners.
- Progress bars and segmented VRAM charts use crisp `2px` corners to preserve exact pixel-bound metric proportions.

## Components

### Telemetry Metric Cards
Metric cards feature a muted uppercase mono header, a primary readout combining `metric-display` with an inline `metric-unit`, and a sub-metric line (e.g., `TTFT 462 ms`). Sparkline charts occupy either the card's right quadrant or span the bottom with a 1.5px continuous stroke (`#10B981`) and a linear fill gradient fading to zero opacity at the baseline.

### Device & Cluster Grid Table
The device telemetry cluster displays hardware units (CPU, local GPUs, and remote RPC nodes). Rows maintain high-density padding (`8px 12px`), monospace hardware addresses (`rpc[192.168.69.2:50052#0]`), tabular numerical alignment for memory capacity, and a real-time status pip (emerald for connected, cyan for streaming, amber for throttling).

### Segmented VRAM Estimator
A multi-colored horizontal bar chart tracking memory allocations:
- Model Weights: Violet (`#8B5CF6`)
- Other processes: Slate (`#4B5563`)
- Active KV Cache: Emerald (`#10B981`)
- Unallocated Headroom: Charcoal (`#1F2937`)

### Interactive Controls & Sliders
Token slider tracks feature a solid background (`#1F2937`) with filled segments utilizing `#10B981`. Handles are circular (`14px`), high-contrast `#FFFFFF`, with a subtle outer glow on focus. Companion numeric input fields use monospace numbers, embedded directly next to the scrubber.

### Buttons & Mode Badges
- **Primary Button**: Solid emerald (`#10B981`) with dark slate text (`#0B0F14`) and font-weight 600.
- **Secondary / Action Button**: Transparent background with `#374151` border, transitioning to `#4B5563` on hover.
- **Status Badges**: Semi-transparent backgrounds (`rgba(16, 185, 129, 0.1)`) with a solid 1px border and a pulsing `6px` status dot.