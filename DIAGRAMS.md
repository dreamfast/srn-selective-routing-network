# SRN Diagram Specifications

Instructions for generating SVG diagrams to explain SRN architecture. Each diagram should be clean, minimal, and use a consistent color scheme. Target audience: someone who knows what a neural network is but not the details of Transformers or routing.

**Style guide:**
- Dark background (#1a1a2e) with light text (#e0e0e0) — looks good on GitHub dark mode
- Accent colors: routing blue (#4fc3f7), attention orange (#ffb74d), expert green (#81c784), bottleneck purple (#ce93d8), gate red (#ef5350)
- Rounded rectangles for modules, arrows for data flow
- Tensor shapes annotated in monospace where helpful
- Keep text large enough to read at 600px width
- No gradients or shadows — flat design

---

## Diagram 1: "Transformer vs SRN — The Core Difference"

**Purpose:** Show why SRN exists. Side-by-side comparison.

**Left side — Transformer:**
- Show N tokens (e.g. 8 colored dots in a row)
- Draw arrows from EVERY token to EVERY other token (n² connections)
- Label: "Self-Attention: every token talks to every other token"
- Label: "O(n²) — gets expensive fast"
- Show the connection count growing: n=1K → 1M connections, n=32K → 1B connections

**Right side — SRN:**
- Show same N tokens (8 dots)
- Show k memory slots (e.g. 4 slots, drawn as squares/diamonds above the tokens)
- Draw arrows from each token to only 2-3 of the slots (sparse routing)
- Label: "Slot Routing: each token picks a few memory slots"
- Label: "O(n·k) — stays cheap regardless of sequence length"
- Show: n=1K → 4K connections, n=32K → 128K connections

**Bottom:** A simple bar chart or comparison showing the 512× reduction at 32K tokens.

---

## Diagram 2: "Inside an SRN Layer"

**Purpose:** Show the three modules (DSR → CSP → GEM) and what each does.

**Layout:** Vertical flow, one box per module with residual connections shown as bypass arrows.

```
Input x
  │
  ├──────────────────────┐
  │                      │ (residual)
  ▼                      │
┌─────────────────────┐  │
│  DSR (Router)       │  │
│  "Where should this │  │
│   token look?"      │  │
│  [BLUE]             │  │
└─────────┬───────────┘  │
          │              │
          ▼ (+)◄─────────┘
          │
  ├──────────────────────┐
  │                      │ (residual)
  ▼                      │
┌─────────────────────┐  │
│  CSP (Bottleneck)   │  │
│  "What info is      │  │
│   worth keeping?"   │  │
│  [PURPLE]           │  │
└─────────┬───────────┘  │
          │              │
          ▼ (+)◄─────────┘
          │
  ├──────────────────────┐
  │                      │ (residual)
  ▼                      │
┌─────────────────────┐  │
│  GEM (Experts)      │  │
│  "Process with      │  │
│   specialist FFNs"  │  │
│  [GREEN]            │  │
└─────────┬───────────┘  │
          │              │
          ▼ (+)◄─────────┘
          │
       Output x
```

**Annotations:**
- DSR box: small illustration of token → slot routing arrows
- CSP box: show wide → narrow → wide (D → D/4 → D) with a gate icon
- GEM box: show 8 expert boxes but only 2 highlighted (active), others grayed out

---

## Diagram 3: "WCSG — How Routing Stays Causal"

**Purpose:** Explain the key innovation. This is the hardest to visualize.

**Panel A — "The Problem":**
- Show a sequence of 6 tokens: [The] [cat] [sat] [on] [the] [mat]
- Show 4 memory slots above
- Problem: slots are global/shared — how does token 3 ("sat") know not to use information from token 6 ("mat")?
- Show a red X on the connection from "sat" to future-influenced slots
- Label: "Memory slots are shared — no natural causal boundary"

**Panel B — "The Solution: Score Gating":**
- Show the same 6 tokens
- For token 3 ("sat"), show a small window highlighting tokens 1-3 only (the causal window)
- Show this window producing a gate vector: [0.8, 0.9, 0.2, 0.1] (one per slot)
- Show the raw routing scores being multiplied by this gate
- Result: slots 3 and 4 get suppressed (low gate), slots 1 and 2 stay strong
- Label: "Each position gets a causal gate based on its local past context"
- Label: "Gate can suppress routes but can't create new ones — intentional trade-off"

**Panel C — "Memory Cost":**
- Simple comparison boxes:
  - "Naive approach: per-position keys → 4.3 GB/layer" [RED, big box]
  - "WCSG: gate tensor → 1 MB/layer" [GREEN, tiny box]
- Label: "4,300× less memory"

---

## Diagram 4: "Sparse Experts — Why 75% of the Model Sleeps"

**Purpose:** Explain the GEM module and why sparsity = efficiency.

**Layout:** Show 8 expert boxes in a row. A token comes in from the left.

- A small router box decides which 2 experts to activate
- 2 experts are highlighted (green, active), 6 are grayed out (dormant)
- The token flows through only the 2 active experts
- Outputs are combined with learned weights

**Key stats annotated:**
- "8 experts total → large knowledge capacity"
- "2 active per token → small compute cost"
- "75% of FFN parameters dormant per forward pass"
- "Different tokens activate different experts → specialization"

**Bonus panel:** Show 4 different tokens each activating different expert pairs, illustrating how the full capacity gets used across a batch even though each token only uses a fraction.

---

## Diagram 5: "Hybrid SRN — The Best of Both Worlds"

**Purpose:** Show the architecture that actually works (based on Exp1 results).

**Layout:** Show a stack of 12 layers, vertical.

```
Layer 0:  [ORANGE] Attention    ← "Captures global patterns"
Layer 1:  [BLUE]   Routing      ← "Cheap, efficient"
Layer 2:  [BLUE]   Routing
Layer 3:  [BLUE]   Routing
Layer 4:  [ORANGE] Attention    ← "Periodic attention anchor"
Layer 5:  [BLUE]   Routing
Layer 6:  [BLUE]   Routing
Layer 7:  [BLUE]   Routing
Layer 8:  [ORANGE] Attention    ← "Corrects routing drift"
Layer 9:  [BLUE]   Routing
Layer 10: [BLUE]   Routing
Layer 11: [BLUE]   Routing
```

**Side annotations:**
- "75% routing layers → O(n·k) cheap"
- "25% attention layers → O(n²) but only 3 of them"
- "Result: near-Transformer quality at a fraction of the cost"

**Bottom comparison bar:**
- Pure Transformer: 12/12 attention layers, val_loss = 1.37
- Hybrid SRN: 3 attention + 9 routing, val_loss = ~1.5 (TBD)
- Pure SRN: 12/12 routing layers, val_loss = 2.53

---

## Diagram 6: "The Gap Decomposition"

**Purpose:** Visualize the ablation experiment results.

**Layout:** Horizontal bar chart or waterfall chart.

**Bars (left to right):**
1. Transformer 184M: val_loss = 1.373 [ORANGE bar]
2. Transformer 112M: val_loss = 1.470 [lighter ORANGE]
3. Hybrid SRN (Exp1): val_loss = TBD [BLUE-ORANGE gradient]
4. SRN Baseline: val_loss = 2.528 [BLUE bar]

**Annotations between bars:**
- Gap between Transformer 184M and 112M: "Compute gap: 0.097 (negligible)"
- Gap between Transformer 112M and SRN: "Architecture gap: 1.058"
- Gap closed by hybrid: arrow showing how much Exp1 recovered

**Bottom text:** "92% of the quality gap is architectural, not computational. Adding just 3 attention layers closes most of it."

---

## Diagram 7: "Scaling Vision — SRN at Different Sizes"

**Purpose:** Show the efficiency thesis at scale.

**Layout:** Table/infographic showing model sizes.

| Size | Total Params | Active/Token | VRAM (fp16) | Activity |
|------|-------------|-------------|-------------|----------|
| SRN-Small | 328M | 119M | 0.6 GB | 36% |
| SRN-Medium | 3.9B | 1.0B | 7.3 GB | 26% |
| SRN-Large | 37.7B | 5.4B | 70 GB | 14% |

**Visual:** Nested circles or boxes showing total params (outer) vs active params (inner, highlighted). The ratio gets more dramatic at larger scales.

**Punchline:** "A 3.9B SRN fits in 8GB VRAM and computes like a 1B dense model, but has the knowledge capacity of a 3.9B model."

---

## How to generate these

Paste this file into Claude (web) and ask:

> "Generate SVG diagrams for each of these 7 specifications. Make them clean, minimal, dark-themed, and suitable for embedding in a GitHub README. Each should be self-contained and work at 600-800px width."

Then save each SVG to `docs/diagrams/` and reference them in the README with:

```markdown
![Transformer vs SRN](docs/diagrams/01-transformer-vs-srn.svg)
```
