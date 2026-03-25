# V-JEPA Integration Plan for `samples/agents/robot`

This note captures a practical plan for evolving the robot sample from
single-frame vision classification into a temporal visual-state pipeline using
V-JEPA.

## Goal

Upgrade the robot sample so its vision path models scene dynamics over time
instead of classifying isolated frames.

The target outcome is:

- better temporal awareness
- better memory retrieval from visual context
- stronger grounding for the language model
- a clearer path toward future action-planning or anomaly-detection workflows

## Concrete Model Choice

For the **Ephapsys robot integration path**, we should standardize on the
official Hugging Face checkpoint:

- **`facebook/vjepa2-vitl-fpc64-256`**
- Hugging Face URL: `https://huggingface.co/facebook/vjepa2-vitl-fpc64-256`

Why:

- it is an official, directly consumable Hugging Face model artifact
- it is the cleanest HF-first integration path for Ephapsys
- it avoids making the robot integration depend on custom fork-only weights
- it gives us a stable model-template target for idempotent publish first

### Important distinction

We should separate:

1. **HF-first Ephapsys integration**
   - model target: `facebook/vjepa2-vitl-fpc64-256`
   - goal: secure wrapping, robot integration, runtime functionality

2. **Independent evaluation / fork path**
   - repo target: `facebookresearch/vjepa2`
   - likely backbone for constrained GPU environments: **ViT-B**
   - goal: experiment, modify, benchmark, and demo world-model behavior

### Default recommendation

- **Robot sample integration target:** `facebook/vjepa2-vitl-fpc64-256`
- **Independent evaluation target on limited VRAM:** V-JEPA 2 ViT-B from the fork path

### Not recommended for v1 robot integration

- building the robot sample around a fork-specific checkpoint first
- requiring the fork path before we can ship a functional Ephapsys integration

## Ephaptic Modulation Strategy

The long-term goal is not only to use V-JEPA as a vision/world-model backend,
but also to support **ephaptic modulation of that visual backbone** through the
same model-template and TrustedAgent lifecycle used elsewhere in Ephapsys.

However, the first robot-sample rollout should prioritize functionality and
integration safety.

### Phase 0: Functional baseline with idempotent mode

For the robot sample, the initial production-facing path should use:

- **idempotent mode**
- certificate generation
- provenance wrapping
- policy binding
- no quality-impacting modulation updates

This gives us:

- a working V-JEPA-backed robot path
- full Ephapsys asset lifecycle coverage
- no early regression risk from changing model behavior

### Phase 1: True ephaptic modulation later

After the backend works end to end in idempotent mode, we can introduce actual
ephaptic modulation experiments for V-JEPA:

- define visual KPIs
- run controlled modulation trials
- compare baseline vs modulated scene embeddings / downstream task quality
- evaluate latency and VRAM impact

### Initial robot-sample rule

For the robot sample:

- **all required models can start in idempotent mode**
- especially the V-JEPA vision model
- the sample remains functionally usable before any performance-tuning work

This is the right order:

1. make it work
2. wrap it securely
3. evaluate it
4. modulate it for performance only after we have a stable baseline

## Why V-JEPA

The current robot sample treats vision as a simple image understanding step.
That is good for a basic demo, but it is not ideal for robotics-style context,
where the agent should reason over:

- what changed in the scene
- what persists across frames
- what is likely to happen next
- whether the current situation resembles a previous episode

V-JEPA is a better fit for this because it is fundamentally a temporal visual
representation model rather than a single-frame classifier.

## What Should Change

Instead of:

1. capture one frame
2. run a classifier/detector
3. send a label into the LLM

we want:

1. capture a rolling window of recent frames
2. encode that window with V-JEPA
3. produce a compact scene/state embedding
4. optionally compare it against prior visual memories
5. summarize the result into structured context for the language model

## Proposed v1 Architecture

### Existing high-level robot flow

1. microphone input
2. STT
3. camera frame
4. vision inference
5. memory lookup
6. language response
7. TTS

### Proposed V-JEPA flow

1. microphone input
2. STT
3. rolling camera frame buffer
4. V-JEPA scene encoder
5. visual embedding + similarity lookup
6. language response with temporal visual context
7. TTS

## Proposed New Components

### 1. `vision/vjepa_adapter.py`

Purpose:

- load V-JEPA model and preprocessing stack
- accept a short frame sequence
- return a temporal scene embedding

Likely API:

```python
class VJEPAAdapter:
    def __init__(self, model_id: str, device: str = "cpu"):
        ...

    def encode_frames(self, frames: list) -> dict:
        """
        Returns:
          {
            "embedding": np.ndarray,
            "shape": [dim],
            "meta": {...}
          }
        """
```

### 2. `vision/frame_buffer.py`

Purpose:

- maintain the most recent N frames
- support stride / sampling interval
- expose a ready-to-encode frame window

Likely API:

```python
class FrameBuffer:
    def __init__(self, max_frames: int = 8):
        ...

    def push(self, frame) -> None:
        ...

    def ready(self) -> bool:
        ...

    def get_window(self) -> list:
        ...
```

### 3. `vision/visual_memory.py`

Purpose:

- store V-JEPA embeddings in FAISS
- retrieve the most similar past visual episodes
- enable temporal scene recall

This can reuse the existing embedding/memory pattern already present in the
robot sample, but with visual-state vectors instead of text-only memory.

### 4. `vision/visual_summary.py`

Purpose:

- convert raw visual embedding + nearest-neighbor matches into compact LLM input
- keep the prompt stable and lightweight

Example output:

```json
{
  "visual_state": {
    "summary": "Indoor workspace scene with a persistent seated person and limited motion.",
    "change_score": 0.14,
    "novelty_score": 0.08,
    "nearest_memories": [
      "Prior desk conversation scene",
      "Previous seated operator state"
    ]
  }
}
```

## Prompting Strategy

Do not feed raw embeddings into the language model.

Instead, feed a summarized visual state block, for example:

```text
Visual context:
- Scene state: desk environment with one person present
- Motion/change: low
- Similar past situations: previous seated desk interactions
```

This keeps the LLM prompt readable, deterministic, and cheap.

## v1 Scope

The first integration should be narrow.

### In scope

- frame window buffering
- V-JEPA embedding extraction
- FAISS-backed nearest-memory lookup for visual state
- summarized visual context injected into the LLM prompt

### Out of scope

- robot control policies
- navigation
- action planning
- reinforcement learning
- full video event detection
- training or fine-tuning V-JEPA itself

## Model/Runtime Considerations

Questions to settle before implementation:

1. Which V-JEPA checkpoint do we want to support first?
2. Can we run it acceptably on CPU for sample/demo purposes?
3. Do we want GPU-only support in v1?
4. Do we need a smaller fallback path for local laptops?

Likely practical answer:

- support **`facebook/vjepa2-vitl-fpc64-256`** first for Ephapsys integration
- support a GPU-friendly path first
- keep the current vision model as fallback
- expose vision backend selection via env var

Suggested env var:

```bash
ROBOT_VISION_BACKEND=yolos   # or vjepa
ROBOT_VJEPA_MODEL_ID=facebook/vjepa2-vitl-fpc64-256
ROBOT_VISION_MODE=idempotent   # later: modulated
```

## Evaluation Task

A practical external evaluation track for V-JEPA should use the public
`facebookresearch/vjepa2` repo and focus on a small, demoable set of
modifications.

Suggested task:

1. Fork `https://github.com/facebookresearch/vjepa2`.
2. Make a small set of modifications around:
   - live webcam inference
   - real-time video recognition or interaction detection
   - clearer demo/runtime ergonomics
3. Prepare a short demo and summary of:
   - what was changed
   - how the world model behaves in live webcam mode
   - what parts are most relevant to eventual Ephapsys integration

Because V-JEPA 2 is VRAM-heavy, the evaluation path should assume:

- a larger GPU if available for the heavier checkpoints
- or a smaller **ViT-B** configuration when running in more constrained
  environments

### What this implies for our integration plan

- the repo to study and fork is `facebookresearch/vjepa2`
- the near-term independent evaluation should focus on **V-JEPA 2 ViT-B**
- the target demo is **live webcam world-model / video recognition**
- the integration should favor temporal inference and scene-state reasoning,
  not just image classification
- we should expect GPU-backed experimentation for serious evaluation

## Suggested Execution Track

### Track A: External fork evaluation

1. Fork `facebookresearch/vjepa2`.
2. Add small changes around:
   - webcam inference
   - real-time recognition/demo UX
   - logging/visualization
   - runtime ergonomics
3. Capture the resulting evaluation setup and outcomes.

### Track B: Ephapsys robot integration

In parallel, we use what is learned from the fork to shape:

1. a V-JEPA adapter for the robot sample
2. an idempotent-mode model-template path in Ephapsys
3. future modulation experiments on the same backbone

## GPU Guidance

### Preferred

- local 24 GB GPU, or
- comparable cloud GPU access

### Acceptable constrained fallback

- Colab/Kaggle free tier
- **must use ViT-B** for the fork/evaluation track

### Avoid for initial work

- assuming a large V-JEPA 2 checkpoint on 16 GB or smaller environments
- designing the robot sample around hardware most contributors will not have

## Proposed Ephapsys Asset Plan

For the robot sample, the V-JEPA path should eventually have:

1. a **model template** for `facebook/vjepa2-vitl-fpc64-256`
2. an **idempotent publish path** for secure wrapping with no performance change
3. a **robot agent template** referencing that V-JEPA model
4. later, a **true modulation path** for experimental ephaptic tuning

### Initial recommendation

- create and register the `facebook/vjepa2-vitl-fpc64-256` model template
- publish it in **idempotent mode** first
- bind it into the robot agent template
- only later run real ephaptic tuning jobs on that backbone

## Suggested Migration Path

### Phase 1

- add a backend abstraction for vision
- keep existing vision backend as default
- add V-JEPA backend behind an env flag

### Phase 2

- add visual memory retrieval
- add structured visual summaries to prompt assembly

### Phase 3

- add scene change / novelty scoring
- add episode logging for richer replay and evaluation

## File-Level Plan

Likely touch points:

- `robot_agent.py`
  - swap single-frame vision step for backend abstraction
  - add frame buffer lifecycle
  - inject visual-state summary into prompt construction

- `run_local.sh`
  - optional env flags for V-JEPA enablement

- `README.md`
  - document default vision backend
  - document V-JEPA experimental mode

Potential new files:

- `vision/vjepa_adapter.py`
- `vision/frame_buffer.py`
- `vision/visual_memory.py`
- `vision/visual_summary.py`

## Risks

- V-JEPA may be too heavy for the current sample’s default runtime expectations
- embedding quality may be good but not directly interpretable
- prompt quality may degrade if summary logic is weak
- frame-window capture may introduce latency

## Recommendation

Use V-JEPA as an optional experimental backend first, not the default.

That lets us:

- preserve the robot sample’s current simplicity
- compare temporal embeddings against current classifier-based behavior
- keep the integration reversible if runtime cost is too high

## Minimal v1 Success Criteria

We should call the integration successful if:

1. robot sample can run with `ROBOT_VISION_BACKEND=vjepa`
2. it can encode a rolling frame window into a stable embedding
3. it can retrieve similar prior visual states from memory
4. the LLM can use the summarized visual context in responses
5. fallback to the existing vision backend still works cleanly
6. the initial V-JEPA asset path works in **idempotent mode**
7. the design leaves a clean path for later ephaptic modulation experiments
