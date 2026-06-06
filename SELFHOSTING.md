# useknockout self-hosting guide

[← Back to the README](./README.md)

Want to run your own instance? One command after Modal setup.

### Prerequisites

```bash
pip install modal
modal token new
```

### Clone & deploy

```bash
git clone https://github.com/useknockout/api.git
cd api

# create your bearer-token secret
modal secret create knockout-secrets API_TOKEN=$(openssl rand -hex 32)

# deploy
modal deploy main.py
```

Modal prints your live HTTPS URL. First deploy takes ~5 min (image build + weight bake). Subsequent deploys take seconds.

### Tune for your workload

Edit `main.py`:

```python
@app.cls(
    gpu="L4",              # or "A10", "A100", "H100"
    scaledown_window=60,   # seconds of idle before scale-to-zero
    max_containers=10,     # max concurrent containers
)
```

- **Latency-critical?** Keep one warm: `min_containers=1` (costs ~$0.80/hr 24/7).
- **Throughput-critical?** Bump `max_containers` and use `@modal.concurrent(max_inputs=4)` to batch.
- **Higher quality?** Change `MODEL_INPUT_SIZE` to `(2048, 2048)` — 4x slower, sharper edges.
