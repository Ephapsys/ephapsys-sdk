# ============================================================
# 🚀 Ephapsys Modulator Trainer Dockerfile (CUDA + HF Evaluate)
# ============================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app

# ------------------------------------------------------------
# 🔧 System Setup
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git python3-pip jq pciutils ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# ------------------------------------------------------------
# 📦 Python Dependencies
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 🧹 Hugging Face Config
# ------------------------------------------------------------
RUN rm -rf /root/.cache/huggingface && \
    unset HF_EVALUATE_OFFLINE && unset HF_HUB_OFFLINE
ENV HF_EVALUATE_OFFLINE=0 \
    HF_HUB_OFFLINE=0 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    HF_HUB_ENABLE_HF_TRANSFER=0

# ------------------------------------------------------------
# 🔐 Token Handling
# ------------------------------------------------------------
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ------------------------------------------------------------
# 🧪 Optional Hugging Face sanity check (build-safe)
# ------------------------------------------------------------
RUN cat <<'PY' > /tmp/hfcheck.py
import importlib, sys, requests, os
print("🔍 Checking installed versions:")
for pkg in ["evaluate", "huggingface_hub", "datasets"]:
    mod = importlib.import_module(pkg)
    print(f"✅ {pkg}: {getattr(mod, '__version__', 'unknown')}")

print("\n🌐 Testing authenticated Hugging Face access...")
url = "https://huggingface.co/datasets/metrics/accuracy/raw/main/accuracy.py"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"} if os.getenv("HF_TOKEN") else {}
r = requests.get(url, headers=headers, timeout=10)
print(f"✅ Hub responded {r.status_code}, length {len(r.text)}")

print("\n🧩 Verifying direct metric access...")
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        "metrics/accuracy",
        "accuracy.py",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    )
    print(f"✅ Metric file fetched: {path}")
except Exception as e:
    print(f"⚠️  Hub file fetch failed: {type(e).__name__}: {e}")

print("\n🎉 Hugging Face Hub sanity check complete.")
PY

RUN if [ -n "$HF_TOKEN" ]; then \
      echo "🔐 HF_TOKEN detected — running Hugging Face sanity check..."; \
      python3 /tmp/hfcheck.py; \
    else \
      echo "⚠️  No HF_TOKEN provided — skipping Hugging Face check (will authenticate at runtime)."; \
    fi && rm -f /tmp/hfcheck.py

# ------------------------------------------------------------
# 📁 Copy Project Files
# ------------------------------------------------------------
COPY .docker_sdk/ephapsys /app/ephapsys
COPY train_language.py .

# ------------------------------------------------------------
# 🚀 Entrypoint
# ------------------------------------------------------------
ENTRYPOINT ["python3", "train_language.py"]
