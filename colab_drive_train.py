import subprocess
import os
import sys
import re
import threading
import shutil
import time

# ── 1. Dependencies ───────────────────────────────────────────────────────────
print("=== Installing dependencies ===")
subprocess.run(["pip", "install", "-q", "stable-worldmodel[train]"], check=True)

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
if not os.path.exists("/content/le-wm"):
    print("=== Cloning le-wm ===")
    subprocess.run(["git", "clone", "https://github.com/lucas-maes/le-wm.git", "/content/le-wm"], check=True)

# Patch config to use lance
config_path = "/content/le-wm/config/train/data/pusht.yaml"
with open(config_path) as f:
    cfg = f.read()
cfg = re.sub(r"pusht_expert_train(?:\.lance)?(?:\.h5)?", "pusht_expert_train.lance", cfg)
with open(config_path, "w") as f:
    f.write(cfg)
print("=== Config set to lance format ===")

# ── 3. Mount Google Drive for persistent checkpoint storage ───────────────────
DRIVE_CKPT_DIR = "/content/drive/MyDrive/lewm_pusht_checkpoints"
drive_available = False
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    drive_available = True
    print(f"=== Google Drive mounted — checkpoints will be backed up to {DRIVE_CKPT_DIR} ===")
except Exception as e:
    print(f"=== Drive mount failed ({e}) — checkpoints will only be on VM disk ===")

# ── 4. Background checkpoint watcher — copies each .pt to Drive as written ───
LOCAL_CKPT_DIR = "/content/stable-wm/checkpoints/lewm"
_stop_watcher = threading.Event()

def checkpoint_watcher():
    known = set()
    while not _stop_watcher.is_set():
        if drive_available and os.path.exists(LOCAL_CKPT_DIR):
            for fname in os.listdir(LOCAL_CKPT_DIR):
                if fname.endswith(".pt") and fname not in known:
                    src = os.path.join(LOCAL_CKPT_DIR, fname)
                    dst = os.path.join(DRIVE_CKPT_DIR, fname)
                    try:
                        shutil.copy2(src, dst)
                        print(f"=== Drive backup: {fname} ===", flush=True)
                        known.add(fname)
                    except Exception as e:
                        print(f"Warning: Drive backup of {fname} failed: {e}", flush=True)
        _stop_watcher.wait(30)

watcher = threading.Thread(target=checkpoint_watcher, daemon=True)
watcher.start()

# ── 5. Download dataset from HuggingFace ──────────────────────────────────────
h5_path = "/content/stable-wm/datasets/pusht_expert_train.h5"
zst_path = h5_path + ".zst"
lance_path = "/content/stable-wm/datasets/pusht_expert_train.lance"
os.makedirs("/content/stable-wm/datasets", exist_ok=True)

if not os.path.exists(h5_path):
    print("=== Downloading dataset from HuggingFace (~12 GB) ===")
    from huggingface_hub import hf_hub_download
    tmp = hf_hub_download(
        repo_id="quentinll/lewm-pusht",
        filename="pusht_expert_train.h5.zst",
        repo_type="dataset",
        local_dir="/content/stable-wm/datasets",
    )
    print(f"Downloaded to: {tmp}")
    print("=== Installing zstd ===")
    subprocess.run(["apt-get", "install", "-y", "-q", "zstd"], check=True)
    print("=== Decompressing ===")
    subprocess.run(["zstd", "-d", zst_path, "-o", h5_path, "--rm"], check=True)
else:
    print("=== H5 already present ===")

size_gb = os.path.getsize(h5_path) / 1e9
print(f"H5 size: {size_gb:.1f} GB")
if size_gb < 30:
    raise SystemExit(f"H5 file too small ({size_gb:.1f} GB) — download may be incomplete")

# ── 6. Convert H5 → Lance ─────────────────────────────────────────────────────
if os.path.exists(lance_path):
    print("=== Lance dataset already present, skipping conversion ===")
else:
    print("=== Converting H5 → Lance (one-time, ~15-30 min) ===")
    subprocess.run(["pip", "install", "-q", "hdf5plugin"], check=True)
    import hdf5plugin
    import h5py
    import numpy as np
    from tqdm import tqdm
    from stable_worldmodel.data.format import get_format

    KEYS = ["pixels", "action", "proprio", "state"]

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens = f["ep_len"][:]
        n_eps = len(ep_lens)
        print(f"  {n_eps} episodes, {ep_offsets[-1] + ep_lens[-1]} total steps")

    LanceFormat = get_format("lance")

    def episodes():
        with h5py.File(h5_path, "r") as hf:
            for ep_idx in tqdm(range(n_eps), desc="converting"):
                offset = int(ep_offsets[ep_idx])
                length = int(ep_lens[ep_idx])
                yield {k: list(hf[k][offset:offset + length]) for k in KEYS}

    with LanceFormat.open_writer(lance_path, mode="overwrite") as writer:
        writer.write_episodes(episodes())

    print(f"=== Lance written to {lance_path} ===")

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("=== Starting training (max_epochs=10) ===")
env = os.environ.copy()
env["LOCAL_DATASET_DIR"] = "/content/stable-wm"
env["STABLEWM_HOME"] = "/content/stable-wm"
env["PYTHONUNBUFFERED"] = "1"

proc = subprocess.Popen(
    ["python", "-u", "train.py", "subdir=lewm_pusht", "loader.batch_size=128",
     "num_workers=4", "+loader.multiprocessing_context=spawn", "trainer.max_epochs=10"],
    cwd="/content/le-wm", env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for line in iter(proc.stdout.readline, b""):
    sys.stdout.write(line.decode(errors="replace"))
    sys.stdout.flush()
proc.wait()

_stop_watcher.set()

# ── 8. Final Drive sync — copy any remaining checkpoints ──────────────────────
if drive_available and os.path.exists(LOCAL_CKPT_DIR):
    print("=== Final Drive sync ===")
    for fname in sorted(os.listdir(LOCAL_CKPT_DIR)):
        if fname.endswith(".pt"):
            src = os.path.join(LOCAL_CKPT_DIR, fname)
            dst = os.path.join(DRIVE_CKPT_DIR, fname)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copy2(src, dst)
                print(f"  synced {fname}")
    print(f"=== All checkpoints on Drive at {DRIVE_CKPT_DIR} ===")

if proc.returncode != 0:
    raise SystemExit(f"train.py exited with code {proc.returncode}")

print("=== Training complete ===")
