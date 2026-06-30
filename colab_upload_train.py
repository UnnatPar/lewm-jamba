import subprocess
import os
import sys
import glob

# ── 1. Install deps ───────────────────────────────────────────────────────────
print("=== Installing dependencies ===")
subprocess.run(["pip", "install", "-q", "stable-worldmodel[train]"], check=True)

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
if not os.path.exists("/content/le-wm"):
    print("=== Cloning le-wm ===")
    subprocess.run(
        ["git", "clone", "https://github.com/lucas-maes/le-wm.git", "/content/le-wm"],
        check=True,
    )
else:
    print("=== le-wm already cloned ===")

# ── 3. Extract dataset from uploaded tar ──────────────────────────────────────
tar_path = "/content/pusht_lance.tar"
lance_path = "/content/stable-wm/datasets/pusht_expert_train.lance"

# Reassemble from split parts if tar not already present
parts = sorted(glob.glob("/content/pusht_lance_part_*"))
if not os.path.exists(tar_path) and parts:
    print(f"=== Reassembling {len(parts)} parts into tar ===")
    with open(tar_path, "wb") as out:
        for part in parts:
            size_mb = os.path.getsize(part) / 1e6
            print(f"  cat {os.path.basename(part)} ({size_mb:.0f} MB)")
            with open(part, "rb") as f:
                while chunk := f.read(64 * 1024 * 1024):
                    out.write(chunk)
    print("=== Reassembly complete ===")
    for part in parts:
        os.remove(part)

if not os.path.exists(tar_path):
    raise SystemExit("No tar or parts found — upload pusht_lance_part_* files first")

print(f"=== Extracting {tar_path} ({os.path.getsize(tar_path)/1e9:.1f} GB) ===")
os.makedirs("/content/stable-wm/datasets", exist_ok=True)
subprocess.run(
    ["tar", "-xf", tar_path, "-C", "/content/stable-wm/datasets"],
    check=True,
)
os.remove(tar_path)  # free up space after extraction
print("=== Extraction complete ===")

# ── 4. Verify all shards present ──────────────────────────────────────────────
data_dir = os.path.join(lance_path, "data")
if not os.path.exists(data_dir):
    raise SystemExit(f"data dir missing after extraction: {data_dir}")

shards = [f for f in os.listdir(data_dir) if f.endswith(".lance")]
print(f"=== Found {len(shards)} lance shard(s) in {data_dir} ===")
for s in sorted(shards):
    size_gb = os.path.getsize(os.path.join(data_dir, s)) / 1e9
    print(f"  {s}  {size_gb:.2f} GB")

if len(shards) < 3:
    raise SystemExit(f"Expected 3 shards, got {len(shards)} — tar may be incomplete")

total_gb = sum(os.path.getsize(os.path.join(data_dir, s)) for s in shards) / 1e9
if total_gb < 12:
    raise SystemExit(f"Total shard size {total_gb:.1f} GB is too small — data corrupt")

print(f"=== Dataset verified: {total_gb:.1f} GB across {len(shards)} shards ===")

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("=== Starting training (max_epochs=9) ===")
env = os.environ.copy()
env["LOCAL_DATASET_DIR"] = "/content/stable-wm"
env["STABLEWM_HOME"] = "/content/stable-wm"
env["PYTHONUNBUFFERED"] = "1"

proc = subprocess.Popen(
    [
        "python", "-u", "train.py",
        "subdir=lewm_pusht",
        "loader.batch_size=128",
        "num_workers=4",
        "+loader.multiprocessing_context=spawn",
        "trainer.max_epochs=9",
    ],
    cwd="/content/le-wm",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

for line in iter(proc.stdout.readline, b""):
    sys.stdout.write(line.decode(errors="replace"))
    sys.stdout.flush()

proc.wait()
if proc.returncode != 0:
    raise SystemExit(f"train.py exited with code {proc.returncode}")
