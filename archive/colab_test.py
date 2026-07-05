import subprocess, os, sys, re

print("=== Installing dependencies ===", flush=True)
subprocess.run(["pip", "install", "-q", "stable-worldmodel[train]"], check=True)

if not os.path.exists("/content/le-wm"):
    subprocess.run(["git", "clone", "https://github.com/lucas-maes/le-wm.git", "/content/le-wm"], check=True)

config_path = "/content/le-wm/config/train/data/pusht.yaml"
with open(config_path) as f:
    cfg = f.read()
cfg = re.sub(r"pusht_expert_train(?:\.lance)?(?:\.h5)?", "pusht_expert_train.lance", cfg)
with open(config_path, "w") as f:
    f.write(cfg)

# ── Download + decompress ─────────────────────────────────────────────────────
h5_path    = "/content/stable-wm/datasets/pusht_expert_train.h5"
zst_path   = h5_path + ".zst"
lance_path = "/content/stable-wm/datasets/pusht_expert_train.lance"
os.makedirs("/content/stable-wm/datasets", exist_ok=True)

if not os.path.exists(h5_path):
    print("=== Downloading dataset (~12 GB) ===", flush=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="quentinll/lewm-pusht", filename="pusht_expert_train.h5.zst",
                    repo_type="dataset", local_dir="/content/stable-wm/datasets")
    subprocess.run(["apt-get", "install", "-y", "-q", "zstd"], check=True)
    subprocess.run(["zstd", "-d", zst_path, "-o", h5_path, "--rm"], check=True)

size_gb = os.path.getsize(h5_path) / 1e9
print(f"H5 size: {size_gb:.1f} GB", flush=True)
if size_gb < 30:
    raise SystemExit(f"H5 too small ({size_gb:.1f} GB) — download incomplete")

# ── Convert first 10 episodes to Lance ───────────────────────────────────────
if not os.path.exists(lance_path):
    print("=== Converting first 10 episodes to Lance ===", flush=True)
    subprocess.run(["pip", "install", "-q", "hdf5plugin"], check=True)
    import hdf5plugin, h5py
    from stable_worldmodel.data.format import get_format

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]

    def episodes():
        with h5py.File(h5_path, "r") as hf:
            for i in range(10):
                o, l = int(ep_offsets[i]), int(ep_lens[i])
                yield {k: list(hf[k][o:o+l]) for k in ["pixels","action","proprio","state"]}

    with get_format("lance").open_writer(lance_path, mode="overwrite") as w:
        w.write_episodes(episodes())
    print("=== Lance ready (10 episodes) ===", flush=True)

# Signal that setup is done so download_ckpts.sh can verify the connection
with open("/content/test_ready.txt", "w") as f:
    f.write("ok\n")
print("=== SETUP DONE — terminal 2 can now verify download ===", flush=True)

# ── Train 1 epoch ─────────────────────────────────────────────────────────────
env = os.environ.copy()
env["LOCAL_DATASET_DIR"] = "/content/stable-wm"
env["STABLEWM_HOME"]     = "/content/stable-wm"
env["PYTHONUNBUFFERED"]  = "1"

proc = subprocess.Popen(
    ["python", "-u", "train.py",
     "subdir=test",
     "loader.batch_size=4",
     "num_workers=0",
     "loader.prefetch_factor=null",
     "loader.persistent_workers=false",
     "trainer.max_epochs=1"],
    cwd="/content/le-wm", env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for line in iter(proc.stdout.readline, b""):
    sys.stdout.write(line.decode(errors="replace"))
    sys.stdout.flush()
proc.wait()

if proc.returncode != 0:
    raise SystemExit(f"train.py failed with code {proc.returncode}")

# Copy Lightning checkpoint to predictable path for download
import glob, shutil
os.makedirs("/content/stable-wm/checkpoints", exist_ok=True)
ckpts = [p for p in glob.glob("/root/.cache/stable-pretraining/runs/*/*/*/checkpoints/epoch=*.ckpt")
         if "last" not in p]
for path in sorted(ckpts):
    m = re.search(r"epoch=(\d+)", path)
    if m:
        dest = f"/content/stable-wm/checkpoints/weights_epoch_{m.group(1)}.ckpt"
        shutil.copy2(path, dest)
        print(f"=== CHECKPOINT READY: {dest} ===", flush=True)

print("=== TEST COMPLETE ===", flush=True)
