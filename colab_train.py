import subprocess, os, sys, re, threading, shutil, glob, time

# ── 1. Dependencies ───────────────────────────────────────────────────────────
print("=== Installing dependencies ===", flush=True)
subprocess.run(["pip", "install", "-q", "stable-worldmodel[train]"], check=True)

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
if not os.path.exists("/content/le-wm"):
    subprocess.run(["git", "clone", "https://github.com/lucas-maes/le-wm.git", "/content/le-wm"], check=True)

config_path = "/content/le-wm/config/train/data/pusht.yaml"
with open(config_path) as f:
    cfg = f.read()
cfg = re.sub(r"pusht_expert_train(?:\.lance)?(?:\.h5)?", "pusht_expert_train.lance", cfg)
with open(config_path, "w") as f:
    f.write(cfg)

# ── 3. Download + decompress ──────────────────────────────────────────────────
h5_path    = "/content/stable-wm/datasets/pusht_expert_train.h5"
zst_path   = h5_path + ".zst"
lance_path = "/content/stable-wm/datasets/pusht_expert_train.lance"
os.makedirs("/content/stable-wm/datasets", exist_ok=True)

if not os.path.exists(h5_path):
    print("=== Downloading dataset (~12 GB) ===", flush=True)
    dl = subprocess.Popen(
        ["wget", "--progress=dot:giga", "-O", zst_path,
         "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in iter(dl.stdout.readline, b""):
        sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
    dl.wait()
    if dl.returncode != 0:
        raise SystemExit("wget failed")
    subprocess.run(["apt-get", "install", "-y", "-q", "zstd"], check=True)
    subprocess.run(["zstd", "-d", zst_path, "-o", h5_path, "--rm"], check=True)

size_gb = os.path.getsize(h5_path) / 1e9
print(f"H5 size: {size_gb:.1f} GB", flush=True)
if size_gb < 30:
    raise SystemExit(f"H5 too small ({size_gb:.1f} GB) — download incomplete")

# ── 4. Convert H5 → Lance (full dataset, skip if already complete) ────────────
lance_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, files in os.walk(lance_path) for f in files
) if os.path.exists(lance_path) else 0

if lance_size < 1_000_000_000:  # less than 1 GB means partial (test run) or missing
    if os.path.exists(lance_path):
        print(f"=== Lance incomplete ({lance_size/1e9:.1f} GB), reconverting full dataset ===", flush=True)
        shutil.rmtree(lance_path)
    else:
        print("=== Converting H5 → Lance (~15-30 min) ===", flush=True)

    subprocess.run(["pip", "install", "-q", "hdf5plugin"], check=True)
    import hdf5plugin, h5py
    from tqdm import tqdm
    from stable_worldmodel.data.format import get_format

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]
        n_eps      = len(ep_lens)
    print(f"  {n_eps} episodes", flush=True)

    def episodes():
        with h5py.File(h5_path, "r") as hf:
            for i in tqdm(range(n_eps), desc="converting"):
                o, l = int(ep_offsets[i]), int(ep_lens[i])
                yield {k: list(hf[k][o:o+l]) for k in ["pixels","action","proprio","state"]}

    with get_format("lance").open_writer(lance_path, mode="overwrite") as w:
        w.write_episodes(episodes())
    print("=== Lance ready ===", flush=True)
else:
    print(f"=== Lance already complete ({lance_size/1e9:.1f} GB) ===", flush=True)

# Signal session is ready
with open("/content/test_ready.txt", "w") as f:
    f.write("ok\n")
print("=== SETUP DONE ===", flush=True)

# ── 5. Watcher: copy each epoch checkpoint to predictable path ────────────────
CKPT_OUT = "/content/stable-wm/checkpoints"
os.makedirs(CKPT_OUT, exist_ok=True)
_stop = threading.Event()
_seen = set()

def watcher():
    while not _stop.is_set():
        for path in glob.glob("/root/.cache/stable-pretraining/runs/*/*/*/checkpoints/epoch=*.ckpt"):
            if path in _seen:
                continue
            m = re.search(r"epoch=(\d+)", path)
            if not m:
                continue
            # wait for file to finish writing
            try:
                s1 = os.path.getsize(path)
                time.sleep(5)
                s2 = os.path.getsize(path)
                if s1 != s2 or s1 == 0:
                    continue
            except OSError:
                continue
            dest = os.path.join(CKPT_OUT, f"weights_epoch_{m.group(1)}.ckpt")
            shutil.copy2(path, dest)
            _seen.add(path)
            size_mb = os.path.getsize(dest) / 1e6
            print(f"=== CHECKPOINT READY: weights_epoch_{m.group(1)}.ckpt ({size_mb:.0f} MB) ===", flush=True)
        _stop.wait(30)

threading.Thread(target=watcher, daemon=True).start()

# ── 6. Train 10 epochs ────────────────────────────────────────────────────────
print("=== Starting training (10 epochs) ===", flush=True)
env = os.environ.copy()
env["LOCAL_DATASET_DIR"] = "/content/stable-wm"
env["STABLEWM_HOME"]     = "/content/stable-wm"
env["PYTHONUNBUFFERED"]  = "1"

proc = subprocess.Popen(
    ["python", "-u", "train.py",
     "subdir=lewm_pusht",
     "loader.batch_size=128",
     "num_workers=4",
     "+loader.multiprocessing_context=spawn",
     "trainer.max_epochs=10"],
    cwd="/content/le-wm", env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for line in iter(proc.stdout.readline, b""):
    sys.stdout.write(line.decode(errors="replace"))
    sys.stdout.flush()
proc.wait()
_stop.set()

if proc.returncode != 0:
    raise SystemExit(f"train.py failed with code {proc.returncode}")

print("=== TRAINING COMPLETE ===", flush=True)
ckpts = sorted(glob.glob(f"{CKPT_OUT}/weights_epoch_*.ckpt"))
print(f"=== Checkpoints saved: {[os.path.basename(c) for c in ckpts]} ===", flush=True)
