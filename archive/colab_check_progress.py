import os, subprocess

# H5 download progress
zst = "/content/stable-wm/datasets/pusht_expert_train.h5.zst"
h5  = "/content/stable-wm/datasets/pusht_expert_train.h5"
if os.path.exists(h5):
    print(f"H5: {os.path.getsize(h5)/1e9:.1f} GB (complete)")
elif os.path.exists(zst):
    print(f"zst: {os.path.getsize(zst)/1e9:.2f} / 12.2 GB")

# Lance conversion progress
lance = "/content/stable-wm/datasets/pusht_expert_train.lance"
if os.path.exists(lance):
    r = subprocess.run(["du", "-sh", lance], capture_output=True, text=True)
    print(f"Lance: {r.stdout.strip()}")
else:
    print("Lance: not started")

# Checkpoints
ckpt_dir = "/content/stable-wm/checkpoints"
if os.path.exists(ckpt_dir):
    files = os.listdir(ckpt_dir)
    print(f"Checkpoints: {files}")
else:
    print("Checkpoints: none yet")
