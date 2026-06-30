# LeWM Training — Debugging Reference

This document captures every failure we hit getting `train.py` running end-to-end. If you're starting fresh on a new machine or picking up a broken session, read this first.

---

## MOST IMPORTANT: Debug for TIME, not just crashes

> **LEWM trained in a few hours according to the paper. If any step is taking unnaturally long, something is wrong — treat it as a bug even if there is no error message.**

Baseline expectations:
- rsync of a 6 GB lance file: 10–30 min on a decent connection
- WSL startup: < 2 min
- `spawn` worker initialization (4 workers): ~15–20 min total (one-time cost at startup)
- Training throughput: **several it/s** — anything below ~1 it/s is broken
- GPU utilization during training: should be near **100%**
- Full training run (10 epochs): a few hours total

Silent failure pattern to watch for: the log line `[atomic_save] installed crash-safe checkpoint plugin` repeating every ~3 minutes with CPU near 0% and GPU at 0% — this means workers are deadlocked and no training is happening.

**If training looks like it's running but GPU is 0% or it/s < 0.5 for more than 5 minutes after the first batch — kill it and investigate.**

---

## Correct Launch Command

```bash
wsl -e bash -c "cd /mnt/c/Users/Unnat/le-wm && source .venv-linux/bin/activate && LOCAL_DATASET_DIR=/home/unnat/.stable-wm PYTHONUNBUFFERED=1 python -u train.py num_workers=4 +loader.multiprocessing_context=spawn 2>&1"
```

Key flags explained:
- `LOCAL_DATASET_DIR=/home/unnat/.stable-wm` — points to the `.stable-wm` **root**, not the `datasets` subdirectory (see failure #3 below)
- `num_workers=4` — overrides the default 6; tune to your CPU
- `+loader.multiprocessing_context=spawn` — required for Lance/PyArrow safety (see failure #5 below); needs `+` prefix due to Hydra struct config
- `PYTHONUNBUFFERED=1 python -u` — ensures output is not buffered so you see logs in real time

Check GPU health during training:
```bash
wsl -e bash -c "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

---

## Failure #1: WSL ext4 filesystem goes read-only

**Symptom:** rsync aborts mid-transfer with:
```
rsync: [receiver] write failed on '...': Read-only file system (30)
... Bus error
```
Or any WSL write operation fails with "Read-only file system".

**Root cause:** WSL2 ext4 virtual disk (`.vhdx`) silently goes read-only when the **Windows host C: drive runs out of space**. WSL cannot extend the virtual disk if there is no room on the host.

**Fix:**
1. Check Windows free space: `Get-PSDrive C` in PowerShell — need at least 20 GB headroom before any large transfer
2. Free space by deleting large files (see what's safe below)
3. Remount WSL: `wsl --shutdown` then reopen terminal (may need to wait 1–2 min)
4. If WSL returns `E_UNEXPECTED` on restart — wait longer (up to 5 min) and retry; the WSL service recovers once Windows has space

**What we deleted (75 GB freed):**
- Windows `C:\Users\Unnat\.stable-wm\datasets\pusht_expert_train.h5.zst` (12 GB compressed) — safe, we had the decompressed version
- Windows `.h5` raw file (43 GB) — safe only after confirming the WSL copy (`/home/unnat/.stable-wm/`) was complete and same size
- Windows `.venv` (old non-WSL venv, 6 GB) — safe, training uses `.venv-linux` in WSL
- WSL `/home/unnat/.stable-wm/pusht_expert_train.h5.zst` (13 GB) — safe, same logic

**Before deleting large files:** always verify the copy you're keeping is intact and the same size (`ls -lh` on both sides).

**Note on Windows long paths:** PowerShell `Remove-Item -Recurse` can fail on deeply nested paths (like `.venv`). Use this instead:
```powershell
cmd.exe /c "rd /s /q C:\path\to\folder"
```

---

## Failure #2: WSL service crashes with E_UNEXPECTED

**Symptom:** After `wsl --shutdown`, WSL refuses to restart:
```
[process exited with code 4294967295 (0xffffffff)]
Catastrophic failure
```

**Root cause:** WSL service was mid-recovery from the disk-full event when shutdown was issued too quickly.

**Fix:** Wait 3–5 minutes and try again. The WSL service recovers on its own once Windows has adequate disk space. Do not force kill WSL processes or delete the `.vhdx`.

---

## Failure #3: FileNotFoundError — cannot resolve dataset

**Symptom:**
```
FileNotFoundError: Cannot resolve 'pusht_expert_train.lance'
```

**Root cause:** `LOCAL_DATASET_DIR` not set, or set to the wrong path.

In `train.py` line 56: `cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)`

This is passed to `swm.data.load_dataset()`, which internally calls `get_cache_dir(override_root, sub_folder='datasets')`. That function appends `datasets` to whatever path you give it. So:

- If `LOCAL_DATASET_DIR=/home/unnat/.stable-wm` → looks in `/home/unnat/.stable-wm/datasets/pusht_expert_train.lance` ✓
- If `LOCAL_DATASET_DIR=/home/unnat/.stable-wm/datasets` → looks in `/home/unnat/.stable-wm/datasets/datasets/pusht_expert_train.lance` ✗

**Fix:** Always set `LOCAL_DATASET_DIR` to the **root** (`.stable-wm`), not the `datasets` subfolder.

---

## Failure #4: Lance dataset is incomplete (rsync cut off mid-transfer)

**Symptom:**
```
ArrowInvalid: Not found: .../pusht_expert_train.lance/data/1000010110110000111100002dfcf04cd69e1b9f75da0efb5d.lance
```
Training starts, loads a few batches, then crashes with a missing data file error.

**Root cause:** rsync was interrupted (by disk filling or network drop) and only copied 2 of 3 data shards inside the lance directory. The manifest pointed to a file that didn't exist in WSL.

**How to check completeness:**
```bash
# Count files in both locations
ls -la /home/unnat/.stable-wm/datasets/pusht_expert_train.lance/data/ | wc -l
ls -la /mnt/c/Users/Unnat/.stable-wm/datasets/pusht_expert_train.lance/data/ | wc -l
# Compare total sizes
du -sh /home/unnat/.stable-wm/datasets/pusht_expert_train.lance/
du -sh /mnt/c/Users/Unnat/.stable-wm/datasets/pusht_expert_train.lance/
```

**Fix:** Copy the missing file:
```bash
cp /mnt/c/Users/Unnat/.stable-wm/datasets/pusht_expert_train.lance/data/<missing_file>.lance \
   /home/unnat/.stable-wm/datasets/pusht_expert_train.lance/data/
```
Or re-run rsync with `--checksum` to only transfer what's different:
```bash
rsync -av --checksum /mnt/c/Users/Unnat/.stable-wm/datasets/pusht_expert_train.lance/ \
    /home/unnat/.stable-wm/datasets/pusht_expert_train.lance/
```

---

## Failure #5: DataLoader workers silently deadlock (Lance fork-safety)

**Symptom:** Training appears to start (Lightning logs appear, sanity val check passes), but then:
- GPU stays at 0%
- CPU hovers near 0%
- `[atomic_save] installed crash-safe checkpoint plugin` log line repeats every ~3 min
- `it/s` drops to ~0.1 or shows nothing
- No actual training steps appear in output

**Root cause:** Lance/PyArrow is **not fork-safe**. The default DataLoader multiprocessing mode on Linux is `fork`, which forks the parent process after Lance has already opened file readers. The forked child workers inherit corrupted file handles and deadlock silently — no Python exception is raised.

**Fix:** Use `spawn` multiprocessing context, which starts a fresh Python interpreter per worker instead of forking:
```bash
python train.py +loader.multiprocessing_context=spawn
```

The `+` prefix is required because `multiprocessing_context` is not a pre-declared key in the Hydra struct config (`config/train/lewm.yaml`). Without `+`, Hydra raises:
```
Key 'multiprocessing_context' is not in struct
```

**Alternative if spawn causes issues:** `forkserver` is another fork-safe option, though less tested here.

---

## Failure #6: prefetch_factor error when num_workers=0

**Symptom:**
```
ValueError: prefetch_factor option could only be specified in multiprocessing.
let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.
```

**Root cause:** `config/train/lewm.yaml` sets `prefetch_factor: 3`, but this is only valid when `num_workers > 0`. If you test with `num_workers=0` to bypass the worker deadlock, `prefetch_factor` also needs to be unset.

**Fix:** Pass both overrides together:
```bash
python train.py num_workers=0 loader.prefetch_factor=null
```

Note: `num_workers=0` runs data loading in the main process and completely avoids the Lance fork issue — but it's too slow for real training (~0.1 it/s, ~39 hrs/epoch with this dataset). Use it only for quick smoke tests.

---

## Failure #7: spawn workers re-initializing every epoch (persistent_workers=False)

**Symptom:** Training starts, first epoch runs fine, but between epochs there is a 3–4 minute pause per worker (4 workers = up to 16 min) where each worker reimports all of PyTorch, re-opens the dataset, etc. Training effectively stalls between epochs.

**Root cause:** With `spawn` context, each worker starts a cold Python interpreter — this takes ~3–4 min per worker. If `persistent_workers=False`, workers are torn down and recreated every epoch.

**Fix:** The config default is `persistent_workers: True` (in `lewm.yaml` line 28). Do **not** override it to `False`. With `persistent_workers=True`, workers initialize once at the start and stay alive for all 100 epochs — the ~16 min startup cost is paid once.

---

## Failure #8: CUDA out of memory at first training step

**Symptom:** Sanity val check passes fine, but the very first training step crashes:
```
RuntimeError: CUDA error: out of memory
```
Traceback points to `ViT.forward → MLP → activation_fn`. GPU jumps to ~3900 MiB / 4096 MiB right before crash.

**Root cause:** The paper's default `batch_size=128` was trained on a much larger GPU (likely A100 40 GB or V100 32 GB). The RTX A1000 Laptop GPU has only **4 GB VRAM**. At batch_size=128 with 224×224 images, the ViT-tiny forward + backward pass + predictor + projectors fills all 4 GB.

**Fix:** Reduce batch size via Hydra CLI override — no code change needed:
```bash
python train.py loader.batch_size=64 num_workers=2 +loader.multiprocessing_context=spawn
```
If 64 still OOMs, try 32. The learning rate may need adjustment too (linear scaling rule: halving batch size → halve LR), but start with just the batch size and see if loss still converges.

**Note on spawn worker memory:** With `spawn` context + `persistent_workers=True`, each worker holds a full copy of the dataset and PyTorch imports (~1.5 GB each). On a 16 GB system:
- `num_workers=2` → 4 workers total (2 train + 2 val) × 1.5 GB = 6 GB for workers
- Main process grows to ~5–9 GB during step estimation (loading data through workers)
- `num_workers=4` → 8 workers × 1.5 GB = 12 GB + main = likely OOM on 16 GB
- Safe setting on 16 GB RAM: `num_workers=2`

---

## Config Reference

Key settings in `config/train/lewm.yaml`:
```yaml
trainer:
  max_epochs: 100
  accelerator: gpu
  precision: bf16
  gradient_clip_val: 1.0

loader:
  batch_size: 128         # TOO LARGE for 4 GB GPU — override to 64 on CLI
  num_workers: 6          # override with num_workers=2 on the CLI for 16 GB RAM safety
  persistent_workers: True  # CRITICAL with spawn — do not override to False
  prefetch_factor: 3        # must be null if num_workers=0
  pin_memory: True
```

Dataset config in `config/train/data/pusht.yaml`:
```yaml
dataset:
  name: pusht_expert_train.lance
  keys_to_load: [pixels, action, proprio, state]
```

---

## Environment Checklist (new machine / fresh session)

Before launching training, verify:

1. **Windows C: drive free space** — need ≥ 20 GB headroom (lance dataset is ~18 GB)
2. **WSL is up** — `wsl -e bash -c "echo ok"`
3. **Dataset is complete in WSL** — all data shards present in `/home/<user>/.stable-wm/datasets/pusht_expert_train.lance/data/`
4. **Venv is activated** — `.venv-linux/bin/activate` inside the repo root (this is a WSL-native venv, not the Windows one)
5. **GPU is visible from WSL** — `wsl -e bash -c "nvidia-smi"` should show the GPU
6. **`LOCAL_DATASET_DIR` points to `.stable-wm` root** — NOT the `datasets` subdirectory
7. **Launch with `+loader.multiprocessing_context=spawn`** — every single run, not optional

---

## Diagnostic Commands

```bash
# Check GPU during training
wsl -e bash -c "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# Check training processes are alive
wsl -e bash -c "ps aux | grep python | grep -v grep"

# Check WSL disk usage
wsl -e bash -c "df -h /home"

# Check Windows C: free space (PowerShell)
(Get-PSDrive C).Free / 1GB

# Verify lance dataset completeness
wsl -e bash -c "ls /home/unnat/.stable-wm/datasets/pusht_expert_train.lance/data/ | wc -l"
```
