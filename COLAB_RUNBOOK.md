# Running a real training job on Colab

`bin/colab-run.sh` in unnat-brain is for one-shot benchmarks: fresh session, run a script, tear
down. A training run is a different shape — it lasts hours, it must be pollable while it runs,
and its output must survive the runtime. This is that procedure.

Everything below was paid for once on 2026-07-30. None of it is guesswork.

## The shape

```
colab new -s <name> --gpu A100        # ~40 s
stage1_env      exec, foreground      # ~60 s   wheels + deps + repo
stage2_data     exec, DETACHED        # ~22 min  download -> h5 -> lance
stage3_smoke    exec, foreground      # ~10 min  short run, both encoders
stage4_train    exec, DETACHED        # hours    the real run
bash bin/colab-pull-ckpt.sh <name>    # on THIS machine, in parallel, forever
```

**Anything longer than a few minutes must be detached.** One kernel process serves every
`colab exec` call to a session, so a foreground training run makes the session unpollable —
no log tail, no `nvidia-smi`, no checkpoint listing. Launch with `subprocess.Popen` writing to
a logfile and return immediately; later `exec` calls tail that file.

`colab exec -f` ships the file's **text** to the kernel. There is no `argv` on the far side, so
scripts are parameterised by editing constants at the top and re-shipping. Uploads
(`colab upload`) are still needed for anything that must exist as a *file* on the VM.

## Disk is the binding constraint, not GPU memory

An A100 runtime has ~113 GB total with ~47 GB of base image, so ~64 GB free. The dataset needs

| artifact | size |
|---|---|
| `pusht_expert_train.h5.zst` (download) | 13.1 GB |
| `pusht_expert_train.h5` (decompressed) | 46.3 GB |
| `pusht_expert_train.lance` (converted) | ~12 GB |

and the h5 and the lance must coexist during conversion: **46 + 12 = 58 GB against 64 GB free.**
It fits, but only just, and the zst has to be removed by `zstd -d --rm` on the way through.

To buy headroom, evict base-image packages this run never imports (`clean.py` pattern):
RAPIDS, TensorFlow, JAX, pyspark, spacy, xgboost — worth ~6.6 GB. **Do not evict
`opencv-contrib-python`**: `stable_worldmodel` imports `cv2`, and because the contrib and
headless builds share the `cv2/` directory, uninstalling contrib deletes the files while leaving
headless's dist-info behind — so `pip install opencv-python-headless` then reports "already
satisfied" and installs nothing. The fix is `--force-reinstall --no-deps`.

Delete the h5 as soon as the lance is written; it reclaims 46 GB.

## Two config keys must be overridden per run, or two runs collide

`config/train/launcher/local.yaml` sets `wandb.config.name: ${output_model_name}` and
`wandb.config.id: ${subdir}` with `resume: allow`. Both default to the same value for every run,
so a second run **resumes into the first run's wandb history** instead of creating its own, and
`SaveCkptCallback` writes both runs' weights to the same filename. Always pass:

```
subdir=<tag> output_model_name=<tag>
```

## A runtime lives about 90 minutes, so size the run to that

First attempt at a 3.2 h run: at ~1.5 h in, the tunnel started returning **404 on
`/api/kernels/<id>`** and on `POST /api/kernels`. The backend was gone, not just the kernel, and
the CLI responded by wiping `~/.config/colab-cli/sessions.json` to `{}` — after which the runtime
is unaddressable by any `-s` name even though `colab sessions` still lists it as `[?]`. There is
no reattach command. 25 minutes of training went with it, because the first checkpoint had not
landed yet.

Two consequences, both design constraints rather than bad luck:

1. **The whole run — prep included — should fit in ~90 minutes.** 22 min of dataset prep plus
   ~65 min of training is about the practical ceiling. Cap the epoch with
   `+trainer.limit_train_batches=N` and say in the config comment that the cap is a budget
   decision, not a modelling one.
2. **A checkpoint that has not been pulled off the VM does not exist.** Make epochs short enough
   that one lands early — an epoch that takes longer than the mean time to session death
   produces nothing.

Cross-session resume is not a way out: the upload direction of the contents API is the
unreliable one (documented SSL EOF at 163 MB), so a 335 MB Lightning checkpoint cannot be put
back reliably. Pull the 112 MB `weights_epoch_N.pt` files, not the `.ckpt`s — three times less
tunnel traffic, and the optimizer state has nothing to be restored into.

## Checkpoints

`colab drivemount` blocks on an interactive OAuth prompt a headless session cannot answer, so
Google Drive is not available as the durable store. Instead the training wrapper copies every
stable Lightning `.ckpt` and every `weights_epoch_N.pt` into `/content/ckpt/`, and
`unnat-brain/bin/colab-pull-ckpt.sh` mirrors that directory to `C:\Users\Unnat\lewm-ckpt` from
this machine on a timer. It exits when the session dies, which is also how the session's death
gets noticed.

## Verify the config before provisioning a GPU

A **CPU** runtime (`colab new -s prep`, no `--gpu`) costs approximately nothing and catches
almost everything: `pip install stable-worldmodel[train]` (36 s), hydra composition,
`hydra.utils.instantiate(cfg.model)`, parameter counts, and the wandb block. Two real bugs were
caught there before the A100 was ever provisioned. Note that a bare `compose()` lacks the
`eval:` resolver that `train.py` gets from importing `stable_worldmodel` — import it first or
`data.dataset.num_steps` fails to resolve.
