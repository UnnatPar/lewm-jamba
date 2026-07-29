"""Is mamba_expand the whole reason Jamba lost?

Uncompiled AND compiled, at expand=2, the real JambaEncoder loses to a parameter-matched ViT by
2.5-3x at every window from 784 to 2352 tokens. Every historical crossover number -- including
the 1,723 that chose a 10-frame window -- was measured at expand=1, which halves the scan's
inner width.

This sweeps expand in {1, 2} side by side with the ViT re-matched to each, so the answer is
either "expand=2 is the cost and expand=1 restores the win" (a capacity-vs-speed decision) or
"expand is not the explanation and the window premise is wrong regardless".

Uncompiled, because that is what train.py runs, and because compiling helped the ViT more than
Jamba anyway (0.430 -> 0.381 at batch 8 / 10 frames).
"""

import os
import runpy

os.environ["CROSS2_COMPILE"] = "0"
os.environ["CROSS2_EXPANDS"] = "1,2"
os.environ["CROSS2_BATCHES"] = "8"
os.environ["CROSS2_FRAMES"] = "8,10,12"

runpy.run_path("/content/crossover2.py", run_name="__main__")
