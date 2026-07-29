"""Run crossover2 with torch.compile on, at the lengths that matter.

Uncompiled, the real JambaEncoder at expand=2 loses to a parameter-matched ViT by 2-3x at every
window from 784 to 2352 tokens, with no trend toward crossing. But every historical crossover
number was measured under torch.compile(mode="reduce-overhead"), and cudagraphs is exactly the
fix for Jamba's many small mamba kernel launches. If compiling recovers the win, the finding is
"train.py must compile the encoder". If it does not, the 1,960-token window is indefensible and
the architecture premise needs rethinking. Those are very different conclusions.
"""

import os
import runpy

os.environ["CROSS2_COMPILE"] = "1"
os.environ["CROSS2_BATCHES"] = "8,16"
os.environ["CROSS2_FRAMES"] = "8,10,12"

runpy.run_path("/content/crossover2.py", run_name="__main__")
