import subprocess
result = subprocess.run(
    ["find", "/content/stable-wm", "-name", "*.pt", "-o", "-name", "*.ckpt"],
    capture_output=True, text=True
)
print(result.stdout or "No .pt or .ckpt files found under /content/stable-wm")
result2 = subprocess.run(["find", "/root/.cache", "-name", "*.pt", "-o", "-name", "*.ckpt"],
    capture_output=True, text=True)
print(result2.stdout or "No .pt or .ckpt files found under /root/.cache")
