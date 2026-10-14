"""Check the user's already-activated environment. Never install anything."""
import importlib.util
import sys

REQUIRED = ("torch","torchvision","timm","modelscope","safetensors","nibabel",
            "numpy","scipy","sklearn")


def main():
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing dependencies: "+", ".join(missing)+
            ". Activate your own conda environment and install requirements manually.")
    import torch
    print("[preflight] Python",sys.version.split()[0],
          "PyTorch",torch.__version__,
          "CUDA available",torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[preflight] GPU",torch.cuda.get_device_name(0))
    else:
        print("[preflight] WARNING: CUDA not available; inference will run on CPU.")


if __name__=="__main__":
    main()
