---
name: cpu-vision-models
description: "Run vision models (monocular depth, x4 upscaling, person segmentation) on CPU without PyTorch or Hugging Face, using GitHub-release weights and onnxruntime. Use when a photo or video task would benefit from depth maps, super-resolution or person masks and the environment blocks huggingface.co / download.pytorch.org or has no GPU."
---

# CPU vision models without PyTorch / Hugging Face

Many agent sandboxes, CI runners and corporate proxies block huggingface.co,
download.pytorch.org and most model CDNs, but allow GitHub releases and PyPI.
These three models all ship weights on GitHub releases and run on CPU through
`onnxruntime`. The `cpuvision` package in this repo wraps them.

```bash
pip install numpy pillow onnxruntime onnx      # no torch
python -m cpuvision fetch                      # ~250 MB, resumable, sha256-checked
python -m cpuvision depth   in.jpg depth.png --color
python -m cpuvision mask    in.jpg mask.png    [--cutout]
python -m cpuvision upscale in.jpg big.png     [--blend 0.6]
```

## What runs where (2 CPU cores, warm)

| Task | Weights (GitHub release) | Time |
|---|---|---|
| Depth (2.5D parallax, depth fog, fake DoF) | MiDaS v2.1 small ONNX, 67 MB, 256x256 input | ~0.09 s per frame |
| Person mask | U^2-Net human-seg ONNX, 176 MB, 320x320 input | ~0.8 s per frame |
| x4 upscale | Real-ESRGAN realesr-general-x4v3 `.pth`, 5 MB → ONNX built locally | ~6.6 s for 464x688 |

## Things that matter

- **Reading `.pth` without torch:** the file is a zip with `data.pkl` plus raw
  storages under `data/<key>`. `cpuvision/torchless.py` unpickles it with a
  restricted unpickler (only `OrderedDict` and the tensor rebuild functions are
  allowed, so a malicious checkpoint can't run code), maps storages to numpy and
  rebuilds tensors with `as_strided`.
- **Building the ESRGAN graph:** SRVGGNetCompact is Conv/PReLU x33, then
  DepthToSpace(4, mode=CRD) for PixelShuffle, plus a nearest-upsampled copy of
  the input added back. Set `model.ir_version = 8`, or many onnxruntime builds
  refuse the model.
- **ESRGAN looks slightly waxy on its own**, so blend 60/40 with bicubic
  (`--blend 0.6`). Use it on stills and keyframes, not whole clips.
- **U^2-Net preprocessing as in rembg:** 320x320 LANCZOS, divide by max, ImageNet
  normalisation, min-max normalise output 0, resize back.
- **Video:** compute masks at ~6 fps and interpolate linearly while rendering.
  That is enough for soft silhouette effects and cuts the cost by 4-5x.
- **Long downloads under a call time limit:** `fetch()` resumes `.part` files
  with HTTP Range, so repeated short calls finish a large file.
- **Depth on CGI or flat renders is weak:** MiDaS small was trained on photos.
  Expect a smooth gradient on game screenshots and clean subject separation on
  real photos.
