"""Model registry and downloader. Every weight file comes from a GitHub
release, so this works where huggingface.co and download.pytorch.org are
blocked (corporate proxies, CI runners, agent sandboxes)."""
import hashlib
import os
import sys
import urllib.request

MODELS = {
    'midas_small': {
        'url': 'https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx',
        'file': 'model-small.onnx', 'bytes': 66764249,
        'sha256': '2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58',
        'license': 'MIT (isl-org/MiDaS)',
    },
    'realesr_general_x4v3': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'file': 'realesr-general-x4v3.pth', 'bytes': 4885111,
        'sha256': '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
        'license': 'BSD-3-Clause (xinntao/Real-ESRGAN)',
    },
    'u2net_human_seg': {
        'url': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx',
        'file': 'u2net_human_seg.onnx', 'bytes': 175997641,
        'sha256': '01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c',
        'license': 'Apache-2.0 (xuebinqin/U-2-Net)',
    },
}

DEFAULT_DIR = os.environ.get('CPUVISION_HOME',
                             os.path.join(os.path.expanduser('~'), '.cache', 'cpuvision'))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, folder=DEFAULT_DIR):
    """Download (resuming partial files) and verify one model; return its path."""
    m = MODELS[name]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, m['file'])
    if os.path.exists(path) and os.path.getsize(path) == m['bytes']:
        return path
    part = path + '.part'
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(m['url'], headers={'Range': f'bytes={have}-'} if have else {})
    with urllib.request.urlopen(req) as r, open(part, 'ab' if have else 'wb') as out:
        if have and r.status != 206:          # server ignored the range: start over
            out.seek(0); out.truncate()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            sys.stderr.write(f'\r{name}: {out.tell() / 1e6:6.1f} / {m["bytes"] / 1e6:.1f} MB')
    sys.stderr.write('\n')
    if _sha256(part) != m['sha256']:
        raise RuntimeError(f'{name}: checksum mismatch - delete {part} and retry')
    os.replace(part, path)
    return path


def session(path, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
