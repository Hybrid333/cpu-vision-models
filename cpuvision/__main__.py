"""Command line:
    python -m cpuvision depth   in.jpg out.png [--color]
    python -m cpuvision mask    in.jpg out.png [--cutout]
    python -m cpuvision upscale in.jpg out.png [--blend 0.6]
    python -m cpuvision fetch   [all|midas_small|u2net_human_seg|realesr_general_x4v3]
"""
import argparse
import time

from PIL import Image

from . import colorize, cutout, depth, person_mask, upscale
from .models import MODELS, fetch


def main():
    ap = argparse.ArgumentParser(prog='cpuvision')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('depth', 'mask', 'upscale'):
        p = sub.add_parser(name)
        p.add_argument('src'); p.add_argument('dst')
    sub.choices['depth'].add_argument('--color', action='store_true')
    sub.choices['mask'].add_argument('--cutout', action='store_true')
    sub.choices['upscale'].add_argument('--blend', type=float, default=0.6)
    f = sub.add_parser('fetch'); f.add_argument('which', nargs='?', default='all')
    a = ap.parse_args()

    if a.cmd == 'fetch':
        for n in (MODELS if a.which == 'all' else [a.which]):
            print(n, '->', fetch(n))
        return
    img = Image.open(a.src)
    t = time.time()
    if a.cmd == 'depth':
        d = depth(img)
        out = colorize(d) if a.color else Image.fromarray((d * 255).astype('uint8'), 'L')
    elif a.cmd == 'mask':
        m = person_mask(img)
        out = cutout(img, m) if a.cutout else m
    else:
        out = upscale(img, blend=a.blend)
    out.save(a.dst)
    print(f'{a.cmd}: {img.size[0]}x{img.size[1]} -> {out.size[0]}x{out.size[1]} '
          f'in {time.time() - t:.2f}s -> {a.dst}')


if __name__ == '__main__':
    main()
