"""Make the README demo strip: original | depth | mask | cutout on a new background.

    python examples/demo_strip.py photo.jpg strip.jpg
"""
import sys

import numpy as np
from PIL import Image

import cpuvision as cv

src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert('RGB')
img.thumbnail((720, 960))
W, H = 360, round(360 * img.height / img.width)

d = cv.colorize(cv.depth(img))
m = cv.person_mask(img)
cut = cv.cutout(img, m)

y = np.linspace(0, 1, H)[:, None, None]
bg = (np.array([18, 58, 46]) * (1 - y) + np.array([201, 169, 110]) * y).repeat(W, 1)
bg = Image.fromarray(bg.astype(np.uint8)).convert('RGBA')
bg.alpha_composite(cut.resize((W, H)))

strip = Image.new('RGB', (W * 4, H))
for i, im in enumerate([img, d, m.convert('RGB'), bg.convert('RGB')]):
    strip.paste(im.resize((W, H)), (i * W, 0))
strip.save(dst, quality=85)
print('->', dst)
