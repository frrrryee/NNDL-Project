import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import numpy as np
import torch
from PIL import Image
import config
from member3_structure_preserving.attention_map import (AttentionConfig,
                                                      ObjectnessAttention)
from member3_structure_preserving.line_art import (LineArtConfig,
                                                   attention_from_map)
from build_line_art_sets import render

am = ObjectnessAttention(AttentionConfig(), device='cuda')
lc = LineArtConfig()
rel, _ = config.read_split(config.SPLIT_FILES['photo_train'])[0]
im = np.asarray(Image.open(config.IMAGE_ROOT / rel).convert('RGB'))
t = torch.from_numpy(
    np.ascontiguousarray(im.astype(np.float32) / 255.)).permute(2, 0, 1)
t = t.unsqueeze(0).cuda()
att = attention_from_map(am.objectness(t), im.shape[0], im.shape[1],
                         lc.smooth_sigma)

p1, v1 = render(im, att, lc, 'line_plain_total')
p2, v2 = render(im, att, lc, 'line_plain_foreground')
print('identical:', np.array_equal(p1, p2))
print('unique grey p1:', np.unique(p1.reshape(-1, 3), axis=0)[:6])
print('unique grey p2:', np.unique(p2.reshape(-1, 3), axis=0)[:6])
diff = np.abs(p1.astype(int) - p2.astype(int))
print('max diff:', diff.max(), ' changed px:', int((diff > 0).any(axis=2).sum()))
print('v1 fg_w/bg_w:', v1.get('foreground_weight'), v1.get('background_weight'))
print('v2 fg_w/bg_w:', v2.get('foreground_weight'), v2.get('background_weight'))
