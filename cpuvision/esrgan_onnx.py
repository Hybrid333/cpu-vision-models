"""Build Real-ESRGAN's SRVGGNetCompact (realesr-general-x4v3) as ONNX from the
.pth weights - no PyTorch needed.

Architecture (from basicsr's SRVGGNetCompact):
    body.0          Conv 3->64, 3x3
    body.1          PReLU(64)
    body.2k/2k+1    (Conv 64->64 3x3, PReLU) x num_conv
    body.last       Conv 64 -> 3*scale^2, 3x3
    PixelShuffle(scale)  == DepthToSpace(mode=CRD)
    + nearest-neighbour upsample of the input (residual)
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .torchless import load_pth


def build(pth_path, onnx_path, scale=4):
    sd = load_pth(pth_path)
    sd = sd.get('params_ema', sd.get('params', sd))
    idx = sorted({int(k.split('.')[1]) for k in sd if k.startswith('body.')})

    nodes, inits = [], []
    x = 'input'
    for i in idx:
        w = sd[f'body.{i}.weight']
        if w.ndim == 4:                                   # conv
            b = sd[f'body.{i}.bias']
            inits += [numpy_helper.from_array(w.astype(np.float32), f'w{i}'),
                      numpy_helper.from_array(b.astype(np.float32), f'b{i}')]
            nodes.append(helper.make_node('Conv', [x, f'w{i}', f'b{i}'], [f'c{i}'],
                                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
            x = f'c{i}'
        else:                                             # PReLU, slope per channel
            inits.append(numpy_helper.from_array(
                w.astype(np.float32).reshape(-1, 1, 1), f'a{i}'))
            nodes.append(helper.make_node('PRelu', [x, f'a{i}'], [f'p{i}']))
            x = f'p{i}'

    nodes.append(helper.make_node('DepthToSpace', [x], ['ps'],
                                  blocksize=scale, mode='CRD'))
    inits.append(numpy_helper.from_array(
        np.array([1, 1, scale, scale], np.float32), 'scales'))
    nodes.append(helper.make_node('Resize', ['input', '', 'scales'], ['base'],
                                  mode='nearest'))
    nodes.append(helper.make_node('Add', ['ps', 'base'], ['output']))

    graph = helper.make_graph(
        nodes, 'srvgg_compact',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, None, None])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, None, None])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8          # newer IR versions are rejected by many onnxruntime builds
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    return onnx_path
