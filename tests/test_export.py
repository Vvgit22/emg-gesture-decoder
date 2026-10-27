"""Parity between torch and ONNX on a random-weights model — fast, no data."""

import numpy as np
import torch

from emgdec.export import OPSET, verify
from emgdec.models import TinyCNN


def test_onnx_parity_random_weights(tmp_path):
    torch.manual_seed(0)
    net = TinyCNN(12, 17)
    net.set_normalisation(torch.randn(12), torch.rand(12) + 0.5)
    net.eval()
    pt = tmp_path / "m.pt"
    torch.save(net.state_dict(), pt)
    onnx = tmp_path / "m.onnx"
    torch.onnx.export(
        net, (torch.zeros(1, 12, 256),), str(onnx),
        input_names=["emg"], output_names=["logits"], opset_version=OPSET,
        dynamic_axes={"emg": {0: "batch"}, "logits": {0: "batch"}})
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, (64, 12, 256)).astype(np.float32)
    res = verify(onnx, pt, x)
    assert res["max_logit_diff"] < 1e-4
    assert res["argmax_agreement"] == 1.0
    assert res["ok"]
