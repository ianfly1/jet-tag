import torch
import pytest
from jet_tagging.models.cnn import JetCNN
from jet_tagging.models.encdec import EncDec
from jet_tagging.models.mlp import JetMLP

def test_jet_cnn_forward_shape():
    model = JetCNN()
    # Batch size 8, 1 channel, 25x25 image grid
    x = torch.randn(8, 1, 25, 25)
    out = model(x)
    assert out.shape == (8, 2)

def test_encdec_forward_shape():
    model = EncDec(input_size=90, output_size=2)
    # Batch size 8, 90 flattened particle features
    x = torch.randn(8, 90)
    reconstructed, logits = model(x)
    assert reconstructed.shape == (8, 90)
    assert logits.shape == (8, 2)

def test_jet_mlp_forward_shape():
    model = JetMLP(input_size=90, output_size=2)
    x = torch.randn(8, 90)
    out = model(x)
    assert out.shape == (8, 2)