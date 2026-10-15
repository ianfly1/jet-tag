import torch
import pytest
from jet_tagging.utils.metrics import BackgroundRejectionAtEfficiency

def test_background_rejection_computation():
    metric = BackgroundRejectionAtEfficiency(target_eff=0.50)
    
    # Fake logits and ground truth binary targets
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    targets = torch.tensor([1, 0, 1, 0])
    
    metric.update(preds, targets)
    score = metric.compute()
    
    assert isinstance(score, torch.Tensor)
    assert score.item() > 0.0