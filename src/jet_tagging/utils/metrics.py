import torch
import torchmetrics
from torchmetrics import Metric

class BackgroundRejectionAtEfficiency(Metric):
    """Calculates Background Rejection (1 / False Positive Rate) at target Signal Efficiency (True Positive Rate)."""
    def __init__(self, target_eff: float = 0.50, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.target_eff = target_eff
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self.preds).cpu()
        target = torch.cat(self.target).cpu()
        
        scores = torch.softmax(preds, dim=-1)[:, 1] if preds.ndim > 1 else preds
        
        signal_mask = (target == 1)
        bkg_mask = (target == 0)
        
        signal_scores = scores[signal_mask]
        bkg_scores = scores[bkg_mask]

        if len(signal_scores) == 0 or len(bkg_scores) == 0:
            return torch.tensor(0.0)

        threshold = torch.quantile(signal_scores, 1.0 - self.target_eff)
        false_positives = (bkg_scores >= threshold).sum().float()
        total_bkg = bkg_scores.numel()

        fpr = false_positives / total_bkg
        return 1.0 / (fpr + 1e-7)