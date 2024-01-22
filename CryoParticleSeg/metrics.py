import torch

class ConfusionMatrix():
  def __init__(self, num_classes, device):
    self.num_classes = num_classes
    self.confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

  def update(self, targets, outputs):
    n = self.num_classes
    with torch.no_grad():
      mask = (targets >= 0) & (targets < n)
      inds = n * targets[mask].to(torch.int64) + outputs[mask]
      self.confmat += torch.bincount(inds, minlength=n**2).reshape(n, n)

  def confusion_matrix(self, targets, outputs):
    n = self.num_classes
    with torch.no_grad():
      mask = (targets >= 0) & (targets < n)
      inds = n * targets[mask].to(torch.int64) + outputs[mask]
      return torch.bincount(inds, minlength=n**2).reshape(n, n)

  def reduce_from_all_processes(self):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      torch.distributed.barrier()
      torch.distributed.all_reduce(self.confmat)

  def get_accuracy(self):
    """Returns the global accuracy."""
    return self.confmat.diag().sum() / self.confmat.sum()

  def get_row_accuracy(self):
    """Returns the row accuracy."""
    return self.confmat.diag() / self.confmat.sum(1)

  def get_row_iou(self):
    """Returns the row IoU."""
    m = self.confmat.float()
    return m.diag() / (m.sum(1) + m.sum(0) - m.diag())

  def get_mean_iou(self):
    """Returns the mean IoU."""
    return self.get_row_iou().mean()

  def get_ap(self):
    """Returns the average precision (area under precision-recall curve)."""
    precision = self.confmat.diag().cumsum(-1) / self.confmat.sum(1).cumsum(-1)
    recall = self.confmat.diag().cumsum(-1) / self.confmat.sum()
    return torch.trapz(precision, recall)

  def __str__(self):
    acc_global = self.get_accuracy()
    acc = self.get_row_accuracy()
    iou = self.get_row_iou()
    return (
      f"global correct: {acc_global.item() * 100:.2f}\n"
      f"average row correct: {['{:.1f}'.format(i) for i in (acc * 100).tolist()]}\n"
      f"IoU: {['{:.1f}'.format(i) for i in (iou * 100).tolist()]}\n"
      f"mean IoU: {iou.mean().item() * 100:.2f}")