import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Callback:
  def __init__(self, lr_scheduler=None, patience=5, save_best_only=True, verbose=1):
    self.lr_scheduler = lr_scheduler
    self.patience = patience
    self.save_best_only = save_best_only
    self.verbose = verbose
    self.counter = 0

  def callback(self, epoch, loss, ckpt_dir, *, verbose=None):
    if ckpt_dir is not None:
      ckpt_file = os.path.join(ckpt_dir, f"checkpoint{epoch}.pt")
    if verbose is None:
      verbose = self.verbose
    self._lr_scheduler(loss)
    self._modelcheckpoint(loss, ckpt_file, verbose=verbose)
    stop_training = self._earlystop(loss, verbose=verbose)
    return stop_training

  def _modelcheckpoint(self, loss, ckpt_file, verbose=None):
    if self.save_best_only and loss >= self.best_loss:
      return None
    if verbose is None:
      verbose = self.verbose
    self.save_model(ckpt_file, training=True, verbose=verbose)

  def _lr_scheduler(self, loss):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step(loss)

      # Forgot why I wrote this:
      # if isinstance(self.lr_scheduler, ReduceLROnPlateau):
      #   self.lr_scheduler.step(loss)
      # else:
      #   self.lr_scheduler.step()

  def _earlystop(self, loss, verbose=1):
    if loss < self.best_loss:
      self.counter = 0
    else:
      self.counter += 1
      if self.counter < self.patience:
        if verbose:
          print(f"No improvement for {self.counter} epoch.")
      else:
        if verbose:
          print("Early stopping")
        self.counter = 0
        return True
    return False