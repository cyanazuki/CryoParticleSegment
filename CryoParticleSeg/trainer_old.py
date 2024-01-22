import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from lr_scheduler import Callback
from utils import tqdm_plugin_for_loader


def plot_image(image, mask=None):
  fig, ax = plt.subplots(figsize=(4, 4))

  ax.imshow(image, cmap='gray')
  if mask is not None:
    ax.imshow(mask, cmap='inferno', alpha=0.2)
  ax.set_axis_off()

  plt.tight_layout()
  plt.show()

def plot_result(image, output, mask=None):
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  ax = axes.flatten()

  ax[0].imshow(image, cmap='gray')
  if mask is not None:
    ax[0].imshow(mask, cmap='inferno', alpha=0.2)
  ax[0].set_axis_off()

  ax[1].imshow(output, cmap='inferno')
  ax[1].set_axis_off()

  plt.tight_layout()
  plt.show()

class ConfusionMatrix(object):
  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.mat = None

  def update(self, a, b):
    n = self.num_classes
    if self.mat is None:
      self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
    with torch.no_grad():
      k = (a >= 0) & (a < n)
      inds = n * a[k].to(torch.int64) + b[k]
      self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

  def reset(self):
    if self.mat is not None:
      self.mat.zero_()

  def compute(self):
    h = self.mat.float()
    acc_global = torch.diag(h).sum() / h.sum()
    acc = torch.diag(h) / h.sum(1)
    iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    return acc_global, acc, iou

  def reduce_from_all_processes(self):
    if not torch.distributed.is_available():
      return
    if not torch.distributed.is_initialized():
      return
    torch.distributed.barrier()
    torch.distributed.all_reduce(self.mat)

  def __str__(self):
    acc_global, acc, iou = self.compute()
    return (
      f"global correct: {acc_global.item() * 100:.2f}\n"
      f"average row correct: {['{:.1f}'.format(i) for i in (acc * 100).tolist()]}\n"
      f"IoU: {['{:.1f}'.format(i) for i in (iou * 100).tolist()]}\n"
      f"mean IoU: {iou.mean().item() * 100:.2f}")

class SimpleTrainer:
  def __init__(self, model, train_dataset, criterion, optimizer, device,
               val_dataset=None, metrics=['loss'], num_classes = 2):
    self.model = model.to(device)
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.criterion = criterion
    self.optimizer = optimizer
    self.metrics = metrics
    self.device = device
    self.best_epoch = 0
    self.num_classes = num_classes
    self.loss = list()
    self.best_loss = np.inf
    self.step_action = {}

  def train_per_epochs(self, loader):
    self.model.train()
    results = {metric: np.zeros((len(loader))) for metric in self.metrics}
    for batch_idx, (inputs, targets, *_) in enumerate(loader):
      inputs = inputs.to(self.device)
      targets = targets.to(self.device)
      # Training
      self.optimizer.zero_grad()
      outputs = self.model(inputs)['out']
      loss = self.criterion(outputs, targets)
      loss.backward()
      self.optimizer.step()
      # Evaluating
      results['loss'][batch_idx] = loss.item()
      if 'acc' in self.metrics:
        results['acc'][batch_idx] = self.get_accuracy(outputs, targets)
      for func_name in self.step_action:
        self.step_action[func_name](self, loader, results, batch_idx)
    gc.collect()
    return results

  def evaluate(self, loader):
    self.model.eval()
    confmat = ConfusionMatrix(self.num_classes)
    results = {metric: np.zeros(len(loader)) for metric in self.metrics}
    with torch.no_grad():
      for batch_idx, (inputs, targets, *_) in enumerate(loader):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)['out']
        # Evaluating
        results['loss'][batch_idx] = self.criterion(outputs, targets).item()
        if random.random() < 0.1:
          plot_result(image=inputs.cpu()[0,0],
                output=outputs.cpu().detach().numpy()[0,0],
                mask=targets.cpu()[0,0])
        confmat.update(targets.argmax(1).flatten(), outputs.argmax(1).flatten())
        for func_name in self.step_action:
          self.step_action[func_name](self, loader, results, batch_idx)
      confmat.reduce_from_all_processes()
      val_info = str(confmat)
      print(val_info)
    gc.collect()
    return results

  def train(self, num_epochs, batch_size=64, ckpt_dir=None, *, verbose=1):
    # Set Loader
    train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True)
    val_loader = None if self.val_dataset is None else \
      DataLoader(self.val_dataset, batch_size, shuffle=False)
    # Training
    self._best_state = self.model.state_dict()
    for epoch in range(num_epochs):
      if verbose:
        print(f"Epoch {epoch + 1:3d}/{num_epochs:3d}:")
      # Training Step
      train_results = self.train_per_epochs(train_loader)
      # Evaluating Step
      if val_loader is None:
        self.loss.append(loss := train_results['loss'].mean())
      else:
        val_results = self.evaluate(val_loader)
        self.loss.append(loss := val_results['loss'].mean())
      # History
      if hasattr(self, 'callback'):
        stop_training = self.callback(epoch, loss, ckpt_dir, verbose=verbose)
        if stop_training:
          break
      if loss < self.best_loss:
        self.best_loss = loss
        self.best_epoch = epoch
        self._best_state = self.model.state_dict()
        if verbose:
          print(f"Loss improve to {loss}.")
      gc.collect()
    self.model.load_state_dict(self._best_state)
    del(self._best_state)

  def predict(self, loader):
    self.model.eval()
    predictions = list()
    with torch.no_grad():
      for batch_idx, (inputs, *_) in enumerate(loader):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)['out']
        predictions.extend(self.get_predictions(outputs).numpy())
    return predictions

  def save_model(self, ckpt_file, *, verbose=0, training=False):
    if ckpt_file is not None:
      torch.save(self.model.state_dict(), ckpt_file)
      if verbose:
        print(f"Saving model at {ckpt_file}")
    elif not training:
      if verbose:
        print("'ckpt_file' should be given.")

  def get_predictions(self, outputs):
    preds = outputs.argmax(dim=-1)
    return preds.cpu().detach()

  def get_accuracy(self, outputs, targets):
    preds = outputs.argmax(dim=1)
    corrects = preds.eq(targets.argmax(dim=1))
    return corrects.sum().item() / targets.size(0)

class TrainerClassifier(SimpleTrainer, Callback):
  def __init__(self, model, train_dataset, val_dataset, criterion, optimizer,
               device, metrics=['loss'], num_classes = 2,
               lr_scheduler=None, patience=5):
    SimpleTrainer.__init__(self, model, train_dataset, criterion, optimizer,
                           device, val_dataset, metrics, num_classes)
    Callback.__init__(self, lr_scheduler, patience)


def tqdm_plugin_for_Trainer(trainer):
  """
  Present progress bar with tqdm for each epochs.

  Parameters
  ----------
  trainer : Trainer
    Class of a trainer.

  Returns
  -------
  TrainerWithTqdm : Trainer
    Class of a trainer.

  """
  class TrainerWithTqdm(trainer):
    def __init__(self, *arg, **kwarg):
      super().__init__(*arg, **kwarg)
  
    @tqdm_plugin_for_loader(desc="Training")
    def train_per_epochs(self, *arg, **kwarg):
      return super(TrainerWithTqdm, self).train_per_epochs(*arg, **kwarg)
  
    @tqdm_plugin_for_loader(desc="Validation")
    def evaluate(self, *arg, **kwarg):
      return super(TrainerWithTqdm, self).evaluate(*arg, **kwarg)

  return TrainerWithTqdm