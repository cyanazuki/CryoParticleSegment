"""Trainer."""
from collections import OrderedDict
import gc
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import reconstruct_patched
from metrics import ConfusionMatrix
from lr_scheduler import Callback
from utils import tqdm_plugin_for_loader
from plot import plot_result

__all__ = ["Evaluator", "Trainer", "TrainerWithScheduler", "SemanticSegmentationTrainer",
           "CryoEMEvaluator", "CryoEMTrainer", "CryoEMTrainerWithScheduler", "tqdm_plugin_for_Trainer"]

class Evaluator():
  def __init__(self, model, device, metrics, num_classes: int = 2):
    self.model = model.to(device)
    self.device = device
    self.metrics = metrics
    self.num_classes = num_classes
    self.step_action = {}

  def evaluate(self, loader, end_string: str=None):
    self.model.eval()
    self.initialize_evaluate(num_of_step=len(loader))
    with torch.no_grad():
      for batch_idx, (inputs, targets, *_) in enumerate(loader):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)['out']
        # Evaluating
        self.step_evaluate(inputs, outputs, targets, batch_idx)
    results = self.end_evaluate(end_string)
    for func_name in self.step_action:
      self.step_action[func_name](self, loader, results, batch_idx)
    gc.collect()
    torch.cuda.empty_cache()
    return results

  def initialize_evaluate(self, num_of_step: int):
    raise NotImplementedError
    self._results = OrderedDict()

  def step_evaluate(self, inputs, outputs, targets, batch_idx):
    raise NotImplementedError

  def end_evaluate(self, end_string: str=None) -> OrderedDict:
    if end_string is not None:
      print(end_string)
    return self._results

  def predict(self, loader):
    self.model.eval()
    predictions = list()
    with torch.no_grad():
      for batch_idx, (inputs, *_) in enumerate(loader):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)['out']
        predictions.extend(self.get_predictions(outputs).numpy())
    return predictions

  def get_predictions(self, outputs):
    preds = outputs.argmax(dim=1)
    return preds.cpu().detach()


class Trainer(Evaluator):
  """
  This trainer class is implemented only for four usages:
    Training, evaluation, prediction, and saving model.
  """
  def __init__(self, model, train_dataset, criterion, optimizer, device,
               metrics=['loss'], val_metrics=None, num_classes = 2):
    self.model = model.to(device)
    self.train_dataset = train_dataset
    self.criterion = criterion
    self.optimizer = optimizer
    self.metrics = metrics
    self.val_metrics = metrics if val_metrics is None else val_metrics
    self.device = device
    self.best_epoch = 0
    self.num_classes = num_classes
    self.train_loss = list()
    self.loss = list()
    self.best_loss = np.inf
    self.step_action = {}

  def train_per_epochs(self, loader):
    self.model.train()
    results = OrderedDict(loss=np.zeros((len(loader))))
    if 'acc' in self.metrics:
      results['acc'] = np.zeros((len(loader)))
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
    torch.cuda.empty_cache()
    return results

  def train(self, num_epochs, val_loader=None, batch_size=64, ckpt_dir=None, random_state=None, *, verbose=1):
    # Set Loader
    gen = torch.Generator()
    gen.manual_seed(random_state)
    train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, generator=gen, pin_memory=True)
    if val_loader is None:
      val_loader = DataLoader(self.train_dataset, batch_size, shuffle=False, pin_memory=True)

    # Training
    self._best_state = self.model.state_dict()
    for epoch in range(num_epochs):
      if verbose:
        print(f"Epoch {epoch + 1:3d}/{num_epochs:3d}:")
      # Training Step
      train_results = self.train_per_epochs(train_loader)
      print("Training score:")
      for metric in self.metrics:
        print(f"  {metric}\t: {train_results[metric].mean():.4f}")
      self.train_loss.append(train_results['loss'].mean())
      # Evaluating Step
      val_results = self.evaluate(val_loader, end_string="Validation score:")
      self.loss.append(loss := val_results['loss'].mean())
      # History
      if hasattr(self, 'callback'):
        stop_training = self.callback(epoch+1, loss, ckpt_dir, verbose=verbose)
        if stop_training:
          break
      # Save best result
      if loss < self.best_loss:
        self.best_loss = loss
        self.best_epoch = epoch
        self._best_state = self.model.state_dict()
        if verbose:
          print(f"Loss improve to {loss}.")
    self.model.load_state_dict(self._best_state)
    del(self._best_state)

  def save_model(self, ckpt_file, *, verbose=0, training=False):
    if ckpt_file is not None:
      torch.save(self.model.state_dict(), ckpt_file)
      if verbose:
        print(f"Saving model at {ckpt_file}")
    elif not training:
      if verbose:
        print("'ckpt_file' should be given.")

  def initialize_evaluate(self, num_of_step: int):
    self._results = OrderedDict()
    self._results['loss'] = np.zeros(num_of_step)

  def step_evaluate(self, inputs, outputs, targets, batch_idx):
    self._results['loss'][batch_idx] = self.criterion(outputs, targets).item()

  def end_evaluate(self, end_string: str=None) -> OrderedDict:
    results = super(Trainer, self).end_evaluate(end_string)
    print(f"  loss : {self._results['loss'].mean():.4f}")
    return results

  def get_accuracy(self, outputs, targets):
    preds = outputs.argmax(dim=1)
    corrects = preds.eq(targets.argmax(dim=1))
    return corrects.sum().item() / targets.size(0)


class TrainerWithScheduler(Trainer, Callback):
  def __init__(self, model, train_dataset, val_dataset, criterion, optimizer,
               device, metrics=['loss'], val_metrics=['loss'], num_classes = 2,
               lr_scheduler=None, patience=5):
    Trainer.__init__(self, model, train_dataset, criterion, optimizer,
                           device, val_dataset, metrics, val_metrics, num_classes)
    Callback.__init__(self, lr_scheduler, patience)


class SemanticSegmentationTrainer(Trainer):
  def __init__(self, *arg, **kwarg):
    super().__init__(*arg, **kwarg)
    self.plot = False

  def initialize_evaluate(self, num_of_step):
    super(SemanticSegmentationTrainer, self).initialize_evaluate(num_of_step)
    if 'iou' in self.val_metrics:
      self._results['confmat'] = ConfusionMatrix(self.num_classes, self.device)

  def step_evaluate(self, inputs, outputs, targets, batch_idx):
    super(SemanticSegmentationTrainer, self).step_evaluate(inputs, outputs, targets, batch_idx)
    if 'confmat' in self._results.keys():
      self._results['confmat'].update(targets.argmax(dim=1).flatten(), outputs.argmax(dim=1).flatten())
    # if self.plot:
    #   for idx in range(inputs.size(0)):
    #     if random.random() < 0.3:
    #       plot_result(image=inputs.cpu()[idx,0],
    #                   output=outputs.cpu().detach().numpy()[idx,0],
    #                   mask=targets.cpu()[idx,0])

  def end_evaluate(self, end_string: str=None) -> OrderedDict:
    if 'confmat' in self._results.keys():
      self._results['confmat'].reduce_from_all_processes()
    result = super(SemanticSegmentationTrainer, self).end_evaluate(end_string)
    if 'iou' in self.val_metrics:
      result['iou'] = self._results['confmat'].get_row_iou()
      print(self._results['confmat'])
    return result


class CryoEMTrainer(SemanticSegmentationTrainer):
  def __init__(self, *arg, **kwarg):
    super().__init__(*arg, **kwarg)

  def evaluate(self, loader, end_string: str=None):
    self.model.eval()
    self.initialize_evaluate(num_of_step=len(loader))
    with torch.no_grad():
      for idx, (inputs, targets, grid, *_) in enumerate(loader):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)['out']
        inputs = reconstruct_patched(inputs, grid)[None, :]
        targets = reconstruct_patched(targets, grid)[None, :]
        outputs = reconstruct_patched(outputs, grid)[None, :]
        # Evaluating
        self.step_evaluate(inputs, outputs, targets, idx)
    gc.collect()
    torch.cuda.empty_cache()
    return self.end_evaluate(end_string)

  def predict(self, loader):
    self.model.eval()
    predictions = list()
    with torch.no_grad():
      for batch_idx, (inputs, targets, grid, *_) in enumerate(loader):
        # inputs = inputs.view(-1,*inputs.shape[-3:]).to(self.device)
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)['out']
        inputs = reconstruct_patched(inputs, grid)
        outputs = reconstruct_patched(outputs, grid)
        preds = self.get_predictions(outputs)
        predictions.extend(preds.numpy())
    return predictions


class CryoEMEvaluator(Evaluator):
  def __init__(self, model, device, metrics, num_classes: int = 2, *arg, **kwarg):
    super().__init__(model, device, metrics, num_classes, *arg, **kwarg)
    self.plot = False

  def evaluate(self, loader, end_string: str=None):
    self.model.eval()
    self.initialize_evaluate(num_of_step=len(loader))
    with torch.no_grad():
      for batch_idx, (inputs, targets, grid, *_) in enumerate(loader):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)['out']
        inputs = reconstruct_patched(inputs, grid)[None, :]
        targets = reconstruct_patched(targets, grid)[None, :]
        outputs = reconstruct_patched(outputs, grid)[None, :]
        # Evaluating
        self.step_evaluate(inputs, outputs, targets, batch_idx)
    results = self.end_evaluate(end_string)
    gc.collect()
    torch.cuda.empty_cache()
    return results

  def predict(self, loader):
    self.model.eval()
    predictions = list()
    with torch.no_grad():
      for batch_idx, (inputs, targets, grid, *_) in enumerate(loader):
        # inputs = inputs.view(-1,*inputs.shape[-3:]).to(self.device)
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)['out']
        inputs = reconstruct_patched(inputs, grid)
        outputs = reconstruct_patched(outputs, grid)
        preds = self.get_predictions(outputs)
        predictions.extend(preds.numpy())
    return predictions

  def initialize_evaluate(self, num_of_step):
    self._results = OrderedDict()
    self._results['confmat'] = ConfusionMatrix(self.num_classes, self.device)

  def step_evaluate(self, inputs, outputs, targets, batch_idx):
    self._results['confmat'].update(targets.argmax(dim=1).flatten(), outputs.argmax(dim=1).flatten())

  def end_evaluate(self, end_string: str=None) -> OrderedDict:
    self._results['confmat'].reduce_from_all_processes()
    result = super(CryoEMEvaluator, self).end_evaluate(end_string)
    if 'iou' in self.metrics:
      result['iou'] = self._results['confmat'].get_row_iou()
      print(self._results['confmat'])
    return result

class CryoEMTrainerWithScheduler(CryoEMTrainer, Callback):
  def __init__(self, model, train_dataset, criterion, optimizer,
               device, metrics=['loss'], val_metrics=['loss', 'iou'], num_classes = 2,
               lr_scheduler=None, patience=5):
    CryoEMTrainer.__init__(self, model, train_dataset, criterion, optimizer,
                           device, metrics, val_metrics, num_classes)
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