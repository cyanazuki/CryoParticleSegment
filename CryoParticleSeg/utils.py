import functools
from tqdm.auto import tqdm


def _step_tqdm(trainer, tqdm_loader, results, batch_idx):
  tqdm_loader.set_postfix(
    {metric.title(): f"{results[metric][batch_idx]:.4f}"
     for metric in trainer.metrics})
  
def tqdm_plugin_for_loader(desc=None):
  """Let the progression of the loader present by tqdm."""
  def tqdm_plugin(function):
    @functools.wraps(function)
    def wrapper(self, loader, *arg, **kwarg):
      with tqdm(loader, desc=desc) as tqdm_loader:
        self.step_action['tqdm'] = _step_tqdm
        results = function(self, tqdm_loader, *arg, **kwarg)
        tqdm_loader.set_postfix({metric.title(): f"{results[metric].mean():.4f}"
                       for metric in self.metrics})
        self.step_action.pop('tqdm')
      return results

    return wrapper

  return tqdm_plugin