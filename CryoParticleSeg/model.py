import torch
import torch.nn as nn
from collections import OrderedDict
from convcrf import GaussCRF, ConvCRF


def create_model(backbone):
  model = backbone
  model.classifier.add_module(name='output', module=nn.Softmax(dim=1))
  return model


def create_crf_model(backbone, config, shape, num_classes, use_gpu=False, freeze_backbone=False):
  if freeze_backbone:
    for params in backbone.parameters():
      params.requires_grad = False
  model = ModelWithGausscrf(backbone, config=config, shape=shape, num_classes=num_classes, use_gpu=use_gpu)
  return model


class ModelWithGausscrf(nn.Module):
  def __init__(self, backbone, config, shape, num_classes, use_gpu=False):
    super().__init__()
    self.backbone = backbone
    self.config = config
    self.num_classes = num_classes
    self.shape = shape
    self.use_gpu = use_gpu
    self.gausscrf = GaussCRF(conf=self.config, shape=self.shape,
                             nclasses=self.num_classes, use_gpu= self.use_gpu)

  def forward(self, x):
    unary = self.backbone(x)['out']
    return OrderedDict([
      ('backbone', unary),
      ('out', self.gausscrf(unary, x))
    ])


try:
  import CRF

  class ModelWithFWCRF():
      def __init__(self, backbone, params, num_classes:int,
                    alpha=160, beta=0.05, gamma=3.0, iterations=5):
          super().__init__()
          self.backbone = backbone
          self.crf = CRF.DenseGaussianCRF(
            classes=num_classes,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            spatial_weight=1.0,
            bilateral_weight=1.0,
            compatibility=1.0,
            init='potts',
            solver='fw',
            iterations=iterations,
            params=params)

      def forward(self, x):
          """
          x is a batch of input images
          """
          logits = self.backbone(x)['out']
          logits = self.crf(x, logits)
          return logits

  def create_fwcrf_model(backbone, params, num_classes, alpha=160, beta=0.05, gamma=3.0, iterations=5, freeze_backbone=False):
    if freeze_backbone:
      for params in backbone.parameters():
        params.requires_grad = False
    model = ModelWithFWCRF(backbone, params, num_classes, alpha=alpha, beta=beta, gamma=gamma, iterations=iterations)
    return model
except:
  pass