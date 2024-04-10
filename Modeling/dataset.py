"""Definition of the datasets and associated functions."""
import os
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

__all__ = [
  "get_slice_points",
  "reconstruct_patched",
  "MicrographDataset",
  "MicrographDatasetEvery"
]

def get_slice_points(image_size, crop_size):
  """
  Obtain the grid that can split image width or height into multiple ticks.
  
  Args:
    image_size (int): the width or the height of the image.
    crop_size (int): the width or the height of the cropped image.

  Examples::

    >>> image.shape
    torch.Size([3, 281, 500])
    >>> crop_size = (64, 128)
    >>> get_slice_points(image.size[-2], crop_size[-2])
    tensor([  0,  57, 114, 171, 228])
    >>> get_slice_points(image.size[-1], crop_size[-1])
    tensor([  0, 125, 250, 375])
  """
  num = math.ceil(image_size / crop_size)
  return torch.arange(0, image_size, math.ceil(image_size/num))
  # Following code is depricated since 2023/12/19.
  #   redundent = math.ceil((num * crop_size - image_size)/num) # might be wrong
  #   return torch.tensor([*range(0, image_size-crop_size, crop_size - redundent), image_size-crop_size])
  # This image demonstrate the idea of the code. 
  #   ┌──┬──┬──┬──┬────┐
  #   ├──┼──┼──┼──┼────┤
  #   ├──┼──┼──┼──┼────┤
  #   ├──┼──┼──┼──┼────┤
  #   ├──┼──┼──┼──┼────┤
  #   │  │  │  │  │    │
  #   └──┴──┴──┴──┴────┘
  # problem found 2023/12/19:
  # print((512*8-3840)/8, (3840-3840//512*512)/8)
  # print((64*8-499)/8, (499-499//64*64)/8)


def reconstruct_patched(images, grid):
  """
  Reconstruct image or feature maps from patches with a specified grid.
  
  Args:
    images (Tensor): patched images or feature maps.
    grid (Tensor): grid.

  Examples::

    >>> image.shape
    torch.Size([3, 281, 500])
    >>> crop_size = (64, 128)
    >>> patched_images.shape
    torch.Size([20, 3, 64, 128])
    >>> grid
    tensor([[[  0,   0,   0,   0,   0],
             [ 57,  57,  57,  57,  57],
             [114, 114, 114, 114, 114],
             [171, 171, 171, 171, 171],
             [228, 228, 228, 228, 228],
             [281, 281, 281, 281, 281]],

            [[  0, 125, 250, 375, 500],
             [  0, 125, 250, 375, 500],
             [  0, 125, 250, 375, 500],
             [  0, 125, 250, 375, 500],
             [  0, 125, 250, 375, 500],
             [  0, 125, 250, 375, 500]]])
    >>> get_slice_points(patched_images, grid).shape
    torch.Size([3, 281, 500])
  """
  i_num = grid.size(-2)-1
  j_num = grid.size(-1)-1
  return torch.concatenate([
    torch.concatenate([
      TF.crop(images[j_num*i_idx+j_idx], top=0, left=0,
              height=grid[0, i_idx+1, j_idx]-grid[0, i_idx, j_idx], 
              width=grid[1, i_idx, j_idx+1]-grid[1, i_idx, j_idx]) \
      for j_idx in range(j_num)
    ], dim=-1) for i_idx in range(i_num)
  ], dim=-2)

class MicrographDataset(Dataset):
  """
  Dataset for cryo-EM dataset.
  The micrographs and ground truths will be random crop to `crop_size`.
  """
  def __init__(self, image_dir, label_dir, filenames=None, crop_size=(512, 512), img_ext='.npy', crop=None):
    self.image_dir = image_dir
    self.label_dir = label_dir
    if filenames is not None:
      self.filenames = filenames
    else:
      self.filenames = sorted(os.listdir(image_dir))
    basenames = [os.path.splitext(filename)[0] for filename in filenames]
    self.images = [os.path.join(image_dir, basename+img_ext) for basename in basenames]
    self.labels = [os.path.join(label_dir, basename+'.png') for basename in basenames]
    if crop is None: # To be formalized.
      self.crop = transforms.CenterCrop(3840) # = 4096-256, uses because of the property of EMPIAR-10017
    else:
      self.crop = crop
    self.crop_size = crop_size

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    mask = TF.to_tensor(Image.open(self.labels[idx]).convert("L"))
    image = torch.from_numpy(np.load(self.images[idx]).reshape((-1,4096,4096))) # (4096, 4096) is the image size of micrographs EMPIAR-10017
    return self.transform(image, mask)

  def transform(self, image, mask):
    image = self.crop(image)
    mask = self.crop(mask)

    i, j, h, w = transforms.RandomCrop.get_params(
      image, output_size=self.crop_size)
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)
    mask = torch.concat([1-mask, mask], dim=0) # Remove this line if background is not consider.

    return image, mask

class MicrographDatasetEvery(MicrographDataset):
  """
  Dataset for cryo-EM dataset.
  The micrographs and ground truths will be divided into grid.
  """
  def __init__(self, *arg, **kwarg):
    super().__init__(*arg, **kwarg)

  def transform(self, image, mask):
    image = self.crop(image)
    mask = self.crop(mask)

    grid_i = get_slice_points(image_size=image.size(-2), crop_size=self.crop_size[-2])
    grid_j = get_slice_points(image_size=image.size(-1), crop_size=self.crop_size[-1])
    grid = torch.cartesian_prod(grid_i, grid_j)
    images = [TF.crop(image, i, j, *self.crop_size) for i, j in grid]
    masks = [TF.crop(mask, i, j, *self.crop_size) for i, j in grid]
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    grid = torch.stack([
      TF.pad(TF.pad(
        grid_i.repeat(len(grid_j),1).T, padding=(0,0,0,1), fill=image.shape[-2]
        ), padding=(0,0,1,0), padding_mode="edge"),
      TF.pad(TF.pad(
        grid_j.repeat(len(grid_i),1), padding=(0,0,1,0), fill=image.shape[-1]
        ), padding=(0,0,0,1), padding_mode="edge")
      ], dim=0)
    masks = torch.concat([1-masks, masks], dim=1) # Remove this line if background is not consider.

    return images.view(-1,1,*self.crop_size), masks, grid, mask