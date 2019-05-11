# PLOTTING: Helper Functions (dervied from https://github.com/tomgoldstein/loss-landscape/blob/master/plot_1D.py)

import torch, torchvision
from torchvision import transforms
from torch.autograd.variable import Variable

import os, copy
from os.path import exists, commonprefix

import h5py
import numpy as np
from matplotlib import pyplot as pp

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

def plot_1d_loss_err(surf_file, xmin=-1.0, xmax=1.0, value_max=5, log=False, show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_value' in f.keys(), "'train_value' does not exist"
    train_value = f['train_value'][:]

    print("train_value")
    print(train_value)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    # train_value curve
    pp.figure()
    if log:
        pp.semilogy(x, train_value)
    else:
        pp.plot(x, train_value)
    pp.ylabel('Training Value', fontsize='xx-large')
    pp.xlim(xmin, xmax)
    pp.ylim(0, value_max)
    pp.savefig(surf_file + '_1d_train_value' + ('_log' if log else '') + '.pdf',
                dpi=300, bbox_inches='tight', format='pdf')

    if show: pp.show()
    f.close()


# PLOTTING: Helper Functions (dervied from https://github.com/tomgoldstein/loss-landscape/blob/master/plot_2D.py)
    
def plot_2d_contour(surf_file, surf_name='train_value', vmin=0.001, vmax=0.01, vlevel=0.003, show=False):
  f = h5py.File(surf_file, 'r')
  x = np.array(f['xcoordinates'][:])
  y = np.array(f['ycoordinates'][:])
  X, Y = np.meshgrid(x, y)

  if surf_name in f.keys(): Z = np.array(f[surf_name][:])
  else: print ('%s is not found in %s' % (surf_name, surf_file))

  print('------------------------------------------------------------------')
  print('plot_2d_contour')
  print('------------------------------------------------------------------')
  print("loading surface file: " + surf_file)
  print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
  print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
  print(Z)

  if (len(x) <= 1 or len(y) <= 1): print('The length of coordinates is not enough for plotting contours')
  if (len(x) <= 1 or len(y) <= 1): return

  # --------------------------------------------------------------------
  # Plot 2D contours
  # --------------------------------------------------------------------
  fig = pp.figure()
  CS = pp.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
  pp.clabel(CS, inline=1, fontsize=8)
  fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
              bbox_inches='tight', format='pdf')

  fig = pp.figure()
  print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
  CS = pp.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
  fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
              bbox_inches='tight', format='pdf')

  # --------------------------------------------------------------------
  # Plot 2D heatmaps
  # --------------------------------------------------------------------
  fig = pp.figure()
  sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                         xticklabels=False, yticklabels=False)
  sns_plot.invert_yaxis()
  sns_plot.get_figure().savefig(surf_file + '_' + surf_name + '_2dheat.pdf',
                                dpi=300, bbox_inches='tight', format='pdf')

  # --------------------------------------------------------------------
  # Plot 3D surface
  # --------------------------------------------------------------------
  fig = pp.figure()
  ax = Axes3D(fig)
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
              bbox_inches='tight', format='pdf')

  f.close()
  if show: pp.show()