import numpy as np
from scipy.signal import fftconvolve


################################################################################

def convolution_simple(veg, mask, dx):

  """
  veg: numpy array of shape (nx, ny)
  mask: numpy array of shape (mx, my)
  dx: grid size [m]

  return:
    vh: numpy array of shape (nx, ny)
  """

  # scaling factor
  fs = fftconvolve(veg, np.ones(mask.shape, dtype = float), mode = 'same') * \
       (dx ** 2)
  fs[fs < 1] = 1

  # convolution product
  vh = np.exp(fftconvolve(veg / (fs ** .5), np.log(mask), mode = 'same'))

  return vh


################################################################################

def mask_diffusion_1d(mask, s, axis = 1):

  """
  mask: numpy array of shape (nx, ny)
  s: diffusion number (s = nu * dt / dx ** 2)
     nu: diffusion coefficient [m^2/s]
     dt: time step [s]
     dx: grid size [m]
  axis: axis along which the 1d diffusion is applied

  return:
    mask: numpy array of shape (nx, ny)
  """

  # rotate mask
  if axis == 0:
    mask = mask.T

  # number of iterations
  smax = .5
  if s < smax:
    n = 1
  else:
    n = np.int(np.floor(s / smax)) + 1

  # iteration diffusion number
  ss = s / n

  # for each iteration
  for i in range(n):

    # no-flux boundary condition
    mask[:,  0] = 4 / 3 * mask[:,  1] - 1 / 3 * mask[:,  2]
    mask[:, -1] = 4 / 3 * mask[:, -2] - 1 / 3 * mask[:, -3]

    # inside domain
    mask[:, 1:-1] += ss * (mask[:, :-2] - 2 * mask[:, 1:-1] + mask[:, 2:])

  # rotate mask
  if axis == 0:
    mask = mask.T

  return mask


################################################################################

def mask_export(filename, mask, dx):

  """
  filename: file name in which mask will be exported
  mask: numpy array of shape (nx, ny)
  dx: mask grid size
  """

  # open file
  file = open(filename, 'w')

  # write header
  np.array(mask.shape[0], dtype = int).tofile(file)
  np.array(mask.shape[1], dtype = int).tofile(file)
  np.array(dx, dtype = float).tofile(file)

  # write data
  np.array(mask, dtype = float).tofile(file)

  # close file
  file.close()


################################################################################

def mask_import(filename):

  """
  filename: file name from which mask will be imported

  return:
    mask: numpy array of shape (nx, ny)
    dx: mask grid size
  """

  # open file
  file = open(filename, 'r')

  # read header
  nx = np.fromfile(file, dtype = int, count = 1)[0]
  ny = np.fromfile(file, dtype = int, count = 1)[0]
  dx = np.fromfile(file, dtype = float, count = 1)[0]

  # read data
  mask = np.reshape(np.fromfile(file, dtype = float, count = nx * ny), (nx, ny))

  # close file
  file.close()

  # return
  return [mask, dx]


################################################################################

def veg_export(filename, veg, x0, y0, dx):

  """
  filename: file name in which mask will be exported
  veg: numpy array of shape (nx, ny)
  x0, y0: center coordinates of the lower left cell
  dx: mask grid size
  """

  # open file
  file = open(filename, 'w')

  # write header
  np.array(x0, dtype = float).tofile(file)
  np.array(y0, dtype = float).tofile(file)
  np.array(veg.shape[0], dtype = int).tofile(file)
  np.array(veg.shape[1], dtype = int).tofile(file)
  np.array(dx, dtype = float).tofile(file)

  # write data
  np.array(veg, dtype = int).tofile(file)

  # close file
  file.close()


################################################################################

def veg_import(filename):

  """
  filename: file name from which mask will be imported

  return:
    veg: numpy array of shape (nx, ny)
    x0, y0: center coordinates of the lower left cell
    dx: mask grid size
  """

  # open file
  file = open(filename, 'r')

  # read header
  x0 = np.fromfile(file, dtype = float, count = 1)[0]
  y0 = np.fromfile(file, dtype = float, count = 1)[0]
  nx = np.fromfile(file, dtype = int, count = 1)[0]
  ny = np.fromfile(file, dtype = int, count = 1)[0]
  dx = np.fromfile(file, dtype = float, count = 1)[0]

  # read data
  veg = np.reshape(np.fromfile(file, dtype = int, count = nx * ny), (nx, ny))

  # close file
  file.close()

  # return
  return [veg, x0, y0, dx]
