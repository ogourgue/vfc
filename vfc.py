import numpy as np
from scipy.ndimage import rotate
from scipy.signal import fftconvolve


################################################################################

def convolution(x, y, tri, u, v, X, Y, V, M, fsp = 1, nbin = 8):

  # vegetation and mask grid size
  DX = X[1] - X[0]

  # bin size
  dtheta = 360. / nbin

  # central bin direction
  theta0 = np.linspace(0., 360., nbin, endpoint = False)

  # convolution in each bin
  TMP = []
  for i in range(nbin):

    # rotate mask
    M_i = rotate(M, theta0[i], mode = 'constant', cval = 1)

    # patch surface scaling factor
    FPS = fftconvolve(V, np.ones(M_i.shape), mode = 'same') * (DX ** 2)
    FPS[FPS < 1] = 1

    # convolution product
    TMP.append(np.exp(fftconvolve(V / (FPS ** .5), np.log(M_i), mode = 'same')))

    # species rescaling
    TMP[i] = 1. + (TMP[i] - 1.) * fsp

  # original flow direction and corresponding bin
  theta = np.degrees(np.arctan2(v, u))
  bin = np.zeros(theta.shape, dtype = 'int')
  for i in range(nbin):
    bin[theta > theta0[i] - dtheta / 2] = i
  bin[theta > theta0[-1] + dtheta / 2] = 0

  # voronoi neighborhood
  VOR = voronoi(x, y, tri, X, Y)

  # select appropriate bin for each voronoi cell
  CONV = np.zeros(V.shape)
  for i in range(nbin):
    CONV[bin[VOR] == i] = TMP[i][bin[VOR] == i]

  # conservative rescaling inside each voronoi cell
  for i in range(len(x)):
    CONV[VOR == i] /= np.mean(CONV[VOR == i])

  # convolution product to velocity
  uv = (u ** 2 + v ** 2) ** .5
  UV = uv[VOR] * CONV
  return UV


################################################################################

def mask_diffusion_1d(mask0, s, axis = 1):

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
    mask0 = mask0.T

  # number of iterations
  smax = .5
  if s < smax:
    n = 1
  else:
    n = np.int(np.floor(s / smax)) + 1

  # iteration diffusion number
  ss = s / n

  # initialize
  mask = np.zeros(mask0.shape)

  # for each iteration
  for i in range(n):

    # no-flux boundary condition
    mask[:,  0] = 4 / 3 * mask0[:,  1] - 1 / 3 * mask0[:,  2]
    mask[:, -1] = 4 / 3 * mask0[:, -2] - 1 / 3 * mask0[:, -3]

    # inside domain
    mask[:, 1:-1] = mask0[:, 1:-1] + \
                    ss * (mask0[:, :-2] - 2 * mask0[:, 1:-1] + mask0[:, 2:])

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


################################################################################

def voronoi(x, y, tri, X, Y):

  """
  compute the voronoi array from an unstructured triangular grid and a structured squared grid

  triangular grid defined by:
    x, y: arrays of shape (npoin) giving node coordinates
    tri: array of shape (nelem, 3) giving node connectivity table

  structured grid defined by:
    X: array of size (nx) giving center cell coordinates in the x direction
    Y: array of size (ny) giving center cell coordinates in the y direction

  return:
    vor: array of shape (nx, ny) giving index of closest triangular grid node (-1 if outside triangular grid)
  """


  # initialize voronoi array
  vor = np.zeros((len(X), len(Y)), dtype = int) - 1

  # for each triangle
  for i in range(tri.shape[0]):

    # triangle vertex coordinates
    x0 = x[tri[i, 0]]
    x1 = x[tri[i, 1]]
    x2 = x[tri[i, 2]]
    y0 = y[tri[i, 0]]
    y1 = y[tri[i, 1]]
    y2 = y[tri[i, 2]]

    # triangle bounding box and corresponding indices on the structured grid
    xmin = min([x0, x1, x2])
    xmax = max([x0, x1, x2])
    ymin = min([y0, y1, y2])
    ymax = max([y0, y1, y2])
    try:imin = int(np.argwhere(X <= xmin)[-1])
    except:imin = 0
    try:imax = int(np.argwhere(X >= xmax)[0])
    except:imax = len(X) - 1
    try:jmin = int(np.argwhere(Y <= ymin)[-1])
    except:jmin = 0
    try:jmax = int(np.argwhere(Y >= ymax)[0])
    except:jmax = len(Y) - 1

    # local grid of the bounding box
    Xloc, Yloc = np.meshgrid(X[imin:imax + 1], Y[jmin:jmax + 1], \
                             indexing = 'ij')

    # compute barycentric coordinates
    s0 = ((y1 - y2) * (Xloc  - x2) + (x2 - x1) * (Yloc  - y2)) \
       / ((y1 - y2) * (x0    - x2) + (x2 - x1) * (y0    - y2))
    s1 = ((y2 - y0) * (Xloc  - x2) + (x0 - x2) * (Yloc  - y2)) \
       / ((y1 - y2) * (x0    - x2) + (x2 - x1) * (y0    - y2))
    s2 = 1. - s0 - s1

    # s[i,j] is True if barycentric coordinates are all positive, and the
    # corresponding structured grid cell is inside the triangle
    s = (s0 >= 0.) * (s1 >= 0.) * (s2 >= 0.)

    # distance to triangle vertices
    d = np.array([(x0 - Xloc) * (x0 - Xloc) + (y0 - Yloc) * (y0 - Yloc), \
                  (x1 - Xloc) * (x1 - Xloc) + (y1 - Yloc) * (y1 - Yloc), \
                  (x2 - Xloc) * (x2 - Xloc) + (y2 - Yloc) * (y2 - Yloc)])

    # tmp[i,j] is the number of the closest vertex...
    tmp = tri[i, np.argmin(d, 0)]

    # ... but outside the triangle, tmp[i,j] is the value of the voronoi array
    vor_loc = vor[imin:imax + 1, jmin:jmax + 1]
    tmp[s == False] = vor_loc[s == False]

    # update voronoi array for structured grid cells inside the triangle
    vor[imin:imax + 1, jmin:jmax + 1] = tmp

  return vor