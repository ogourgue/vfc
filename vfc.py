""" VFC: Vegetation/Flow Convolution

This module allows to simulate flow velocities around vegetation patches using a computationally-efficient convolution method (conceptual approach) or spatially refine flow velocities from a coarse-resolution model (practical implementation)

Note: Practical approach (spatial refinement) currently only available for unstructured triangular grid models

Reference:
  Gourgue, O., van Belzen, J., Schwarz, C., Bouma, T.J., van de Koppel, J. & Temmerman, S. (2020) A convolution method to assess subgrid-scale interactions between flow and patchy vegetation in biogeomorphic models, Journal of Advances in Modeling Earth Systems, submitted.

"""

import numpy as np
from scipy.ndimage import rotate
from scipy.signal import fftconvolve



################################################################################
# conceptual approach ##########################################################
################################################################################

def convolution_conceptual(X, Y, V, M, beta = .5):

  """ Calculate the magnitude of the normalized flow velocity U/U0 resulting from the vegetation distribution V and the convolution mask M (Gourgue et al., 2020, Equation 10)

  Required parameters:
  X, Y (NumPy array of size (m) and (n)): coordinates of the rectangular grid
  V (NumPy array of size (m,n)): Vegetation distribution (equal 1 on vegetated cells, 0 otherwise)
  M (NumPy array of size (k,l)): Convolution mask, that is, magnitude of the normalized flow velocity around an elementary vegetation patch of the size of one rectangular grid cell (Gourgue et al., 2020)

  Optional parameter:
  beta (float, default = 0.5): Convolution parameter ranging from 0 to 1 (high values accentuate the impact of the convolution mask) (Gourgue et al., 2020, Equation 11)

  Returns:
  Numpy array of size (m,n): Magnitude of the normalized flow velocity around the vegetation

  """

  # grid size
  DX = X[1] - X[0]

  # vegetation surface over convolution mask
  SV = fftconvolve(V, np.ones(M.shape), mode = 'same') * (DX ** 2)
  SV[SV < 1] = 1

  # return normalized velocity
  return np.exp(fftconvolve(V / (SV ** beta), np.log(M), mode = 'same'))



################################################################################
# practical implementation - step 1: conceptual approach in N directions #######
################################################################################

def convolution_step_1(X, Y, V, M, beta = .5, N = 8):

  """ Calculate the normalized convoluted velocity matrix in N directions (Gourgue et al., 2020, Equation 12)

  Required parameters:
  X, Y (NumPy array of size (m) and (n)): coordinates of the rectangular grid
  V (NumPy array of size (m,n)): Vegetation distribution (equal 1 on vegetated cells, 0 otherwise)
  M (NumPy array of size (k,l)): Convolution mask, that is, magnitude of the normalized flow velocity around an elementary vegetation patch of the size of one rectangular grid cell (Gourgue et al., 2020)

  Optional parameters:
  beta (float, default = 0.5): Convolution parameter ranging from 0 to 1 (high values accentuate the impact of the convolution mask) (Gourgue et al., 2020, Equation 11)
  N (int, default = 8): Number of convolution directions

  Returns:
  Numpy array of size (N,m,n): Normalized convoluted velocity matrix in N directions

  """

  # bin size
  dtheta = 360. / N

  # convolution directions
  THETA = np.linspace(0., 360., N, endpoint = False)

  # initialize
  CONV = np.zeros((N, V.shape[0], V.shape[1]))

  # convolution in each direction
  for i in range(N):

    # rotate convolution mask
    Mi = rotate(M, THETA[i], mode = 'constant', cval = 1)

    # conceptual approach
    CONV[i, :, :] = convolution_conceptual(X, Y, V, Mi, beta)

  # return normalized convoluted velocity matrix in N directions
  return CONV



################################################################################
# practical implementation - step 2: aggregation of directional maps ###########
################################################################################

def convolution_step_2(x, y, tri, u, v, X, Y, V, M, beta = .5, N = 8):

  """ Calculate the normalized aggregated velocity matrix (Gourgue et al., 2020, Equation 12)

  Required parameters:
  x, y (NumPy arrays of size (npoin)): coordinates of the triangular grid
  tri (NumPy array of size (npoin, 3)): connectivity table of the triangular grid
  u, v (NumPy arrays of size (npoin)): coarse-resolution flow velocities on the triangular grid
  X, Y (NumPy array of size (m) and (n)): coordinates of the rectangular grid
  V (NumPy array of size (m,n)): Vegetation distribution (equal 1 on vegetated cells, 0 otherwise)
  M (NumPy array of size (k,l)): Convolution mask, that is, magnitude of the normalized flow velocity around an elementary vegetation patch of the size of one rectangular grid cell (Gourgue et al., 2020)

  Optional parameters:
  beta (float, default = 0.5): Convolution parameter ranging from 0 to 1 (high values accentuate the impact of the convolution mask) (Gourgue et al., 2020, Equation 11)
  N (int, default = 8): Number of convolution directions

  Returns:
  Numpy array of size (m,n): Normalized aggregated velocity matrix
  NumPy array of size (m,n): Voronoi matrix

  """

  # bin size
  dtheta = 360. / N

  # central bin direction
  THETA = np.linspace(0., 360., N, endpoint = False)

  # convolution step 1
  CONV1 = convolution_step_1(X, Y, V, M, beta = beta, N = N)

  # coarse-resolution flow directions and corresponding convolution bins
  theta = np.degrees(np.arctan2(v, u))
  bin = np.zeros(theta.shape, dtype = 'int')
  for i in range(N):
    bin[theta > THETA[i] - dtheta / 2] = i
  bin[theta > THETA[-1] + dtheta / 2] = 0

  # voronoi neighborhood
  VOR = voronoi(x, y, tri, X, Y)

  # select appropriate convolution bin for each voronoi cell
  CONV = np.zeros(V.shape)
  for i in range(N):
    CONV[bin[VOR] == i] = CONV1[i, :, :][bin[VOR] == i]

  # return normalized aggregated velocity and voronoi matrices
  return CONV, VOR



################################################################################

def convolution_step_3(x, y, tri, u, v, X, Y, V, M, beta = .5, nbin = 8):

  # convolution step 3
  CONV, VOR = convolution_step_2(x, y, tri, u, v, X, Y, V, M, beta = beta, \
                                 nbin = nbin)

  # conservative rescaling inside each voronoi cell
  for i in range(len(x)):
    CONV[VOR == i] /= np.mean(CONV[VOR == i])

  return CONV, VOR


################################################################################

def convolution(x, y, tri, u, v, X, Y, V, M, beta = .5, nbin = 8):

  # convolution step 3
  CONV, VOR = convolution_step_3(x, y, tri, u, v, X, Y, V, M, beta = beta, \
                                 nbin = nbin)

  # convolution product to velocity
  uv = ((u ** 2 + v ** 2) ** .5)
  return uv[VOR] * CONV


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