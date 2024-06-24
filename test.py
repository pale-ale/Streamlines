import math
import numpy as np
import scipy.ndimage

a = np.zeros((7,7), dtype=float)
a[3,3] = 1
halo_size = 3
rt = math.sqrt(halo_size)
scipy.ndimage.gaussian_filter(a, sigma=1, radius=6, output = a)
factor = 1/a[3,3]
a *= factor
print(a)