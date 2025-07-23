import xupy as xp
from skimage.draw import disk

a = xp.zeros((1000,1000))
b = xp.random.normal(0, 1, a.shape)
mask = xp.ones(a.shape)
masked = disk((500,500), 256)
mask[masked] = 0

am = xp.masked_array(data = a, mask=mask)
bm = xp.masked_array(data = b, mask=mask)
