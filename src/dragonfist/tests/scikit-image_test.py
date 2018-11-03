from skimage import data, io, filters

image = data.coins()
gabor = filters.gabor(image, 1)
edges = filters.sobel(image)
io.imshow(edges)
io.show()