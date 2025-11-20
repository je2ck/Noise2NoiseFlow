import tifffile
import matplotlib.pyplot as plt

img = tifffile.imread("./noise2noiseflow/data_atom/test/scene_16001/a.tif")
den = tifffile.imread("./noise2noiseflow/data_atom/test/scene_16001/b.tif")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(den, cmap='gray')
axes[1].set_title("Denoised")
axes[1].axis("off")

# plt.tight_layout()
# plt.show()

def check_tif(path):
    arr = tifffile.imread(path)
    print(path)
    print("dtype:", arr.dtype)
    print("shape:", arr.shape)
    print("min/max:", arr.min(), arr.max())
    print("mean:", arr.mean())
    print("std :", arr.std())
    print("-"*40)

check_tif("./noise2noiseflow/data_atom/test/scene_16002/a.tif")
check_tif("./noise2noiseflow/data/test/scene_16002/a.tif")