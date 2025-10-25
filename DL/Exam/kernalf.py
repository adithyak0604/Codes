import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
image=np.array([
    [10,10,10,10,10],
    [10,50,50,50,10],
    [10,50,100,50,10],
    [10,50,50,50,10],
    [10,10,10,10,10]
], dtype=np.float32)

kernals={
    "Identity":np.array([[0,0,0],[0,1,0],[0,0,0]]),
    "Edge Detection":np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    "Sharpen":np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "Box Blur":np.ones((3,3))/9,
    "Gaussian Blur":(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
}

results={}
for name, kernal in kernals.items():
    results[name]=convolve2d(image, kernal, mode='same', boundary='fill', fillvalue=0)

plt.figure(figsize=(12,6))
plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

for i, (name, result) in enumerate(results.items(), start=2):
    plt.subplot(2,3,i)
    plt.imshow(result, cmap='gray')
    plt.title(name)
    plt.axis('off')
plt.tight_layout()
plt.show()
