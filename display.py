import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def display_results(content_img,style_img,out_img):
    plt.figure(figsize=(15, 5))

    # Content Image
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(Image.open(content_img)))
    plt.title('Content Image')
    plt.axis('off')

    # Style Image
    plt.subplot(1, 3, 2)
    plt.imshow(np.array(Image.open(style_img)))
    plt.title('Style Image')
    plt.axis('off')

    # AdaIn Result Image
    plt.subplot(1, 3, 3)
    plt.imshow(np.array(Image.open(out_img)))
    plt.title('AdaIn Result')
    plt.axis('off')

    plt.show()