import numpy as np
import os
import glob
from fast_slic import Slic
from PIL import Image
from skimage.segmentation import mark_boundaries
from PIL import Image
import matplotlib.pyplot as plt
for files in glob.glob("/data/CoralSCOP/CoralSCOP_data/superpixel/test/images/*.jpg"):
   p,n=os.path.split(files)
   with Image.open(files) as f:
      image = np.array(f)
   # import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
   slic = Slic(num_components=1024, compactness=10)
   assignment = slic.iterate(image) # Cluster Map
   # print(assignment)
   # print(slic.slic_model.clusters) # The cluster information of superpixels.
   # fig = plt.figure()
   plt.figure(figsize=(20, 20))
   # ax = fig.add_subplot(1, 1, 1)
   # ax.set_title("Superpixels -- %d segments" % (160))
   plt.imshow(mark_boundaries(image, assignment))
   plt.axis("off")
   # plt.savefig("/home/user/Pictures/imgs/Brazil_2009###East_Rock_Deep_T1###00010_processed.jpg")
   plt.savefig(os.path.join("/data/CoralSCOP/CoralSCOP_data/superpixel/test/new_super",n),bbox_inches="tight")
   plt.gcf().clear()