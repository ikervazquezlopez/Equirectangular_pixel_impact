import cv2
from PIL import Image
import skimage.measure as metrics
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt





if __name__ == "__main__":





    img = cv2.imread(sys.argv[1])
    #img = cv2.resize(img, None, fx=0.15625, fy=0.15625)
    h, w, c = img.shape
    print(img.shape)

    psnr_list = []

    for r in range(0, int(h/2)):

        if r % 100 == 0:
            plt.clf()
            plt.xlabel("Row")
            plt.ylabel("PSNR")
            plt.ylim(bottom=0, top=400)
            plt.plot(psnr_list)
            plt.savefig('pixel_impact.png')

        row = img[r]
        psnr = 0.0
        for c in range(1,w-1):
            mod_r = row.copy()
            interpolated = 0.5*mod_r[c-1] + 0.5*mod_r[c+1]
            mod_r[c] = interpolated.astype(np.uint8)
            psnr += cv2.PSNR(row, mod_r)/(w-2)
        psnr_list.append(psnr)
        print(r, psnr)


    print("Minimum PSNR: {}".format(np.min(np.array(psnr_list))))

    f = open('minimum_PSNR.txt', "w")
    f.write(np.min(np.array(psnr_list)))
    f.close()
