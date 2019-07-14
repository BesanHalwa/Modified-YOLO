import cv2
import numpy as np
import glob

files = glob.iglob('VOC2012/JPEGImages/*')
count = 0

for file in files:

    print(file)
    count = count + 1

    # load image
    img = cv2.imread(file)
    h, w, c = img.shape
    img = img.reshape((h*w*c))

    # now roll 4 pixels
    new_img = np.roll(img, 4)

    out_img = abs(new_img - img)
    out_img = out_img.reshape((h,w,c))
    cv2.imwrite(file, out_img)

    print('Image Count: ', count)
