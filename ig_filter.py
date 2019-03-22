import cv2, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "images/sinb.jpg"
if args['filename']:
  filename = args['filename']

src = cv2.imread(filename)

# scaleFactor = 1.5
# ycbImage = cv2.cvtColor(src,cv2.COLOR_BGR2YCrCb)
# ycbImage = np.float32(ycbImage)
# Ychannel, Cr, Cb = cv2.split(ycbImage)
# Ychannel = np.clip(Ychannel * scaleFactor , 0, 255)
# ycbImage = np.uint8( cv2.merge([Ychannel, Cr, Cb]) )
# imcontrast = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

# Median blur
kernelSize = 5
dst = cv2.medianBlur(src,kernelSize)

# Gaussian blur
# dst=cv2.GaussianBlur(src,(5,5), 0,0)

# Bilateral filtering
# dia=15;
# sigmaColor=80;
# sigmaSpace=80;
# dst = cv2.bilateralFilter(src, dia, 
#                         sigmaColor, 
#                         sigmaSpace)

dst2 = cv2.edgePreservingFilter(dst, flags=1, sigma_s=60, sigma_r=0.4)

saturationScale = 1.5
hsvImage = cv2.cvtColor(dst2,cv2.COLOR_BGR2HSV)
hsvImage = np.float32(hsvImage)
H, S, V = cv2.split(hsvImage)
S = np.clip(S * saturationScale , 0, 255)
hsvImage = np.uint8( cv2.merge([H, S, V]) )
imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

combined = np.hstack([src, imSat])
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original   --   Output", combined)
cv2.imwrite("results/ig_filter/test1.jpg", imSat)
cv2.waitKey(0)
cv2.destroyAllWindows()