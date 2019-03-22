import cv2,argparse
import numpy as np
import os

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        # var = 0.1
        # sigma = var**0.5
        gauss = np.random.normal(mean,1,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * (gauss/2)
        return noisy

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "images/yuju.jpg"
if args['filename']:
  filename = args['filename']

original = cv2.imread(filename)

b_channel, g_channel, r_channel = cv2.split(original)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
fin_orig = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

scaleDown = cv2.resize(fin_orig, None, fx= 0.7, fy= 0.7, interpolation= cv2.INTER_LINEAR)
dst = np.copy(original)
dst = cv2.resize(original, None, fx= 0.7, fy= 0.7, interpolation= cv2.INTER_LINEAR)
src = cv2.imread("images/farm.jpg")
output = np.copy(scaleDown)

noise_img = noisy('speckle', scaleDown)
noise_img = noise_img.astype('uint8')

b,g,r,a = cv2.split(noise_img)
foreground = cv2.merge((b,g,r))
alpha = cv2.merge((a,a,a))
foreground = foreground.astype(float)
dst = dst.astype(float)
alpha = alpha.astype(float)/255
foreground = cv2.multiply(alpha, foreground)
dst = cv2.multiply(1.0 - alpha, dst)
final = cv2.add(foreground, dst)
final = final.astype('uint8')

output = np.copy(dst)
output = output.astype('uint8')

srcLab = np.float32(cv2.cvtColor(src,cv2.COLOR_BGR2LAB))
dstLab = np.float32(cv2.cvtColor(final,cv2.COLOR_BGR2LAB))
outputLab = np.float32(cv2.cvtColor(output,cv2.COLOR_BGR2LAB))

srcL, srcA, srcB = cv2.split(srcLab)
dstL, dstA, dstB = cv2.split(dstLab)
outL, outA, outB = cv2.split(outputLab)

outL = dstL - dstL.mean()
outA = dstA - dstA.mean()
outB = dstB - dstB.mean()

outL *= srcL.std() / dstL.std()
outA *= srcA.std() / dstA.std()
outB *= srcB.std() / dstB.std()

outL = outL + srcL.mean()
outA = outA + srcA.mean()
outB = outB + srcB.mean()

outL = np.clip(outL, 0, 255)
outA = np.clip(outA, 0, 255)
outB = np.clip(outB, 0, 255)

outputLab = cv2.merge([outL, outA, outB])
outputLab = np.uint8(outputLab)

output= cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)
gaus = cv2.GaussianBlur(output,(5,5),0,0)

cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image", scaleDown)
cv2.namedWindow("Output Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output Image", gaus)
cv2.waitKey(0)
cv2.destroyAllWindows()
