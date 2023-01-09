import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from numpy.linalg import eig
from skimage import filters
def normalization(img):
    Rn = img[:,:,0]/np.max(img[:,:,0])
    Gn = img[:,:,1]/np.max(img[:,:,1])
    Bn = img[:,:,2]/np.max(img[:,:,2])

    r= Rn/(Rn+Gn+Bn)
    r[np.where(np.isnan(r))]=0
    g = Gn/(Rn+Gn+Bn)
    g[np.where(np.isnan(g))]=0
    b = Bn/(Rn+Gn+Bn)
    b[np.where(np.isnan(b))]=0
    return r, g, b
#import plant image
img = imread('plant.jpg')
plant = np.array(img)
pixel_num = plant.shape[0]*plant.shape[1]
#normalize image

r, g, b = normalization(plant)

#different gray scale
ExG = 2*g-r-b
ExGx = np.reshape(ExG, (pixel_num, 1))
ExGx = (ExGx-np.mean(ExGx))
plt.imshow(ExG,cmap='gray')
plt.title('ExG')
plt.show()

ExGR = 3*g-2.4*r-b
ExGRx = np.reshape(ExGR, (pixel_num, 1))
ExGRx = (ExGRx-np.mean(ExGRx))
plt.imshow(ExGR,cmap='gray')
plt.title('ExGR')
plt.show()

# VEG = g/(r**0.667*b**(1-0.667))
# VEG[np.where(np.isnan(VEG))]=0
# VEGx = np.reshape(VEG, (pixel_num, 1))
# VEGx = (VEGx-np.mean(VEGx))
# plt.imshow(VEG,cmap='gray')
# plt.title('VEG')
# plt.show()

NDI = (g-r)/(g+r)
NDI[np.where(np.isnan(NDI))]=0
NDIx = np.reshape(NDI, (pixel_num, 1))
NDIx = (NDIx-np.mean(NDIx))
plt.imshow(NDI,cmap='gray')
plt.title('NDI')
plt.show()

CIVE = 0.441*r-0.811*g+0.385*b+18.78745
CIVEx = np.reshape(CIVE, (pixel_num, 1))
CIVEx = (CIVEx-np.mean(CIVEx))
plt.imshow(CIVE,cmap='gray')
plt.title('CIVE')
plt.show()


X = np.concatenate([ExGx, ExGRx,  NDIx, CIVEx],axis=1)
Cor = 1/(pixel_num-1)*np.matmul(np.transpose(X), X)

w,v= eig(Cor)
pca1 = v[0,:]
PCA_gray = pca1[0]*ExG+pca1[1]*ExGR+pca1[2]*NDI+pca1[3]*CIVE


plt.imshow(PCA_gray,cmap ='gray')
plt.show()


otsu_thresh = filters.threshold_otsu(PCA_gray)
final_image = PCA_gray>otsu_thresh
plt.imshow(final_image,cmap='gray')
plt.show()



