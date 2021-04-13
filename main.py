import numpy as np
import cv2
import os
from utils import face_mask_extraction, interF
from scipy.signal import medfilt2d

# Load camera params
stereoParams_mr = np.load('./stereoParams_mr.npy') 
stereoParams_ml = np.load('./stereoParams_ml.npy') 

# Subject name and image pair
sName = '2'
pNamme = '1'

# Read facial feature points 
features_m = np.load(os.path.join('./s',sName,'/m',pNamme,'.csv'))
features_r = np.load(os.path.join('./s',sName,'/r',pNamme,'.csv'))
features_l = np.load(os.path.join('./s',sName,'/l',pNamme,'.csv'))

# Read images
img_m = cv2.imread(os.path.join('./s',sName,'/subject',sName,'_Middle_',pNamme,'_e1.png'))
img_r = cv2.imread(os.path.join('./s',sName,'/subject',sName,'_Right_',pNamme,'_e.png'))
img_l = cv2.imread(os.path.join('./s',sName,'/subject',sName,'_Left_',pNamme,'_e.png'))

# Face Segmentation 
mask_m = face_mask_extraction(img_m,'m')
mask_r = face_mask_extraction(img_r,'r')
mask_l = face_mask_extraction(img_l,'l')

## Stereo Rectification
# stereo map
left_camera_matrix = np.array([[824.93564, 0., 251.64723],[0., 825.93598, 286.58058],[0., 0., 1.]])
left_distortion = np.array([[0.23233, -0.99375, 0.00160, 0.00145, 0.00000]])
right_camera_matrix = np.array([[853.66485, 0., 217.00856],[0., 852.95574, 269.37140],[0., 0., 1.]])
right_distortion = np.array([[0.30829, -1.61541, 0.01495, -0.00758, 0.00000]])
om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-70.59612, -2.60704, 18.87635]) # 平移关系向量
size = (640, 480) # 图像尺寸
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

# Rectify face images
# 根据更正map对图片进行重构
img_l_rec = cv2.remap(img_l, left_map1, left_map2, cv2.INTER_LINEAR)
img_r_rec = cv2.remap(img_r, right_map1, right_map2, cv2.INTER_LINEAR)
img_m_rec = cv2.remap(img_m, left_map1, left_map2, cv2.INTER_LINEAR)
mask_l_rec = cv2.remap(mask_l, left_map1, left_map2, cv2.INTER_LINEAR)
mask_r_rec = cv2.remap(mask_r, left_map1, left_map2, cv2.INTER_LINEAR)
mask_m_rec = cv2.remap(mask_m, left_map1, left_map2, cv2.INTER_LINEAR)

## Facial Feature Points 
xydif = np.abs(features_m-features_r)
disparity_feature_points1 = np.sqrt(np.power(xydif[:,1],2) + np.power(xydif[:,2],2))
xydif = np.abs(features_m-features_l)
disparity_feature_points2 = -np.sqrt(np.power(xydif[:,1],2) + np.power(xydif[:,2],2))
max_disp_FP = round(np.max(disparity_feature_points1))
min_disp_FP = round(np.min(disparity_feature_points1))
max_disp_FP2 = round(np.max(disparity_feature_points2))
min_disp_FP2 = round(np.min(disparity_feature_points2))

# Disparty map for facial feature points
disp_FP = np.zeros(img_m_rec.shape[0],img_m_rec.shape[0])
disp_FP2 = np.zeros(img_m_rec.shape[0],img_m_rec.shape[0])
for i in range(68):
    disp_FP[features_m[i,0],features_m[i,1]] = disparity_feature_points1[i]
    disp_FP[features_m[i,0],features_m[i,1]] = disparity_feature_points2[i]

# Determine disparity ranges according to featrue points disparities
disparityRange = np.array([max_disp_FP,min_disp_FP])
disparityRange2 = np.array([max_disp_FP2,min_disp_FP2])

## Disparity MAP
# Convert RGB image gray-level image
gray_img_m = cv2.cvtColor(img_m_rec, cv2.COLOR_BGR2GRAY)
gray_img_l = cv2.cvtColor(img_l_rec, cv2.COLOR_BGR2GRAY)
gray_img_r = cv2.cvtColor(img_r_rec, cv2.COLOR_BGR2GRAY)

## Image smoothing
h = cv2.getGaussianKernel(5, 1)
gray_img_m = cv2.filter2D(gray_img_m, -1, h ,borderType=cv2.BORDER_CONSTANT)
gray_img_l = cv2.filter2D(gray_img_l, -1, h ,borderType=cv2.BORDER_CONSTANT)
gray_img_r = cv2.filter2D(gray_img_r, -1, h ,borderType=cv2.BORDER_CONSTANT)

# Disparity PARAMS
bs = 15        #defauld bs=15
cTH = 0.7      #default 0.5
uTH = 15       #default 15
tTH = 0.0000   #default 0.0002 only applies if method is blockmatching
dTH = 15       #default []

#--------------------------l-r pair-------------------------------------
num = cv2.getTrackbarPos("num", "depth")
stereoBM = cv2.StereoBM_create(numDisparities=16*num, blockSize=5)
stereoSGBM = cv2.StereoSGBM_create(minDisparity=0,numDisparities=160,blockSize=5)
disparityMapBM = stereoBM.compute(gray_img_l, gray_img_r)
disparityMap = stereoSGBM.compute(gray_img_l, gray_img_r)

## Unreliable Points
unreliable = disparityMap < -1e+12
unreliable = unreliable | (1-mask_m_rec)

# Get rid of unrelible pixels
dispartity_ref = disparityMap*(1-unreliable)

## Get rid of unrelible pixels usnig feature points and interpolation
face_lower_th = 20
face_upper_th = 100
thDisp1 = 15
disp_int, disp_int_only_face = interF (dispartity_ref,disparityMapBM,mask_m_rec, features_m, 
            disp_FP,disparity_feature_points1, disparityRange, thDisp1,face_lower_th,face_upper_th)

## Median filtering
dsp = disp_int
dsp_med_filt = medfilt2d(dsp,[50,50])
unreliable_out = 1-(dsp_med_filt!=0)

## Smoothing disparity maps by Gaussian filter
h = cv2.getGaussianKernel([10,10], 2)
dsp_gauss = cv2.filter2D(dsp_med_filt, -1, h)

## Generate Point Clouds
#Reproject points into 3D
xyzPoints = cv2.reprojectImageTo3D(dsp_gauss, Q)

## Generate 3D face meshes

mesh_create_func( img_middle_rec, dsp_gauss1, xyzPoints1, unreliable_out1);
mesh_create_func( img_middle_rec2, dsp_gauss2, xyzPoints2, unreliable_out2);




# ## Merging 2 point clouds and estimate error (ICP algoritm is used)
# %Create point cloud object
# ptCloud1 = pointCloud(xyzPoints11);
# ptCloud2 = pointCloud(xyzPoints22);

# %Subsample point clouds by scale
# scale= 1;
# xyzPoints1_down = pcdownsample(ptCloud1,'random',scale);
# xyzPoints2_down = pcdownsample(ptCloud2,'random',scale);

# %Obtain rotation matrix from stereoParams
# R1 = stereoParams_mr.RotationOfCamera2;
# R2 = stereoParams_ml.RotationOfCamera2;

# %Define Initilial Transform
# tformI =  affine3d();
# tformI.T(1:3,1:3) = R2*inv(R1); %R1 is 3x3 matrix so using inv() is okay!

# %Transform the point cloud 1 such a way that overlay point cloud 2
# [tform,movingReg,rmse,squaredError] = pcregrigid(xyzPoints1_down,xyzPoints2_down,'MaxIterations',10, ...
#    'InitialTransform', tformI);

# %Visiluze point clouds
# % figure();pcshow(xyzPoints1_down);
# % figure();pcshow(xyzPoints2_down);

# %ptCloudAligned = pctransform(xyzPoints1_down,tform);

# %Merge point clouds
# ptCloudOut = pcmerge(movingReg,xyzPoints2_down,1);
# %figure();pcshow(ptCloudOut);

# %% Accuracy estimation
# th_dr = 3;
# th_dr2 = 3/2;
# dr = sqrt(squaredError);
# acc = sum(dr<th_dr)/length(dr)
# acc2 = sum(dr<th_dr2)/length(dr)
