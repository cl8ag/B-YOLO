# coding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
objp = np.zeros((11 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

objpoints = []
imgpointsR = []
imgpointsL = []

square_size_mm = 25


scale_factor_sum = 0
valid_images = 0

for i in range(1, 17):
    t = str(i)
    ChessImaR = cv2.imread('./datas/r/' + t + '.jpg', 0)
    ChessImaL = cv2.imread('./datas/l/' + t + '.jpg', 0)
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (8, 11), None)
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (8, 11), None)
    if retR and retL:
        valid_images += 1
        pixel_size_w = np.linalg.norm(cornersL[0] - cornersL[4])
        pixel_size_h = np.linalg.norm(cornersL[0] - cornersL[-5])
        average_pixel_size = (pixel_size_w / 4 + pixel_size_h / 7) / 2
        scale_factor = square_size_mm / average_pixel_size
        print(scale_factor)
        scale_factor_sum += scale_factor
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

if valid_images > 0:
    #print(valid_images)
    average_scale_factor = scale_factor_sum / valid_images
    print(f"Average scale factor: {average_scale_factor} mm/pixel")
else:
    print("No valid images found for scale factor calculation.")

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)


hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))


retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)


hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))


flags = 0
# flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, OmtxL, distL, OmtxR,distR,ChessImaR.shape[::-1], criteria_stereo, flags)
print(retS)
print(T)

rectify_scale = 0
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,ChessImaR.shape[::-1], R, T,rectify_scale, (0, 0))


Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)

Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)



for i in range(1, 2):
    t = str(i)
    frameR = cv2.imread('./datas/' + 'r' + '.jpg', 0)
    frameL = cv2.imread('./datas/' + 'l' + '.jpg', 0)

    Left_rectified = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                               0)
    im_L = Image.fromarray(Left_rectified)

    Right_rectified = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR,
                                cv2.BORDER_CONSTANT, 0)
    im_R = Image.fromarray(Right_rectified)

    width = im_L.size[0] * 2
    height = im_L.size[1]

    img_compare = Image.new('RGB', (width, height))
    img_compare.paste(im_L, (0, 0))
    img_compare.paste(im_R, (im_L.size[0], 0))


    for i in range(1, 20):
        len = 1080 / 20
        plt.axhline(y=i * len, color='r', linestyle='-')

    plt.imshow(img_compare)
    plt.show()


len = __builtins__.len

mean_error = 0
for i in range(len(objpoints)):

    imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)


    errorR = cv2.norm(imgpointsR[i], imgpointsR2, cv2.NORM_L2) / len(imgpointsR[i])
    errorL = cv2.norm(imgpointsL[i], imgpointsL2, cv2.NORM_L2) / len(imgpointsL[i])
    mean_error += (errorR + errorL) / 2


total_error = mean_error / len(objpoints)
print(total_error)

with open("calibration_result.txt", "w") as f:
    f.write(str(mtxR) + "\n")
    f.write(str(distR) + "\n")
    f.write(str(OmtxR) + "\n")
    f.write(str(roiR) + "\n")
    f.write("\n")

    f.write(str(mtxL) + "\n")

    f.write(str(distL) + "\n")

    f.write(str(OmtxL) + "\n")

    f.write(str(roiL) + "\n")
    f.write("\n")

    f.write(str(R) + "\n")

    f.write(str(T) + "\n")

    f.write(str(E) + "\n")

    f.write(str(F) + "\n")
np.savez("stereo_map.npz", Left_Stereo_Map[0], Left_Stereo_Map[1], Right_Stereo_Map[0], Right_Stereo_Map[1])

data = np.load("stereo_map.npz")

left_stereo_map_0 = data["arr_0"]
left_stereo_map_1 = data["arr_1"]
right_stereo_map_0 = data["arr_2"]
right_stereo_map_1 = data["arr_3"]




