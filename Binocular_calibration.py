# coding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                  单目及双目标定
#名称格式参考26，27行
#------------------------------------------------------------------------------------------------------------------------------------------#
# 设置迭代终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
objp = np.zeros((11 * 8, 3), np.float32)  # 8和5采用自己的标定板参数
objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)  # 8和5采用自己的标定板参数
# 用arrays存储所有图片的object points 和 image points
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []
# 假设棋盘格每个方块的真实世界单位长度，例如30毫米
square_size_mm = 25

# 计算尺度因子
# 初始化尺度因子和计数器
scale_factor_sum = 0
valid_images = 0
# 本次实验采集里共计30组待标定图片依次读入进行以下操作
for i in range(1, 17):
    t = str(i)
    ChessImaR = cv2.imread('./datas/r/' + t + '.jpg', 0)  # 右视图
    ChessImaL = cv2.imread('./datas/l/' + t + '.jpg', 0)  # 左视图
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (8, 11), None)  # 提取右图每一张图片的角点
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (8, 11), None)  # 提取左图每一张图片的角点
    if retR and retL:
        valid_images += 1
        # 使用左图像角点计算尺度因子
        pixel_size_w = np.linalg.norm(cornersL[0] - cornersL[4])  # 水平方向
        pixel_size_h = np.linalg.norm(cornersL[0] - cornersL[-5])  # 垂直方向
        average_pixel_size = (pixel_size_w / 4 + pixel_size_h / 7) / 2
        scale_factor = square_size_mm / average_pixel_size
        print(scale_factor)
        scale_factor_sum += scale_factor
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)  # 亚像素精确化，对粗提取的角点进行精确化
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)  # 亚像素精确化，对粗提取的角点进行精确化
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)
# 检查是否至少有一张有效图片
if valid_images > 0:
    #print(valid_images)
    # 计算所有图片尺度因子的平均值
    average_scale_factor = scale_factor_sum / valid_images
    print(f"Average scale factor: {average_scale_factor} mm/pixel")
else:
    print("No valid images found for scale factor calculation.")
# 相机的单双目标定、及校正
#   右侧相机单独标定
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)

#   获取新的相机矩阵后续传递给initUndistortRectifyMap，以用remap生成映射关系
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

#   左侧相机单独标定
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)

#   获取新的相机矩阵后续传递给initUndistortRectifyMap，以用remap生成映射关系
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# 双目相机的标定
# 设置标志位为cv2.CALIB_FIX_INTRINSIC，这样就会固定输入的cameraMatrix和distCoeffs不变，只求解𝑅,𝑇,𝐸,𝐹
flags = 0
# flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, OmtxL, distL, OmtxR,distR,ChessImaR.shape[::-1], criteria_stereo, flags)
print(retS)
print(T)
# 利用stereoRectify()计算立体校正的映射矩阵
rectify_scale = 0
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,ChessImaR.shape[::-1], R, T,rectify_scale, (0, 0))

# 利用initUndistortRectifyMap函数计算畸变矫正和立体校正的映射变换，实现极线对齐。
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)

Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)



#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                 展示极线校正的结果
#------------------------------------------------------------------------------------------------------------------------------------------#
# 立体校正效果显示
for i in range(1, 2):  # 以第一对图片为例
    t = str(i)
    frameR = cv2.imread('./datas/' + 'r' + '.jpg', 0)
    frameL = cv2.imread('./datas/' + 'l' + '.jpg', 0)

    Left_rectified = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                               0)  # 使用remap函数完成映射
    im_L = Image.fromarray(Left_rectified)  # numpy 转 image类

    Right_rectified = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR,
                                cv2.BORDER_CONSTANT, 0)
    im_R = Image.fromarray(Right_rectified)  # numpy 转 image 类

    # 创建一个能同时并排放下两张图片的区域，后把两张图片依次粘贴进去
    width = im_L.size[0] * 2
    height = im_L.size[1]

    img_compare = Image.new('RGB', (width, height))
    img_compare.paste(im_L, (0, 0))
    img_compare.paste(im_R, (im_L.size[0], 0))

    # 在已经极线对齐的图片上均匀画线
    for i in range(1, 20):
        len = 1080 / 20
        plt.axhline(y=i * len, color='r', linestyle='-')
    # 显示
    plt.imshow(img_compare)
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------#
#                                             计算重投影误差（观察标定的优劣）
#------------------------------------------------------------------------------------------------------------------------------------------#
# 重新定义 len 为内建函数
len = __builtins__.len
# 计算重投影误差
mean_error = 0
for i in range(len(objpoints)):
    # 在左右图像上重新投影
    imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)

    # 计算重投影误差
    errorR = cv2.norm(imgpointsR[i], imgpointsR2, cv2.NORM_L2) / len(imgpointsR[i])
    errorL = cv2.norm(imgpointsL[i], imgpointsL2, cv2.NORM_L2) / len(imgpointsL[i])
    mean_error += (errorR + errorL) / 2

# 计算平均重投影误差
total_error = mean_error / len(objpoints)
print("平均重投影误差: ", total_error)


#------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         保存标定结果
# 相机矩阵R/L (mtxR/mtxL)：这些是右侧和左侧相机的内部参数矩阵。它们包含了相机的焦距（在x和y轴上）和光学中心（也称为主点）的坐标。这些参数对于将相机中的3D点转换为2D图像中的像素坐标至关重要。
#
# 畸变系数R/L (distR/distL)：这些系数描述了右侧和左侧相机镜头的畸变情况。镜头畸变是由于镜头的形状而导致的图像扭曲。主要有径向畸变（物体看起来比实际更弯曲或更直）和切向畸变（由于镜头和成像平面不完全平行）。
#
# 光学矩阵R/L (OmtxR/OmtxL)：这些可能指的是校正后的相机矩阵，用于修正畸变并提供一个更“理想”的相机内部参数集。这有助于提高图像质量和立体视觉的准确度。
#
# ROI R/L (roiR/roiL)：ROI指的是“感兴趣区域”（Region of Interest）。在这个上下文中，它可能指校正图像后，图像中无畸变的区域。这对于裁剪或调整图像尺寸以消除畸变边缘可能很有用。
#
# 旋转矩阵 (R)：这个矩阵描述了两个相机之间的相对旋转。在立体视觉中，了解两个相机如何相对于彼此旋转是重要的，因为它影响了如何从两个视角中恢复3D信息。
#
# 平移矩阵 (T)：这描述了两个相机之间的相对位置或平移。它是立体视觉中确定物体深度和位置的关键要素。
#
# 基本矩阵 (E)：基本矩阵与立体相机对的几何属性有关，它结合了相机的内部参数（如焦距和主点）和相机之间的旋转和平移。这个矩阵通常用于立体视觉和3D重建。
#
# 要素矩阵 (F)：要素矩阵是从基本矩阵派生出的，它只考虑相机之间的相对位置和方向，而不考虑它们的内部属性。它在确保匹配点在对应线上时非常有用，是进行立体匹配和三维重建的关键步骤。
#
#------------------------------------------------------------------------------------------------------------------------------------------#
# 输出相机参数和校正映射结果到文件
with open("calibration_result.txt", "w") as f:
    # 写入相机矩阵R
    f.write("相机矩阵R：\n")
    f.write(str(mtxR) + "\n")
    # 写入畸变系数R
    f.write("畸变系数R：\n")
    f.write(str(distR) + "\n")
    # 写入光学矩阵R
    f.write("光学矩阵R：\n")
    f.write(str(OmtxR) + "\n")
    # 写入ROI R
    f.write("ROI R：\n")
    f.write(str(roiR) + "\n")
    f.write("\n")
    # 写入相机矩阵L
    f.write("相机矩阵L：\n")
    f.write(str(mtxL) + "\n")
    # 写入畸变系数L
    f.write("畸变系数L：\n")
    f.write(str(distL) + "\n")
    # 写入光学矩阵L
    f.write("光学矩阵L：\n")
    f.write(str(OmtxL) + "\n")
    # 写入ROI L
    f.write("ROI L：\n")
    f.write(str(roiL) + "\n")
    f.write("\n")
    # 写入旋转矩阵
    f.write("旋转矩阵：\n")
    f.write(str(R) + "\n")
    # 写入平移矩阵
    f.write("平移矩阵：\n")
    f.write(str(T) + "\n")
    # 写入基本矩阵
    f.write("基本矩阵：\n")
    f.write(str(E) + "\n")
    # 写入要素矩阵
    f.write("要素矩阵：\n")
    f.write(str(F) + "\n")
np.savez("stereo_map.npz", Left_Stereo_Map[0], Left_Stereo_Map[1], Right_Stereo_Map[0], Right_Stereo_Map[1])

#------------------------------------------------------------------------------------------------------------------------------------------#
#                                               读取stereo_map.npz文件中的数组
#------------------------------------------------------------------------------------------------------------------------------------------#
data = np.load("stereo_map.npz")
# 获取数组
left_stereo_map_0 = data["arr_0"]
left_stereo_map_1 = data["arr_1"]
right_stereo_map_0 = data["arr_2"]
right_stereo_map_1 = data["arr_3"]
# 打印数组
# print("Left Stereo Map 0:", left_stereo_map_0)
# print("Left Stereo Map 1:", left_stereo_map_1)
# print("Right Stereo Map 0:", right_stereo_map_0)
# print("Right Stereo Map 1:", right_stereo_map_1)

focal_length = mtxL[0, 0]     # 获取左侧相机的焦距
focal_length_r = mtxR[0, 0]
# print(mtxL)
# print(f'focal_length:{focal_length_r}')
# print(f'focal_length:{focal_length}')
# s=focal_length*average_scale_factor
# print(f's:{s}')
#
# print(Q)
# print(T)
print(mtxL)
print(mtxR)
print(distL)
print(distR)
print(R)
T=T*25
print(T)


