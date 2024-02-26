from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import pandas as pd
import os
def load_stereo_maps(data_path):
    """
        加载立体相机数据。

        Args:
            data_path (str): 存储立体相机数据的文件路径。

        Returns:
            tuple: 包含左侧和右侧立体相机参数。
    """
    data = np.load(data_path)
    left_stereo_map = (data["arr_0"], data["arr_1"])
    right_stereo_map = (data["arr_2"], data["arr_3"])
    return left_stereo_map, right_stereo_map


def init_video_stream(video_path, width, height):
    """
        初始化视频流。

        Args:
            video_path (str): 视频文件路径。
            width (int): 帧的宽度。
            height (int): 帧的高度。

        Returns:
            cv2.VideoCapture: 初始化的视频捕捉对象。
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def detect_objects(frame, model, left_map, right_map, corrected, conf, iou):
    """
       使用 YOLO 模型检测图像中的对象。

       Args:
           frame (numpy.ndarray): 输入图像帧。
           model: YOLO 模型。
           left_map: 左侧摄像头的立体地图。
           right_map: 右侧摄像头的立体地图。
           corrected (bool): 是否校正图像。
           conf (float): 置信度阈值。
           iou (float): IoU 阈值。

       Returns:
           tuple: 包含检测结果、左侧图像和右侧图像的元组。
    """
    height, width = frame.shape[:2]
    frameL = frame[:, :width // 2]
    frameR = frame[:, width // 2:]

    if corrected:
        frameL = cv2.remap(frameL, left_map[0], left_map[1], cv2.INTER_LINEAR)
        frameR = cv2.remap(frameR, right_map[0], right_map[1], cv2.INTER_LINEAR)
    start1 = time.time()  # 开始时间
    results = model([frameL, frameR], conf=conf, iou=iou)
    end1 = time.time()  # 结束时间
    frame_time1 = end1 - start1  # 计算帧处理时间
    print(f"Frame Processing Time_yolo: {frame_time1:.2f} seconds")
    return results, frameL, frameR


def crop_detected_objects(results, frame):
    """
        裁剪检测到的对象并提取它们的类别和置信度。

        Args:
            results: YOLO 检测结果。
            frame (numpy.ndarray): 输入图像帧。

        Returns:
            tuple: 包含裁剪的对象、检测到的类别和其他信息的元组。
    """

    cropped_images = []
    for result in results:
        for box, conf, class_index in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box)
            name_list = result.names
            class_name = name_list[int(class_index)]  # 假设您有一个包含类别名称的列表name_list
            cropped_image = frame[y_min:y_max, x_min:x_max]
            cropped_images.append((cropped_image, class_name, conf, (x_min, y_min, x_max, y_max)))

    return cropped_images

def extract_keypoints_and_descriptors(images, sift):
    """
       提取图像中的关键点和描述符。

       Args:
           images (list): 输入图像列表。
           sift: SIFT 特征提取器对象。

       Returns:
           tuple: 包含关键点列表和描述符列表的元组。
    """
    keypoints_list, descriptors_list = [], []
    for image in images:
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测关键点和计算描述符
        kp, des = sift.detectAndCompute(gray_image, None)

        # 将关键点和描述符添加到列表中
        keypoints_list.append(kp)
        descriptors_list.append(des)

    return keypoints_list, descriptors_list

#

def match_objects(descriptors_l, descriptors_r, cropped_images_l, cropped_images_r):
    """
       使用描述符匹配左侧和右侧的裁剪图像。

       Args:
           descriptors_l (list): 左侧图像的描述符列表。
           descriptors_r (list): 右侧图像的描述符列表。

       Returns:
           list: 匹配的对象对列表，每个对象包含左侧索引、右侧索引和匹配得分。
    """
    global time_sift_sum_our
    global time_sift_sum
    global sum_diff_celue
    global sum_diff_jichu
    bf = cv2.BFMatcher()
    similarity_scores = []  # 新增：用于收集 similarity_score
    similarity_scores1 = []  # 新增：用于收集 similarity_score
    similarity_scores2 = []  # 新增：用于收集 similarity_score
    num_left = len(descriptors_l)
    num_right = len(descriptors_r)
    cost_matrix = np.ones((num_left, num_right))  # 初始化成本矩阵
    cost_matrix1 = np.ones((num_left, num_right))  # 初始化成本矩阵
    cost_matrix2 = np.ones((num_left, num_right))  # 初始化成本矩阵
    size_tolerance = 1020  # 尺寸容差，例如：20像素
    y_tolerance = 1010  # Y坐标容差
    size_tolerance1 = 20  # 尺寸容差，例如：20像素
    y_tolerance1 = 10  # Y坐标容差

    for i, desc_l in enumerate(descriptors_l):
        if desc_l is None:
            continue

        for j, desc_r in enumerate(descriptors_r):
            if desc_r is None:
                continue
            left_box = cropped_images_l[i][3]  # (x_min, y_min, x_max, y_max)
            right_box = cropped_images_r[j][3]  # (x_min, y_min, x_max, y_max)

            # 计算两个边界框的尺寸
            width_l, height_l = left_box[2] - left_box[0], left_box[3] - left_box[1]
            width_r, height_r = right_box[2] - right_box[0], right_box[3] - right_box[1]

            # 检查尺寸和 Y 坐标是否在容差内
            if (cropped_images_l[i][1] == cropped_images_r[j][1] and
                    abs(width_l - width_r) <= size_tolerance and
                    abs(height_l - height_r) <= size_tolerance and
                    abs(left_box[1] - right_box[1]) <= y_tolerance and
                    abs(left_box[3] - right_box[3]) <= y_tolerance):

                desc_l = desc_l.astype(np.float32)
                desc_r = desc_r.astype(np.float32)
                matches = bf.knnMatch(desc_l, desc_r, k=2)


                good_matches = []
                for match_pair in matches:
                    if len(match_pair) >= 2:
                        m, n = match_pair
                        if hasattr(n, 'distance'):
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)

                similarity_score = len(good_matches) / max(len(desc_l), len(desc_r))
                similarity_scores.append(len(good_matches))
                cost_matrix[i, j] = 1 - similarity_score


            if (cropped_images_l[i][1] == cropped_images_r[j][1] and
                    abs(width_l - width_r) <= size_tolerance1 and
                    abs(height_l - height_r) <= size_tolerance1 and
                    abs(left_box[1] - right_box[1]) <= y_tolerance1 and
                    abs(left_box[3] - right_box[3]) <= y_tolerance1):
                time_sift_our = time.time()
                desc_l = desc_l.astype(np.float32)
                desc_r = desc_r.astype(np.float32)
                matches2 = bf.knnMatch(desc_l, desc_r, k=2)
                time_sift_our = time.time() - time_sift_our
                time_sift_sum_our += time_sift_our

                good_matches2 = []
                for match_pair2 in matches2:
                    if len(match_pair2) >= 2:
                        m, n = match_pair2
                        if hasattr(n, 'distance'):
                            if m.distance < 0.7 * n.distance:
                                good_matches2.append(m)

                similarity_score2 = len(good_matches2) / max(len(desc_l), len(desc_r))
                similarity_scores2.append(len(good_matches2))
                cost_matrix2[i, j] = 1 - similarity_score2



            time_sift=time.time()
            desc_l = desc_l.astype(np.float32)
            desc_r = desc_r.astype(np.float32)
            matches1 = bf.knnMatch(desc_l, desc_r, k=2)
            time_sift=time.time()-time_sift
            time_sift_sum+=time_sift
            good_matches1 = []
            for match_pair1 in matches1:
                if len(match_pair1) >= 2:
                    m1, n1 = match_pair1
                    if hasattr(n1, 'distance'):
                        if m1.distance < 0.7 * n1.distance:
                            good_matches1.append(m1)

            similarity_score1 = len(good_matches1) / max(len(desc_l), len(desc_r))
            similarity_scores1.append(len(good_matches1))
            cost_matrix1[i, j] = 1 - similarity_score1


    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_pairs = [(row, col, 1 - cost_matrix[row, col]) for row, col in zip(row_ind, col_ind) if
                     cost_matrix[row, col] < 0.9]

    row_ind1, col_ind1 = linear_sum_assignment(cost_matrix1)
    matched_pairs1 = [(row1, col1, 1 - cost_matrix1[row1, col1]) for row1, col1 in zip(row_ind1, col_ind1) if
                     cost_matrix1[row1, col1] < 0.9]

    row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
    matched_pairs2 = [(row2, col2, 1 - cost_matrix2[row2, col2]) for row2, col2 in zip(row_ind2, col_ind2) if
                      cost_matrix2[row2, col2] < 0.9]
    if matched_pairs!=matched_pairs2:
        print("结果不一致")
        set2 = set(matched_pairs)
        set3 = set(matched_pairs2)
        # 计算两个集合的差集
        diff = set2 - set3
        sum_diff_celue += len(diff)
    if matched_pairs1 != matched_pairs2:
        print("结果不一致")
        set2 = set(matched_pairs1)
        set3 = set(matched_pairs2)
        # 计算两个集合的差集
        diff1 = set2 - set3
        sum_diff_jichu += len(diff1)
    return matched_pairs1,similarity_scores


def display_matched_pairs(matched_pairs, cropped_images_l, cropped_images_r):
    """
        显示匹配的对象对和相似度得分。

        Args:
            matched_pairs (list): 匹配的对象对列表。
            cropped_images_l (list): 左侧裁剪图像列表。
            cropped_images_r (list): 右侧裁剪图像列表。
    """
    for i, j, score in matched_pairs:
        img_left = cropped_images_l[i]
        img_right = cropped_images_r[j]
        # 将左右图像并排放置在一起
        h_l, w_l, _ = img_left.shape
        h_r, w_r, _ = img_right.shape
        h_combined = max(h_l, h_r)
        combined_image = np.zeros((h_combined, w_l + w_r, 3), dtype=np.uint8)

        # 放置左图
        combined_image[:h_l, :w_l, :] = img_left
        # 放置右图
        combined_image[:h_r, w_l:w_l + w_r, :] = img_right

        # 显示匹配对及得分
        cv2.putText(combined_image, f"Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Matched Pair - Left: {i}, Right: {j}", combined_image)

        # 按下任意键后继续
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def find_bright_spots(image, threshold):
    # 转换图像为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值化操作获取亮点掩膜
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 进行形态学操作（可选）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓并获取中心坐标
    bright_spots = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bright_spots.append((cx, cy))

    return bright_spots
def main():
    global time_sift_sum_our
    global time_sift_sum
    global sum_diff_jichu
    global sum_diff_celue
    cap = init_video_stream(r'原视频.mp4', 3840, 1080)
    left_map, right_map = load_stereo_maps("stereo_map.npz")
    model = YOLO(r'E:\ultralytics-main\ultralytics\best.pt')
    baseline = 59  # 相机之间的距离
    focal_length = 1392.591  # 相机的焦距
    time_sift_sum=0
    time_sift_sum_our=0
    sum_diff_celue=0
    sum_diff_jichu=0
    # frame_count = 0
    # save_path = 'frames'  # 保存帧的文件夹
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #创建SIFT对象
    # sift = cv2.SIFT_create(
    #     nfeatures=0,  # 最大特征点数，0表示不限制
    #     nOctaveLayers=3,  # 金字塔每组的层数
    #     contrastThreshold=0.02,  # 降低对比度阈值
    #     edgeThreshold=5,  # 降低边缘阈值
    #     sigma=1.2  # 调整高斯滤波的标准差
    # )
    sift = cv2.ORB_create(
        nfeatures=1000,
        edgeThreshold=3,
        nlevels=12,
        patchSize=15
    )
    color_dict = {
        "green_apple": (0, 255, 0),  # 绿色
        "red_apple": (0, 0, 255),  # 红色
        "orange": (0, 165, 255),  # 橙色
        "mango": (255, 200, 0)  # 芒果色
    }

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        start = time.time()  # 开始时间
        results, frameL, frameR = detect_objects(frame, model, left_map, right_map, True, 0.4, 0.7)
        cropped_images_l = crop_detected_objects([results[0]], frameL)
        cropped_images_r = crop_detected_objects([results[1]], frameR)
        print(time.time())
        if len(cropped_images_l) and len(cropped_images_r):

            keypoints_l, descriptors_l = extract_keypoints_and_descriptors([img[0] for img in cropped_images_l],
                                                                           sift)
            keypoints_r, descriptors_r = extract_keypoints_and_descriptors([img[0] for img in cropped_images_r],
                                                                           sift)
            matched_pairs, similarity_scores = match_objects(descriptors_l, descriptors_r,cropped_images_l,cropped_images_r)
            if not matched_pairs:
                print("No matched pairs found.")
                continue  # 跳过当前循环的剩余部分
            #display_matched_pairs(matched_pairs, [img[0] for img in cropped_images_l],
                                  #[img[0] for img in cropped_images_r])
            print(time.time())
            for i, j, _ in matched_pairs:
                # 提取左侧和右侧边界框的坐标
                left_box = cropped_images_l[i][3]
                right_box = cropped_images_r[j][3]
                # 使用类名作为标签
                class_name = cropped_images_l[i][1]
                # 计算左侧边界框的中心X坐标
                center_x_left = (left_box[0] + left_box[2]) // 2
                # 计算右侧边界框的中心X坐标
                center_x_right = (right_box[0] + right_box[2]) // 2
                difference= center_x_left-center_x_right
                # 选择边界框颜色
                box_color = color_dict.get(class_name, (255, 255, 255))  # 如果类别未定义，默认为白色

                #print(difference)
                if difference> 0:
                    # 计算深度值
                    # 计算深度值
                    depth = (focal_length * baseline) / difference
                    label = f"C:{class_name},D:{int(depth / 10)}cm"  # 格式化为字符串
                    print(label)

                    # 获取文本尺寸
                    (text_width, text_height), baselines = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

                    # 计算文本背景矩形的顶部左侧坐标
                    text_offset_x = left_box[0]
                    text_offset_y = left_box[1]  # 将文本向上移动，确保不与边界框线重合

                    # 确保文本背景矩形不会超出图像边界
                    text_offset_y = max(text_height + baselines + 10, text_offset_y)  # 防止文本绘制在图像外

                    # 绘制文本的背景矩形
                    cv2.rectangle(frameL,
                                  (text_offset_x, text_offset_y - text_height - baselines),
                                  (text_offset_x + text_width, text_offset_y),
                                  box_color,
                                  cv2.FILLED)

                    # 在边界框上添加类别标签和深度信息
                    cv2.putText(frameL,
                                label,
                                (text_offset_x, text_offset_y - baselines),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (255, 255, 255),  # 字体颜色改为白色
                                3)

                    # 绘制边界框
                    cv2.rectangle(frameL,
                                  (left_box[0], left_box[1]),
                                  (left_box[2], left_box[3]),
                                  box_color,
                                  2)
            print(time.time())
        end = time.time()  # 结束时间
        frame_time = end - start  # 计算帧处理时间
        print(f"Frame Processing Time: {frame_time:.2f} seconds")
        cv2.imshow("Frame with Similar Boxes", frameL)
        # frame_file = os.path.join(save_path,
        #                           f"frame_{frame_count:05d}.jpg")  # 命名方式: frame_00001.jpg, frame_00002.jpg, 等
        # cv2.imwrite(frame_file, frameL)
        # frame_count += 1


        # # 设置阈值
        # threshold = 220
        #
        # # 调用函数获取亮点的坐标
        # bright_spots_l = find_bright_spots(frameL, threshold)
        # bright_spots_r = find_bright_spots(frameR, threshold)
        #
        # # 绘制亮点并打印坐标
        # #print("Bright spots in the left frame:")
        # for point1 in bright_spots_l:
        #     cv2.circle(frameL, point1, 5, (0, 0, 255), -1)
        #     print("Coordinate:", point1)
        #
        # #print("Bright spots in the right frame:")
        # for point2 in bright_spots_r:
        #     cv2.circle(frameR, point2, 5, (0, 0, 255), -1)
        #     print("Coordinate:", point2)
        # print(int((59.51 * 1531.85) / (point2[0] - point1[0])/10))
        #
        # # 显示图像
        # cv2.imshow("Left Frame with Bright Spots", frameL)
        # cv2.imshow("Right Frame with Bright Spots", frameR)



        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print(time_sift_sum_our)
    print(time_sift_sum)
    print(sum_diff_celue)
    print(sum_diff_jichu)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

