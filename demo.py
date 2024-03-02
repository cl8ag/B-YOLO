from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import pandas as pd
import os
def load_stereo_maps(data_path):

    data = np.load(data_path)
    left_stereo_map = (data["arr_0"], data["arr_1"])
    right_stereo_map = (data["arr_2"], data["arr_3"])
    return left_stereo_map, right_stereo_map


def init_video_stream(video_path, width, height):

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def detect_objects(frame, model, left_map, right_map, corrected, conf, iou):

    height, width = frame.shape[:2]
    frameL = frame[:, :width // 2]
    frameR = frame[:, width // 2:]

    if corrected:
        frameL = cv2.remap(frameL, left_map[0], left_map[1], cv2.INTER_LINEAR)
        frameR = cv2.remap(frameR, right_map[0], right_map[1], cv2.INTER_LINEAR)
    start1 = time.time()
    results = model([frameL, frameR], conf=conf, iou=iou)
    end1 = time.time()
    frame_time1 = end1 - start1
    print(f"Frame Processing Time_yolo: {frame_time1:.2f} seconds")
    return results, frameL, frameR


def crop_detected_objects(results, frame):
    cropped_images = []
    for result in results:
        for box, conf, class_index in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box)
            name_list = result.names
            class_name = name_list[int(class_index)]
            cropped_image = frame[y_min:y_max, x_min:x_max]
            cropped_images.append((cropped_image, class_name, conf, (x_min, y_min, x_max, y_max)))

    return cropped_images

def extract_keypoints_and_descriptors(images, sift):
    keypoints_list, descriptors_list = [], []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        kp, des = sift.detectAndCompute(gray_image, None)

        keypoints_list.append(kp)
        descriptors_list.append(des)

    return keypoints_list, descriptors_list

#

def match_objects(descriptors_l, descriptors_r, cropped_images_l, cropped_images_r):
    global time_sift_sum_our
    global time_sift_sum
    global sum_diff_celue
    global sum_diff_jichu
    bf = cv2.BFMatcher()
    similarity_scores = []
    similarity_scores1 = []
    similarity_scores2 = []
    num_left = len(descriptors_l)
    num_right = len(descriptors_r)
    cost_matrix = np.ones((num_left, num_right))
    cost_matrix1 = np.ones((num_left, num_right))
    cost_matrix2 = np.ones((num_left, num_right))
    size_tolerance = 1020
    y_tolerance = 1010
    size_tolerance1 = 20
    y_tolerance1 = 10
    for i, desc_l in enumerate(descriptors_l):
        if desc_l is None:
            continue

        for j, desc_r in enumerate(descriptors_r):
            if desc_r is None:
                continue
            left_box = cropped_images_l[i][3]
            right_box = cropped_images_r[j][3]
            width_l, height_l = left_box[2] - left_box[0], left_box[3] - left_box[1]
            width_r, height_r = right_box[2] - right_box[0], right_box[3] - right_box[1]


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
        set2 = set(matched_pairs)
        set3 = set(matched_pairs2)
        diff = set2 - set3
        sum_diff_celue += len(diff)
    if matched_pairs1 != matched_pairs2:
        set2 = set(matched_pairs1)
        set3 = set(matched_pairs2)

        diff1 = set2 - set3
        sum_diff_jichu += len(diff1)
    return matched_pairs1,similarity_scores


def display_matched_pairs(matched_pairs, cropped_images_l, cropped_images_r):

    for i, j, score in matched_pairs:
        img_left = cropped_images_l[i]
        img_right = cropped_images_r[j]

        h_l, w_l, _ = img_left.shape
        h_r, w_r, _ = img_right.shape
        h_combined = max(h_l, h_r)
        combined_image = np.zeros((h_combined, w_l + w_r, 3), dtype=np.uint8)


        combined_image[:h_l, :w_l, :] = img_left

        combined_image[:h_r, w_l:w_l + w_r, :] = img_right


        cv2.putText(combined_image, f"Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Matched Pair - Left: {i}, Right: {j}", combined_image)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
def find_bright_spots(image, threshold):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)


    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
    cap = init_video_stream(r'', 3840, 1080)
    left_map, right_map = load_stereo_maps("stereo_map.npz")
    model = YOLO(r'.\best.pt')
    baseline = 59
    focal_length = 1392.591
    time_sift_sum=0
    time_sift_sum_our=0
    sum_diff_celue=0
    sum_diff_jichu=0
    # frame_count = 0
    # save_path = 'frames'
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # sift = cv2.SIFT_create(
    #     nfeatures=0,
    #     nOctaveLayers=3,
    #     contrastThreshold=0.02,
    #     edgeThreshold=5,
    #     sigma=1.2
    # )
    sift = cv2.ORB_create(
        nfeatures=1000,
        edgeThreshold=3,
        nlevels=12,
        patchSize=15
    )
    color_dict = {
        "green_apple": (0, 255, 0),
        "red_apple": (0, 0, 255),
        "orange": (0, 165, 255),
        "mango": (255, 200, 0)
    }

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        start = time.time()
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
                continue
            #display_matched_pairs(matched_pairs, [img[0] for img in cropped_images_l],
                                  #[img[0] for img in cropped_images_r])
            print(time.time())
            for i, j, _ in matched_pairs:

                left_box = cropped_images_l[i][3]
                right_box = cropped_images_r[j][3]

                class_name = cropped_images_l[i][1]

                center_x_left = (left_box[0] + left_box[2]) // 2

                center_x_right = (right_box[0] + right_box[2]) // 2
                difference= center_x_left-center_x_right

                box_color = color_dict.get(class_name, (255, 255, 255))

                #print(difference)
                if difference> 0:

                    depth = (focal_length * baseline) / difference
                    label = f"C:{class_name},D:{int(depth / 10)}cm"
                    print(label)


                    (text_width, text_height), baselines = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)


                    text_offset_x = left_box[0]
                    text_offset_y = left_box[1]
                    text_offset_y = max(text_height + baselines + 10, text_offset_y)

                    cv2.rectangle(frameL,
                                  (text_offset_x, text_offset_y - text_height - baselines),
                                  (text_offset_x + text_width, text_offset_y),
                                  box_color,
                                  cv2.FILLED)


                    cv2.putText(frameL,
                                label,
                                (text_offset_x, text_offset_y - baselines),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (255, 255, 255),
                                3)


                    cv2.rectangle(frameL,
                                  (left_box[0], left_box[1]),
                                  (left_box[2], left_box[3]),
                                  box_color,
                                  2)
            print(time.time())
        end = time.time()
        frame_time = end - start
        print(f"Frame Processing Time: {frame_time:.2f} seconds")
        cv2.imshow("Frame with Similar Boxes", frameL)
        # frame_file = os.path.join(save_path,
        # f"frame_{frame_count:05d}.jpg")
        # cv2.imwrite(frame_file, frameL)
        # frame_count += 1

        # threshold = 220
        #
        # bright_spots_l = find_bright_spots(frameL, threshold)
        # bright_spots_r = find_bright_spots(frameR, threshold)
        #
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

