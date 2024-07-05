import numpy as np
import onnxruntime as ort
import cv2

# 设定静态变量
output_buffer_ = None
bboxes_num_ = 0
classes_num_ = 0
conf_thresh_ = 0.0
nms_thres_ = 0.0
input_w_ = 0
input_h_ = 0
infer_w_ = 0
infer_h_ = 0
infer_ratio_ = 0.0
yolo_size = 0

class YoloRect:
    def __init__(self, confidence, class_id, box, four_points=None, color_id=None):
        self.confidence = confidence
        self.class_id = class_id
        self.box = box
        self.four_points = four_points if four_points is not None else []
        self.color_id = color_id

def nms_set_ratio():
    global infer_ratio_
    width_ratio = input_w_ / infer_w_
    height_ratio = input_h_ / infer_h_

    if width_ratio > height_ratio:
        infer_ratio_ = width_ratio
    else:
        infer_ratio_ = height_ratio

def nms_get_rect_v5(yolo_raw):
    x, y, w, h = yolo_raw[:4]

    return (int(round(x * infer_ratio_)), int(round(y * infer_ratio_)), int(round(w * infer_ratio_)), int(round(h * infer_ratio_)))

def nms_get_rect_fp(yolo_raw):
    min_x = min(yolo_raw[0], yolo_raw[2], yolo_raw[4], yolo_raw[6])
    max_x = max(yolo_raw[0], yolo_raw[2], yolo_raw[4], yolo_raw[6])
    min_y = min(yolo_raw[1], yolo_raw[3], yolo_raw[5], yolo_raw[7])
    max_y = max(yolo_raw[1], yolo_raw[3], yolo_raw[5], yolo_raw[7])

    left = min_x * infer_ratio_
    top = min_y * infer_ratio_
    width = (max_x - min_x) * infer_ratio_
    height = (max_y - min_y) * infer_ratio_

    return (int(round(left)), int(round(top)), int(round(width)), int(round(height)))

def nms_get_fp(yolo_raw):
    x_index = [0, 2, 4, 6]
    y_index = [1, 3, 5, 7]
    four_points = []

    for i in range(4):
        x = yolo_raw[x_index[i]] * infer_ratio_
        y = yolo_raw[y_index[i]] * infer_ratio_
        if x < 0 or x >= input_w_ or y < 0 or y >= input_h_:
            return []
        four_points.append((x, y))
    return four_points

def nms_calcu_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    over_area = w * h
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - over_area + 1e-5

    return over_area / union_area

def nms_select_confidence_fp(detection_list):

    for i in range(bboxes_num_):
        yolo_raw = output_buffer_[i]
        iou_confidence = yolo_raw[8]
        if iou_confidence < conf_thresh_:
            continue

        class_index = -1
        class_confidence = 0
        for j in range(classes_num_):
            confidence = yolo_raw[9 + j] * iou_confidence
            if confidence > class_confidence and confidence > conf_thresh_:
                class_index = j
                class_confidence = confidence

        if class_index == -1:
            continue

        flag_pose = True
        for k in range(4):
            if (yolo_raw[2 * k] < 1e-3 or yolo_raw[2 * k] > (infer_w_ - 1.001) or
                yolo_raw[2 * k + 1] < 1e-3 or yolo_raw[2 * k + 1] > (infer_h_ - 1.001)):
                flag_pose = False
                break

        if not flag_pose:
            continue

        detection_rect = YoloRect(
            confidence=class_confidence,
            class_id=class_index,
            box=nms_get_rect_fp(yolo_raw),
            four_points=nms_get_fp(yolo_raw)
        )

        if len(detection_rect.four_points) != 4:
            continue

        detection_list.append(detection_rect)

def nms_select_confidence_v5(detection_list):
    for i in range(bboxes_num_):
        yolo_raw = output_buffer_[i]
        iou_confidence = yolo_raw[4]
        if iou_confidence < conf_thresh_:
            continue

        class_index = -1
        class_confidence = 0
        for j in range(classes_num_):
            confidence = yolo_raw[5 + j] * iou_confidence
            if confidence > class_confidence and confidence > conf_thresh_:
                class_index = j
                class_confidence = confidence

        if class_index == -1:
            continue

        detection_rect = YoloRect(
            confidence=class_confidence,
            class_id=class_index,
            box=nms_get_rect_v5(yolo_raw)
        )

        detection_list.append(detection_rect)

def nms_select_confidence_fpx(detection_list):

    for i in range(bboxes_num_):
        yolo_raw = output_buffer_[i]
        iou_confidence = yolo_raw[8]
        if iou_confidence < conf_thresh_:
            continue

        color_index = -1
        color_confidence = 0
        for j in range(4):
            confidence = yolo_raw[9 + j] * iou_confidence
            if confidence > color_confidence and confidence > conf_thresh_:
                color_index = j
                color_confidence = confidence

        if color_index == -1:
            continue

        class_index = -1
        class_confidence = 0
        for j in range(classes_num_):
            confidence = yolo_raw[13 + j] * iou_confidence
            if confidence > class_confidence and confidence > conf_thresh_:
                class_index = j
                class_confidence = confidence

        if class_index == -1:
            continue

        flag_pose = True
        for k in range(4):
            if (yolo_raw[2 * k] < 1e-3 or yolo_raw[2 * k] > (infer_w_ - 1.001) or
                yolo_raw[2 * k + 1] < 1e-3 or yolo_raw[2 * k + 1] > (infer_h_ - 1.001)):
                flag_pose = False
                break

        if not flag_pose:
            continue

        detection_rect = YoloRect(
            confidence=class_confidence,
            class_id=class_index,
            color_id=color_index,
            box=nms_get_rect_fp(yolo_raw),
            four_points=nms_get_fp(yolo_raw)
        )

        if len(detection_rect.four_points) != 4:
            continue

        detection_list.append(detection_rect)

def nms_sort_confidence(detection_list):
    detection_list.sort(key=lambda x: x.confidence, reverse=True)

def nms_select_iou(detection_list):
    if len(detection_list) <= 1:
        return

    retained_list = []

    retained_list.append(detection_list[0])

    for focus_index in range(1, len(detection_list)):
        focus_rect = detection_list[focus_index]
        available_rect = True

        for retained_rect in retained_list:
            iou = nms_calcu_iou(focus_rect.box, retained_rect.box)
            if iou > nms_thres_:
                available_rect = False
                break

        if available_rect:
            retained_list.append(focus_rect)

    detection_list[:] = retained_list

def yolo_armor_nms_fp():
    detection_list = []
    nms_select_confidence_fp(detection_list)
    nms_sort_confidence(detection_list)
    nms_select_iou(detection_list)

    return detection_list

def yolo_armor_nms_v5():
    detection_list = []
    nms_select_confidence_v5(detection_list)
    nms_sort_confidence(detection_list)
    nms_select_iou(detection_list)

    return detection_list

def yolo_armor_nms_fpx():
    detection_list = []
    nms_select_confidence_fpx(detection_list)
    nms_sort_confidence(detection_list)
    nms_select_iou(detection_list)

    return detection_list

def nms(model_output, cls_num, confidence_thresh, nms_thresh, in_width, in_height, inf_width, inf_height, mode='fp'):
    global output_buffer_, bboxes_num_, classes_num_, conf_thresh_, nms_thres_, input_w_, input_h_, infer_w_, infer_h_, yolo_size

    bboxes_num_ = model_output.shape[1]
    yolo_size = model_output.shape[2]
    output_buffer_ = model_output.reshape((bboxes_num_, yolo_size))
    
    classes_num_ = cls_num
    conf_thresh_ = confidence_thresh
    nms_thres_ = nms_thresh
    input_w_ = in_width
    input_h_ = in_height
    infer_w_ = inf_width
    infer_h_ = inf_height

    if mode == 'fp':
        return yolo_armor_nms_fp()
    elif mode == 'v5':
        return yolo_armor_nms_v5()
    elif mode == 'fpx':
        return yolo_armor_nms_fpx()
    else:
        raise ValueError("Invalid mode. Choose from 'fp', 'v5', or 'fpx'")

def detect(input_image, session, class_num, conf_thresh, iou_thresh, infer_width = 640, infer_height = 640, mode='fp'):
    global input_w_, input_h_, infer_w_, infer_h_

    input_w_ = input_image.shape[1]
    input_h_ = input_image.shape[0]
    infer_w_ = infer_width
    infer_h_ = infer_height
    
    nms_set_ratio()

    input_data = cv2.resize(input_image, ((int)(input_w_ / infer_ratio_), (int)(input_h_ / infer_ratio_)))
    input_data = cv2.copyMakeBorder(input_data, 0, infer_h_ - input_data.shape[0], 0, infer_w_ - input_data.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float32) / 255.0
    
    output = session.run(None, {"images": input_data})[0]
    
    results = nms(output, class_num, conf_thresh, iou_thresh, input_w_, input_h_, infer_w_, infer_h_, mode)
    return results



if __name__ == "__main__":
    model_path = "/mnt/d/Project/ArmorClassifier/outputs/4p_4c9c_640_640_RP24.onnx"
    image_path = "/mnt/d/Project/ArmorClassifier/000338.jpg"
    output_path = "/mnt/d/Project/ArmorClassifier/out.jpg"
    
    input_image = cv2.imread(image_path)

    session = ort.InferenceSession(model_path)
    results = detect(input_image, session, 9, 0.5, 0.4, 640, 640, mode='fpx')
    
    print(f"检测到 {len(results)} 个目标")

    for rect in results:
        print(f"类别: {rect.class_id}, 置信度: {rect.confidence:.2f}, 边界框: {rect.box}")
        
        if rect.four_points == []:
            cv2.rectangle(input_image, (rect.box[0], rect.box[1]), (rect.box[0] + rect.box[2], rect.box[1] + rect.box[3]), (0, 255, 0), 2)
        else:
            cv2.line(input_image, (int(rect.four_points[0][0]), int(rect.four_points[0][1])), (int(rect.four_points[1][0]), int(rect.four_points[1][1])), (0, 255, 0), 2)
            cv2.line(input_image, (int(rect.four_points[1][0]), int(rect.four_points[1][1])), (int(rect.four_points[2][0]), int(rect.four_points[2][1])), (0, 255, 0), 2)
            cv2.line(input_image, (int(rect.four_points[2][0]), int(rect.four_points[2][1])), (int(rect.four_points[3][0]), int(rect.four_points[3][1])), (0, 255, 0), 2)
            cv2.line(input_image, (int(rect.four_points[3][0]), int(rect.four_points[3][1])), (int(rect.four_points[0][0]), int(rect.four_points[0][1])), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, input_image)
