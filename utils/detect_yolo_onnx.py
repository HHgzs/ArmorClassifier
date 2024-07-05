import onnxruntime
import cv2
import numpy as np
import argparse
import yaml
import nms
import os
from tqdm import tqdm

def find_files(directory, extensions):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matched_files.append(os.path.join(root, file))
    return matched_files


def detect_once(config, image, session):

    model_class_num = config["model"]["class_num"]
    confidence_threshold = config["model"]["conf_thres"]
    iou_threshold = config["model"]["iou_thres"]
    infer_width = config["model"]["infer_width"]
    infer_height = config["model"]["infer_height"]
    nms_mode = config["model"]["mode"]
    class_map = config["output"]["class_map"]
    
    results = nms.detect(image, session, model_class_num, confidence_threshold, iou_threshold, infer_width, infer_height, nms_mode)
    
    if len(results) == 0:
        return None
    
    returns = []
    
    if config["output"]["format"] == "fp":
        for result in results:
            id = class_map[result.class_id]
            fp_list = [[x / image.shape[1], y / image.shape[0]] for x, y in result.four_points]
            fp_list = [item for sublist in fp_list for item in sublist]
            returns.append([id] + fp_list)
    
    elif config["output"]["format"] == "v5":
        for result in results:
            id = class_map[result.class_id]
            cx = result.box[0] / image.shape[1]
            cy = result.box[1] / image.shape[0]
            w  = result.box[2] / image.shape[1]
            h  = result.box[3] / image.shape[0]
            returns.append([id, cx, cy, w, h])

    else:
        raise ValueError("Unsupported output format")

    return returns




def display_once(config, image, detect_results):

    image_w = image.shape[1]
    image_h = image.shape[0]
    class_names = config["output"]["class_names"]
    
    for result in detect_results:
        
        if config["output"]["format"] == "fp":

            cv2.line(image, (int(result[1] * image_w), int(result[2] * image_h)), (int(result[3] * image_w), int(result[4] * image_h)), (0, 255, 0), 2)
            cv2.line(image, (int(result[3] * image_w), int(result[4] * image_h)), (int(result[5] * image_w), int(result[6] * image_h)), (0, 255, 0), 2)
            cv2.line(image, (int(result[5] * image_w), int(result[6] * image_h)), (int(result[7] * image_w), int(result[8] * image_h)), (0, 255, 0), 2)
            cv2.line(image, (int(result[7] * image_w), int(result[8] * image_h)), (int(result[1] * image_w), int(result[2] * image_h)), (0, 255, 0), 2)
            cv2.putText(image, class_names[result[0]], (int(result[1] * image_w), int(result[2] * image_h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
        elif config["output"]["format"] == "v5":
            rect_cx = int(result[1] * image_w)
            rect_cy = int(result[2] * image_h)
            rect_w = int(result[3] * image_w)
            rect_h = int(result[4] * image_h)
            cv2.rectangle(image, (rect_cx - rect_w // 2, rect_cy - rect_h // 2), (rect_cx + rect_w // 2, rect_cy + rect_h // 2), (0, 255, 0), 2)
            cv2.putText(image, class_names[result[0]], (rect_cx - rect_w // 2, rect_cy - rect_h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else:
            raise ValueError("Unsupported output format")
        
    return image
    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./utils/configs/config_detect_yolo_onnx.yaml', help='Path to config file')
    opts = parser.parse_args()
    
    config = None
    with open(opts.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    session = onnxruntime.InferenceSession(config["model"]["onnx"])
    
    if config["mode"] == "image":
        image_path = config["input_path"]
        image = cv2.imread(image_path)
        results = detect_once(config, image, session)
        if results is not None:
            output = display_once(config, image, results)
            cv2.imwrite(config["output_path"], output)
            print(results)
        
    
    
    elif config["mode"] == "video":
        video_path = config["input_path"]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        
        write_video = cv2.VideoWriter(config["output_path"], cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
        with tqdm(total=total_frames, unit='frame') as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = detect_once(config, frame, session)
                if results is not None:
                    output = display_once(config, frame, results)
                    write_video.write(output)
                else:
                    write_video.write(frame)
                    
                pbar.update(1)
        
        write_video.release()
        cap.release()
        


    elif config["mode"] == "video-dir":
        
        file_number = 100000
        video_dir = config["input_path"]
        files = find_files(video_dir, [".mp4", ".avi"])
        
        for file in tqdm(files):
            cap = cv2.VideoCapture(file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            with tqdm(total=total_frames, unit='frame') as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = detect_once(config, frame, session)
                    if results is not None:
                        file_number += 1
                        image_path = os.path.join(config["output_path"], f"{file_number}.jpg")
                        label_path = os.path.join(config["label_path"], f"{file_number}.txt")
                        
                        cv2.imwrite(image_path, frame)
                        with open(label_path, 'w') as f:
                            for result in results:
                                if config["output"]["format"] == "fp":
                                    f.write(f"{result[0]} {result[1]} {result[2]} {result[3]} {result[4]} {result[5]} {result[6]} {result[7]} {result[8]}\n")
                                elif config["output"]["format"] == "v5":
                                    f.write(f"{result[0]} {result[1]} {result[2]} {result[3]} {result[4]}\n")
                                else:
                                    raise ValueError("Unsupported output format")
                    pbar.update(1)
                    
            cap.release()
        
        
