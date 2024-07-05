import cv2
import os

video1 = '/mnt/d/Project/ArmorClassifier/out_v5.avi'
video2 = '/mnt/d/Project/ArmorClassifier/out_rp.avi'

cap1 = cv2.VideoCapture(video1)
fps1 = cap1.get(cv2.CAP_PROP_FPS)

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap2 = cv2.VideoCapture(video2)
fps2 = cap2.get(cv2.CAP_PROP_FPS)

width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 1, (width1 + width2, max(height1, height2)))

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break
    
    ret2, frame2 = cap2.read()
    if not ret2:
        break
    
    combined_frame = cv2.hconcat([frame1, frame2])
    out.write(combined_frame)

cap1.release()
cap2.release()
out.release()