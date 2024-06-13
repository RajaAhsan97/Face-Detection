"""
    
"""


import os
import sys
import cv2
import numpy as np
import time
import argparse

pre_train_mod_fldr = "pre-trained model"

base_dir = os.path.dirname(__file__)

direc = os.listdir(base_dir)

for d in direc:
    if d == pre_train_mod_fldr:
        model_dir = direc[direc.index(d)]

models = os.listdir(model_dir)

model_path = os.path.join(base_dir, model_dir)

for model in models:
    m = os.path.join(model_path, model)
    if model == 'Haarcascade':
        haarcascade_pth = os.path.abspath(os.path.join(m, os.listdir(m)[0]))
    elif model == 'YuNet':
        YuNet_pth = os.path.abspath(os.path.join(m, os.listdir(m)[0]))


parser = argparse.ArgumentParser()
parser.add_argument("--Face_detector_YuNet", type=str, default=YuNet_pth, help="YuNet Face Detector model file, download from opencv zoo (github)")
parser.add_argument("--Face_detector_Haarcascade", type=str, default=haarcascade_pth, help="Haarcascade Face Detector model file")
parser.add_argument("--Yunet_score_threshold", type=float, default=0.85, help="Filter bounding boxes with score < score_threshold")
parser.add_argument("--Yunet_nms_threshold", type=float, default=0.3, help="Suppress k bounding boxes <= nms_threshold")
parser.add_argument("--Yunet_top_k", type=int, default=5000, help="Keep k bounding boxes before nms")
args = parser.parse_args()

"""
    YuNet Face Detector
"""

facedetector = cv2.FaceDetectorYN_create(args.Face_detector_YuNet,"", (0,0), args.Yunet_score_threshold, args.Yunet_nms_threshold, args.Yunet_top_k)

frame = cv2.imread('selfie.jpg')

frame_resol = np.shape(frame)
print("Original image resolution: ", frame_resol)

frame_cpy = frame.copy()


if facedetector.getInputSize() != frame_resol:
    # setInputSie (Img_w, Img_h)
    facedetector.setInputSize((frame_resol[1], frame_resol[0]))

# for YuNet detector the input images must have 3 channels 
start_time = time.time()
_, faces = facedetector.detect(frame)
end_time = time.time()
time_cons = end_time - start_time

# if no face is detected then initiate empty list, otherwise change the datatype
# to int
faces = faces if faces is not None else []
faces_detected = len(faces)

print("------------------YuNet------------------")
print("Time Consumption: ", np.round(time_cons, 3))
print("Faces Detected: ", faces_detected)


face_images_dir_nm = "yuNet faces"
os.mkdir(face_images_dir_nm)
face_images_dir_pth = os.path.abspath(os.path.join(base_dir, face_images_dir_nm))
face_count = 1
for face in faces:
    fc, confidence_score = list(face[:-1].astype('i')), str(face[-1])

    face_img_nm = str(face_count) + ".jpg"
    if fc[0] <= -1:
        pass
    else:
        face_crop = frame[fc[1]:fc[1]+fc[3], fc[0]:fc[0]+fc[2]]
        
    #cv2.imwrite(os.path.join(face_images_dir_pth, face_img_nm), face_crop)
    face_count += 1
    
    x1, x2 = int(fc[0]), int(fc[0]+fc[2])
    y1, y2 = int(fc[1]), int(fc[1]+fc[3])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)    

    # for landmarks in the image
    # right eye
    cv2.circle(frame, (fc[4], fc[5]), 1, (0,0,255), 2)
    # left eye
    cv2.circle(frame, (fc[6], fc[7]), 1, (0,0,255), 2)
    # nose tip
    cv2.circle(frame, (fc[8], fc[9]), 1, (0,255,255), 2)
    # right corner of mouth
    cv2.circle(frame, (fc[10], fc[11]), 1, (255,0,255), 2)
    # left corner of mouth
    cv2.circle(frame, (fc[12], fc[13]), 1, (255,0,255), 2)

##    # confidence
##    cv2.putText(frame, confidence_score, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255,255,255), 1)

cv2.imwrite("largest selfie (YuNet).jpg", frame)



"""
    Haarcascade Face Detector
"""

face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile(args.Face_detector_Haarcascade))

gray_frame = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2GRAY)

start_time = time.time()
face_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=2, minSize=(40,40), flags=cv2.CASCADE_SCALE_IMAGE)
end_time = time.time()
time_cons = end_time - start_time

faces_detected = len(face_rect)

print("---------------Haarcascade--------------")
print("Time Consumption: ", np.round(time_cons, 3))
print("Faces Detected: ", faces_detected)

face_images_dir_nm = "Haarcascade faces"
os.mkdir(face_images_dir_nm)
face_images_dir_pth = os.path.abspath(os.path.join(base_dir, face_images_dir_nm))
face_count = 1

for x1, y1, w, h in face_rect:
    face_img_nm = str(face_count) + ".jpg"

    face_crop = frame_cpy[y1:y1+h, x1:x1+w]
    cv2.imwrite(os.path.join(face_images_dir_pth, face_img_nm), face_crop)
    face_count += 1
    
    cv2.rectangle(frame_cpy, (x1,y1), (x1+w, y1+h), (0,255,0), 2)
    
cv2.imwrite("largest selfie (Haarcascade).jpg", frame_cpy)

