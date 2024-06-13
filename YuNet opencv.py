import os
import sys
import cv2
import numpy as np

pre_train_mod_fldr = "pre-trained model"

base_dir = os.path.dirname(__file__)

direc = os.listdir(base_dir)

for d in direc:
    if d == pre_train_mod_fldr:
        model_dir = direc[direc.index(d)]

model_pth = os.path.abspath(os.path.join(base_dir, os.listdir(model_dir)[-2]))


facedetector = cv2.FaceDetectorYN_create(model_pth,"", (0,0))


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to open camera for the given port....")
    sys.exit(0)

    
while True:
    ret, frame = cap.read()
    frame_resol = np.shape(frame)

    if facedetector.getInputSize() != frame_resol:
        # setInputSie (Img_w, Img_h)
        facedetector.setInputSize((frame_resol[1], frame_resol[0]))

    # for YuNet detector the input images must have 3 channels 
    no_faces, faces = facedetector.detect(frame)

    # if no face is detected then initiate empty list, otherwise change the datatype
    # to int
    faces = faces.astype('i') if faces is not None else []

    for face in faces:
        fc = list(face)
        x1, x2 = int(fc[0]), int(fc[0]+fc[2])
        y1, y2 = int(fc[1]), int(fc[1]+fc[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)    

        # for landmarks in the image
        # right eye
        cv2.circle(frame, (fc[4], fc[5]), 3, (0,0,255), 1)
        # left eye
        cv2.circle(frame, (fc[6], fc[7]), 3, (0,0,255), 1)
        # nose tip
        cv2.circle(frame, (fc[8], fc[9]), 3, (0,255,255), 1)
        # right corner of mouth
        cv2.circle(frame, (fc[10], fc[11]), 3, (255,0,255), 1)
        # left corner of mouth
        cv2.circle(frame, (fc[12], fc[13]), 3, (255,0,255), 1)

        # confidence
        print(fc[14])
        cv2.putText(frame, str(fc[14]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
    cv2.imshow("Webcam", frame)


    
    # ascii code for ESC is 27
    if cv2.waitKey(1) == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
