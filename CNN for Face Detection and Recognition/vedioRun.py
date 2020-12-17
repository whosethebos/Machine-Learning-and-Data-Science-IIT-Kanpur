import numpy as np
import cv2

def vedioRun(vedio_path):
    cap = cv2.VideoCapture(vedio_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow('frame', frame)
            # & 0xFF is required for a 64-bit system
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

