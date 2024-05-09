import cv2 as cv
import numpy as np

cap = cv.VideoCapture("outputvideo.mp4")

frame_buffer = []

while len(frame_buffer) < 10:
    ret, frame = cap.read()
    frame = cv.resize(frame, (224, 224), interpolation=cv.INTER_LINEAR)
    frame_buffer.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

while(1):
    ret, frame = cap.read()
    frame = cv.resize(frame, (224, 224), interpolation=cv.INTER_LINEAR)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_buffer.append(gray)
    prvs = frame_buffer.pop(0)
    prvs = cv.GaussianBlur(prvs, (25,25), 0)
    next = cv.GaussianBlur(gray, (25,25), 0)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame)
        cv.imwrite('opticalhsv.png',bgr)

cap.release()
cv.destroyAllWindows()