import cv2
import mediapipe as mp
import numpy as np
from Handutils import HandDetector
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640,480

camera = cv2.VideoCapture(0)
camera.set(3,wCam)
camera.set(4,hCam)
hand_detector = HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
volValue = 0

while True:
    success, img = camera.read()
    if success:
        img = cv2.flip(img, 1)
        hand_detector.process(img, draw=True)
        position = hand_detector.find_position(img)
        left_finger = position['Left'].get(8, None)
        left_thumb = position['Left'].get(4, None)
        right_finger = position['Right'].get(8, None)
        right_thumb = position['Right'].get(4, None)
        if left_finger:
            cv2.circle(img, (left_finger[0], left_finger[1]), 10, (0, 0, 255), cv2.FILLED)
            x1, y1 = left_finger
            x2, y2 = left_thumb
            cx, cy = (x1+x2)//2, (y1+y2)//2
        
            length = math.hypot(x2-x1, y2-y1)
            #HandRange 30~220
            #VolRange -63.5~0
            vol = np.interp(length,[30,300],[minVol,maxVol])
            volBar = np.interp(length,[30,300],[400,150])
            volValue = np.interp(length,[30,300],[0,100])
            cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
            volume.SetMasterVolumeLevel(vol, None)
            cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
            
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (left_thumb[0], left_thumb[1]), 10, (0, 0, 255), cv2.FILLED)
        if right_finger:
            cv2.circle(img, (right_finger[0], right_finger[1]), 10, (0, 0, 255), cv2.FILLED)
        cv2.imshow('Video', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()