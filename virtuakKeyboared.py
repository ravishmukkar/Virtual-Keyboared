import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from math import hypot
from time import sleep
# For webcam input:
cap = cv2.VideoCapture(0)
import cvzone
import numpy as np

cap.set(3,1288)
cap.set(4,720)


keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]
        ] 

FinalText = ""

def drawALL(image,buttonList):
    imgNew = np.zeros_like(image,np.uint8)
    for button in buttonList:
        x,y = button.pos
        w,h = button.size
        # cv2.rectangle(image,button.pos,(x+w,y+h),(255,0,255),cv2.FILLED)
        # cvzone.cornerRect(imgNew,(button.pos[0],button.pos[1],button.size[0],button.size[1]),20,rt=0)
        cv2.putText(image,button.text,(x+10,y+35),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),5)
    return image    

class Button():
    def __init__(self,pos,text,size=[50,50]):
        self.pos = pos
        self.size = size
        self.text = text
       
ButtonList = []

for i in range(len(keys)):    
    for j,key in enumerate(keys[i]):    
        ButtonList.append(Button([70*j+20,70*i+50],key)) 
    
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
  
    image = drawALL(image,ButtonList)
  
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      lmList = []
      for hand_landmarks in results.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy]) 
            mp_drawing.draw_landmarks(image, hand_landmarks,mp_hands.HAND_CONNECTIONS)
        
        if lmList != []:
            for button in ButtonList:
                x,y = button.pos
                w,h = button.size
                x2, y2 = lmList[8][1], lmList[8][2]
                x3, y3 = lmList[12][1], lmList[12][2]
                if x<lmList[8][1]<x+w and y<lmList[8][2]<y+h:
                    cv2.rectangle(image,button.pos,(x+w,y+h),(175,0,175),cv2.FILLED)
                    cv2.putText(image,button.text,(x+10,y+35),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0,.7),5)
                    
                    length = hypot(x3 - x2, y3 - y2)   
                    print(length) 
                    if length < 50:
                        cv2.rectangle(image,button.pos,(x+w,y+h),(0,255,0),cv2.FILLED)
                        cv2.putText(image,button.text,(x+10,y+35),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),5)
                        FinalText += button.text
                        sleep(0.15)
                        
            cv2.circle(image, (x2, y2), 15, (255, 0, 0), cv2.FILLED) 
            cv2.circle(image, (x3, y3), 15, (255, 0, 0), cv2.FILLED) 
    
    cv2.rectangle(image,(20,250),(600,308),(175,0,175),cv2.FILLED)
    cv2.putText(image,FinalText,(25,300),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0,.7),5)        
   
    cv2.imshow('virtual keyboared', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

