import cv2
import time
import mediapipe as mp
import fingerutils
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

WINDOW_NAME = "Gestures"

class handDetector():
    def __init__(self, mode = False,model_complexity=1 ,maxhands = 2, detectioncon = 0.5, trackcon = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.model_complex = model_complexity
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxhands, self.model_complex,self.detectioncon, self.trackcon)
        self.mpdraw = mp.solutions.drawing_utils

    def findhands(self, img, draw = True) :
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgrgb)

        if self.result.multi_hand_landmarks :
            for handlms in self.result.multi_hand_landmarks :
                if draw :
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)

        return img

    def findposition(self, img, handno = 0, draw = True) :
        LM_list = []
        if self.result.multi_hand_landmarks :
            myhand = self.result.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                LM_list.append([id, cx, cy])
                
                if draw :
                    cv2.circle(img, (cx,cy), 10, (0,0,255), cv2.FILLED)
        return LM_list


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out_mp4 = cv2.VideoWriter('Gestures.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

    detector = handDetector()
    
    while True :
        _, img = cap.read()
        img = detector.findhands(img)
        LM_list = detector.findposition(img)

        if len(LM_list) != 0 :

            x_diff = abs(LM_list[9][1] - LM_list[0][1])
            y_diff = abs(LM_list[9][2] - LM_list[0][2])
            # print(x_diff,y_diff)

            if x_diff < 0.05:      
                m = 1000000000
            else:
                m = y_diff/x_diff

            if m>1:
                if LM_list[9][2] < LM_list[0][2]:
                    msg = "UP"

                if LM_list[9][2] > LM_list[0][2]:
                    msg = "DOWN"
                    
            if m>=0 and m<=1:
                if LM_list[9][1] < LM_list[0][1]:
                    msg = "RIGHT"

                if LM_list[9][1] > LM_list[0][1]:
                    msg = "LEFT"

            if LM_list[7][2] < LM_list[0][2] and LM_list[7][2] > LM_list[5][2] and LM_list[11][2] > LM_list[10][2]:
                msg = "FIST"
                    
            cv2.putText(img, msg, (150,70), cv2.FONT_HERSHEY_DUPLEX, 3, (0,255,0), 3)
                
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0), 3)

        cv2.imshow("Webcam", img)
        out_mp4.write(img)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            cap.release()
            out_mp4.release()
            break

if __name__ == "__main__" :
    main() 