"""
what this code is doing 

Shoulder
    |
 Elbow
    |
 Wrist

        ↓

Calculate Elbow Angle
"""



import cv2
import mediapipe as mp
import numpy as np

# we are calculating the left elbow angle 



def calculate_angle(a,b,c):           #Angle Calculation Function , calcuklte the angle formed at pt b
 # shouler a, elbow b, wrist c
    a=np.array(a)                    
    b=np.array(b)        # Convert points to NumPy arrays
    c=np.array(c)

    radians=np.arctan2(           #Calculate direction of each arm segment, angle 1- angle 2 
        c[1]-b[1],              
        c[0]-b[0]
    ) - np.arctan2(
        a[1]-b[1],
        a[0]-b[0]
    )

    angle=np.abs(radians*180.0/np.pi)      #Convert radians to degrees, easier to understand 

    if angle > 180:                        #Keep angle between 0° and 180°
        angle = 360-angle

    return angle

mp_pose = mp.solutions.pose          #Initialize MediaPipe Pose

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgb)

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        shoulder = [lm[11].x, lm[11].y]
        elbow = [lm[13].x, lm[13].y]
        wrist = [lm[15].x, lm[15].y]

        angle = calculate_angle(              # These coordinates are normalized:
            shoulder,
            elbow,
            wrist
        )
        # fully bent ~ 40 deg
        # half bent ~ 90 deg
        # straight arm ~ 180 deg



        print("Elbow Angle:", int(angle))

    cv2.imshow("Tracking", frame)          #Show Webcam Feed
#OpenCV creates a window named "Tracking" and shows the current image stored in fram


    mp.solutions.drawing_utils.draw_landmarks(         # to draw the landmark s
    frame,
    result.pose_landmarks,
    mp_pose.POSE_CONNECTIONS
)



    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()





# now the  next stage is rig mapping, where MediaPipe joints control a character.