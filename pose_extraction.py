"""
what this is doing :
Webcam
   ↓
MediaPipe
   ↓
Extract Left Wrist
   ↓
Print Coordinates
"""



import cv2
import mediapipe as mp

#media pipe detects 33 body landmarks
# it detects once and then tracks landmarks accross the frame --> faster than redirecting every frame
# each landmark --> (x y z) where z--> depth relative to the camera
# z--> negative are closer to camera and positive--> furthur away
# these coords are normalised

# media pipe uses both geometry and a deep neural network trained on millions of human pose examples


# in most motion capture systems, a separate non-normalized coordinate extraction step is not necessary.

# OPEN CV is used for : webcam access, image processing, display windows
# MEDIA PIPE : pose detection, landmark extraction

#Each landmark contains: x y z visibility
#These coordinates are normalized:


mp_pose = mp.solutions.pose                #shorts cuts created
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(                  #initialize pose detector 
    static_image_mode=False,          # ie video mode 
    model_complexity=1,               # balanced, between fast and most accurate
    min_detection_confidence=0.5,     # Minimum confidence required to accept a pose.
    min_tracking_confidence=0.5       # Tracking confidence threshold after detection.
)

cap = cv2.VideoCapture(0)            # here 0 is the device ID, ie the id of the webcam

while True:                 # continuously process through webframes

    success, frame = cap.read()         # read a frame

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            # conversion from BGR to RGB
    # IMPORTANT:
    #  Open CV stores images as BGR 
    #  media pipe expects RGB


    

    results = pose.process(rgb)           # running pose detection, joints are predicted from the image 
    # contains all detected landmark

    if results.pose_landmarks:            # if detected---> 33 landmarks

        landmarks = results.pose_landmarks.landmark       # all landmarks extracted, therefore we have alll the coords

        left_wrist = landmarks[15]

        print(
            "Left Wrist:",                      # coords of left wrist are printed, to verify coord extraction
            round(left_wrist.x,3),
            round(left_wrist.y,3),
            round(left_wrist.z,3)
        )

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:       # 27 is the ascii code for ESC.  --> stop key
        break

cap.release()            # release web cam
cv2.destroyAllWindows()       #Closes all OpenCV windows.



"""
Webcam
   ↓
Read Frame
   ↓
BGR → RGB
   ↓
MediaPipe Pose
   ↓
33 Landmarks
   ↓
Extract Left Wrist
   ↓
Print Coordinates
   ↓
Draw Skeleton
   ↓
Display Frame
   ↓
Repeat

"""











"""
for this projject, 
we are using mediapip as it gives the best balance between accuracy, speed, and ease of implementation.
we need a pose estimator that is:
Fast (30–60 FPS)
Easy to install
Works on CPU
Produces stable landmarks
Has Python support
Doesn't require training a model
MediaPipe satisfies all of these.


When should you NOT use MediaPipe?
If you want:
Multiple people
Person A
Person B
Person C
then RTMPose or OpenPose may be better.



THEORY that mediapipe uses to extract coords:
MediaPipe outputs normalized coordinates.

first the webcam captures images (RGB) --> at this pt, computer only sees pixels

Entire Image
     ↓
Person Detector
     ↓
Bounding Box 
(ie , the search space is reduced)

now the feature extraction:
The cropped body image is passed through a Convolutional Neural Network (CNN).
the network converts images to features 

--> landmark regression : f(image) = coordinates

The z-coordinate is not true physical depth.
MediaPipe estimates relative depth from visual cues learned during training.
Closer to camera → more negative
Farther away → more positive

temporal tracking :  makes it fast 
Detect person
Detect landmarks
----------------------
Use previous landmarks
Predict new landmarks
---------------------
Track movement
---------------------



then filters are applied to reduce noise (smoothing )

Visibility is the model's confidence that the landmark is visible.

"""



"""
to understand open cv:
open cv sees images as matrices : [H W channels]




"""