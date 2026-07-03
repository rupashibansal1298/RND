"""
2D Sprite Motion Capture
A 2D rig usually consists of:
      Head
       |
   Upper Body
    /      \
 Arm        Arm
    \      /
   Lower Body
    /      \
  Leg      Leg
Instead of moving pixels, we rotate image parts.
"""

"""
MediaPipe
     ↓
Convert coordinates
     ↓
Draw stick figure using pygame

"""

# in this code, we are using the coordinates to draw the lines, but we want to use the angles to move an actual body part image

import cv2
import mediapipe as mp
import pygame
import math

pygame.init()



WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def angle(a,b):

    dx = b[0]-a[0]
    dy = b[1]-a[1]

    return math.degrees(math.atan2(dy,dx))

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False

    #ret, frame = cap.read()

    #frame = cv2.flip(frame,1)

    #rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #  this part is unsafe
    ret, frame = cap.read()

    if not ret:
        print("Could not read from webcam")
        continue

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    
    result = pose.process(rgb)

    screen.fill((30,30,30))

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        ls = (int(lm[11].x*WIDTH),int(lm[11].y*HEIGHT))        # we have these extracted codes alreadt
        rs = (int(lm[12].x*WIDTH),int(lm[12].y*HEIGHT))

        le = (int(lm[13].x*WIDTH),int(lm[13].y*HEIGHT))
        re = (int(lm[14].x*WIDTH),int(lm[14].y*HEIGHT))

        lw = (int(lm[15].x*WIDTH),int(lm[15].y*HEIGHT))
        rw = (int(lm[16].x*WIDTH),int(lm[16].y*HEIGHT))

        lh = (int(lm[23].x*WIDTH),int(lm[23].y*HEIGHT))
        rh = (int(lm[24].x*WIDTH),int(lm[24].y*HEIGHT))

        lk = (int(lm[25].x*WIDTH),int(lm[25].y*HEIGHT))
        rk = (int(lm[26].x*WIDTH),int(lm[26].y*HEIGHT))

        la = (int(lm[27].x*WIDTH),int(lm[27].y*HEIGHT))
        ra = (int(lm[28].x*WIDTH),int(lm[28].y*HEIGHT))

        pygame.draw.line(screen,(255,255,255),ls,rs,5)

        pygame.draw.line(screen,(0,255,0),ls,le,5)
        pygame.draw.line(screen,(0,255,0),le,lw,5)

        pygame.draw.line(screen,(0,255,0),rs,re,5)
        pygame.draw.line(screen,(0,255,0),re,rw,5)

        pygame.draw.line(screen,(255,0,0),lh,rh,5)

        pygame.draw.line(screen,(0,0,255),lh,lk,5)
        pygame.draw.line(screen,(0,0,255),lk,la,5)

        pygame.draw.line(screen,(0,0,255),rh,rk,5)
        pygame.draw.line(screen,(0,0,255),rk,ra,5)

        for p in [ls,rs,le,re,lw,rw,lh,rh,lk,rk,la,ra]:
            pygame.draw.circle(screen,(255,255,0),p,8)

    pygame.display.update()
    clock.tick(60)

cap.release()
pygame.quit()
