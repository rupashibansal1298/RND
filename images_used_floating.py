
# this is giving a really haggered op , cuz the images are not rigged and are just floating aroung, not connected
# we will have to use an actial char that has been rigged

# the images are being roatated around its joints
"""
Webcam
   ↓
MediaPipe
   ↓
Extract Joints
   ↓
Map joints to a 2D sprite rig
   ↓
Animate a character in real time
"""


import cv2
import mediapipe as mp
import pygame
import math

pygame.init()

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Sprite Motion Capture")

clock = pygame.time.Clock()

# -----------------------------
# LOAD SPRITES
# -----------------------------

head = pygame.image.load("sprites/head.png").convert_alpha()
torso = pygame.image.load("sprites/torso.png").convert_alpha()

upper_arm = pygame.image.load("sprites/upper_arm.png").convert_alpha()
lower_arm = pygame.image.load("sprites/lower_arm.png").convert_alpha()

upper_leg = pygame.image.load("sprites/upper_leg.png").convert_alpha()
lower_leg = pygame.image.load("sprites/lower_leg.png").convert_alpha()

# -----------------------------

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_angle(a,b):

    dx = b[0]-a[0]
    dy = b[1]-a[1]

    return -math.degrees(
        math.atan2(dy,dx)
    )

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(rgb)

    screen.fill((30,30,30))

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        ls = (int(lm[11].x*WIDTH),int(lm[11].y*HEIGHT))
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

        # -----------------------------
        # CALCULATE ROTATIONS
        # -----------------------------

        lua = get_angle(ls,le)
        lla = get_angle(le,lw)

        rua = get_angle(rs,re)
        rla = get_angle(re,rw)

        lul = get_angle(lh,lk)
        lll = get_angle(lk,la)

        rul = get_angle(rh,rk)
        rll = get_angle(rk,ra)

        # -----------------------------
        # ROTATE BODY PARTS
        # -----------------------------

        left_upper_arm = pygame.transform.rotate(
            upper_arm,
            lua
        )

        left_lower_arm = pygame.transform.rotate(
            lower_arm,
            lla
        )

        right_upper_arm = pygame.transform.rotate(
            upper_arm,
            rua
        )

        right_lower_arm = pygame.transform.rotate(
            lower_arm,
            rla
        )

        left_upper_leg = pygame.transform.rotate(
            upper_leg,
            lul
        )

        left_lower_leg = pygame.transform.rotate(
            lower_leg,
            lll
        )

        right_upper_leg = pygame.transform.rotate(
            upper_leg,
            rul
        )

        right_lower_leg = pygame.transform.rotate(
            lower_leg,
            rll
        )

        # -----------------------------
        # DRAW CHARACTER
        # -----------------------------

        head_x = (ls[0]+rs[0])//2 - head.get_width()//2
        head_y = min(ls[1],rs[1]) - 100

        screen.blit(
            head,
            (head_x,head_y)
        )

        torso_x = (ls[0]+rs[0])//2 - torso.get_width()//2
        torso_y = (ls[1]+lh[1])//2 - torso.get_height()//2

        screen.blit(
            torso,
            (torso_x,torso_y)
        )

        screen.blit(left_upper_arm,ls)
        screen.blit(left_lower_arm,le)

        screen.blit(right_upper_arm,rs)
        screen.blit(right_lower_arm,re)

        screen.blit(left_upper_leg,lh)
        screen.blit(left_lower_leg,lk)

        screen.blit(right_upper_leg,rh)
        screen.blit(right_lower_leg,rk)

    pygame.display.update()
    clock.tick(60)

cap.release()
pygame.quit()
