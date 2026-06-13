"""
issues with the code
Right now your code is still:
❌ drawing each limb independently
❌ using world-space endpoints directly
❌ no parent-child transforms
❌ no bone rotation propagation
"""
"""
Before (your code)
limbs = independent sprites
position = raw MediaPipe points
no hierarchy
no propagation
"""


import cv2
import mediapipe as mp
import pygame
import math

# =====================================
# PYGAME
# =====================================

pygame.init()

WIDTH = 900
HEIGHT = 700

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Sprite Motion Capture")

clock = pygame.time.Clock()

# =====================================
# LOAD SPRITES
# =====================================

head = pygame.image.load("head.png").convert_alpha()

torso = pygame.image.load("torso.png").convert_alpha()

upper_arm = pygame.image.load("upper_arm.png").convert_alpha()
lower_arm = pygame.image.load("lower_arm.png").convert_alpha()

# using lower_leg for both parts

upper_leg = pygame.image.load("upper_leg.png").convert_alpha()
lower_leg = pygame.image.load("lower_leg.png").convert_alpha()

# =====================================
# MEDIAPIPE
# =====================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# =====================================
# SMOOTHING
# =====================================

memory = {}

def smooth(name, p):

    alpha = 0.7

    if name not in memory:
        memory[name] = p
        return p

    old = memory[name]

    x = alpha * old[0] + (1-alpha) * p[0]
    y = alpha * old[1] + (1-alpha) * p[1]

    memory[name] = (x, y)

    return (x, y)

# =====================================
# ANGLE
# =====================================

def angle(a,b):

    dx = b[0]-a[0]
    dy = b[1]-a[1]

    return math.degrees(
        math.atan2(dy,dx)
    )

# =====================================
# DRAW HORIZONTAL LIMB
# =====================================

# def draw_horizontal(image,start,end):

#     dx = end[0]-start[0]
#     dy = end[1]-start[1]

#     length = math.sqrt(
#         dx*dx+dy*dy
#     )

#     scaled = pygame.transform.scale(
#         image,
#         (
#             max(10,int(length)),
#             image.get_height()
#         )
#     )

#     ang = angle(start,end)

#     rotated = pygame.transform.rotate(
#         scaled,
#         -ang
#     )

#     rect = rotated.get_rect(
#         center=start
#     )

#     screen.blit(
#         rotated,
#         rect
#     )

def draw_horizontal(image, start, end):

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = math.sqrt(dx*dx + dy*dy)

    scaled = pygame.transform.scale(
        image,
        (
            max(1, int(length)),
            image.get_height()
        )
    )

    ang = math.degrees(
        math.atan2(dy, dx)
    )

    rotated = pygame.transform.rotate(
        scaled,
        -ang
    )

    pivot = pygame.math.Vector2(
        0,
        scaled.get_height()/2
    )

    offset = pivot.rotate(ang)

    rect = rotated.get_rect(
        center=(
            start[0] + offset.x,
            start[1] + offset.y
        )
    )

    screen.blit(
        rotated,
        rect
    )

# =====================================
# DRAW VERTICAL LIMB
# =====================================

# def draw_vertical(image,start,end):

#     dx = end[0]-start[0]
#     dy = end[1]-start[1]

#     length = math.sqrt(
#         dx*dx+dy*dy
#     )

#     scaled = pygame.transform.scale(
#         image,
#         (
#             image.get_width(),
#             max(10,int(length))
#         )
#     )

#     ang = angle(start,end)

#     rotated = pygame.transform.rotate(
#         scaled,
#         -ang+90
#     )

#     rect = rotated.get_rect(
#         center=start
#     )

#     screen.blit(
#         rotated,
#         rect
#     )


def draw_vertical(image, start, end):

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = math.sqrt(dx*dx + dy*dy)

    scaled = pygame.transform.scale(
        image,
        (
            image.get_width(),
            max(1, int(length))
        )
    )

    ang = math.degrees(
        math.atan2(dy, dx)
    )

    rotated = pygame.transform.rotate(
        scaled,
        -ang + 90
    )

    pivot = pygame.math.Vector2(
        scaled.get_width()/2,
        0
    )

    offset = pivot.rotate(
        ang - 90
    )

    rect = rotated.get_rect(
        center=(
            start[0] + offset.x,
            start[1] + offset.y
        )
    )

    screen.blit(
        rotated,
        rect
    )


# =====================================
# LOOP
# =====================================

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

    screen.fill((25,25,25))

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        # HEAD

        nose = smooth(
            "nose",
            (
                lm[0].x*WIDTH,
                lm[0].y*HEIGHT
            )
        )

        # LEFT ARM

        ls = smooth(
            "ls",
            (
                lm[11].x*WIDTH,
                lm[11].y*HEIGHT
            )
        )

        le = smooth(
            "le",
            (
                lm[13].x*WIDTH,
                lm[13].y*HEIGHT
            )
        )

        lw = smooth(
            "lw",
            (
                lm[15].x*WIDTH,
                lm[15].y*HEIGHT
            )
        )

        # RIGHT ARM

        rs = smooth(
            "rs",
            (
                lm[12].x*WIDTH,
                lm[12].y*HEIGHT
            )
        )

        re = smooth(
            "re",
            (
                lm[14].x*WIDTH,
                lm[14].y*HEIGHT
            )
        )

        rw = smooth(
            "rw",
            (
                lm[16].x*WIDTH,
                lm[16].y*HEIGHT
            )
        )

        # LEFT LEG

        lh = smooth(
            "lh",
            (
                lm[23].x*WIDTH,
                lm[23].y*HEIGHT
            )
        )

        lk = smooth(
            "lk",
            (
                lm[25].x*WIDTH,
                lm[25].y*HEIGHT
            )
        )

        la = smooth(
            "la",
            (
                lm[27].x*WIDTH,
                lm[27].y*HEIGHT
            )
        )

        # RIGHT LEG

        rh = smooth(
            "rh",
            (
                lm[24].x*WIDTH,
                lm[24].y*HEIGHT
            )
        )

        rk = smooth(
            "rk",
            (
                lm[26].x*WIDTH,
                lm[26].y*HEIGHT
            )
        )

        ra = smooth(
            "ra",
            (
                lm[28].x*WIDTH,
                lm[28].y*HEIGHT
            )
        )

        # BODY CENTERS

        chest = (
            (ls[0]+rs[0])/2,
            (ls[1]+rs[1])/2
        )

        hip = (
            (lh[0]+rh[0])/2,
            (lh[1]+rh[1])/2
        )

        # =====================
        # LEGS
        # =====================

        draw_vertical(
            upper_leg,
            lh,
            lk
        )

        draw_vertical(
            lower_leg,
            lk,
            la
        )

        draw_vertical(
            upper_leg,
            rh,
            rk
        )

        draw_vertical(
            lower_leg,
            rk,
            ra
        )

        # =====================
        # BODY
        # =====================

        body_rect = torso.get_rect(
            center=(
                int(chest[0]),
                int((chest[1]+hip[1])/2)
            )
        )

        screen.blit(
            torso,
            body_rect
        )

        # =====================
        # ARMS
        # =====================

        draw_horizontal(
            upper_arm,
            ls,
            le
        )

        draw_horizontal(
            lower_arm,
            le,
            lw
        )

        draw_horizontal(
            upper_arm,
            rs,
            re
        )

        draw_horizontal(
            lower_arm,
            re,
            rw
        )

        # =====================
        # HEAD
        # =====================

        head_rect = head.get_rect(
            center=(
                int(chest[0]),
                int(chest[1]-120)
            )
        )

        screen.blit(
            head,
            head_rect
        )

        # =====================
        # DEBUG JOINTS
        # =====================

        joints = [
            ls,rs,
            le,re,
            lw,rw,
            lh,rh,
            lk,rk,
            la,ra
        ]

        for p in joints:

            pygame.draw.circle(
                screen,
                (255,255,0),
                (
                    int(p[0]),
                    int(p[1])
                ),
                4
            )

    pygame.display.update()

    clock.tick(60)

cap.release()
pygame.quit()