import cv2
import mediapipe as mp
import pygame
import math

# -----------------------
# PYGAME
# -----------------------

"""
In the rectangular rig, we are NOT rotating an existing rectangle.
We are creating a new rotated rectangle every frame.

"""

pygame.init()

WIDTH = 900
HEIGHT = 700

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Rectangular Motion Capture")

clock = pygame.time.Clock()

# -----------------------
# MEDIAPIPE
# -----------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -----------------------
# SMOOTHING
# -----------------------

points = {}

def smooth(name, p):

    alpha = 0.7

    if name not in points:
        points[name] = p
        return p

    old = points[name]

    x = alpha * old[0] + (1 - alpha) * p[0]
    y = alpha * old[1] + (1 - alpha) * p[1]

    points[name] = (x, y)

    return (x, y)

# -----------------------
# DRAW ROTATED RECTANGLE
# -----------------------

def draw_bone(start, end, thickness, color):

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = math.sqrt(dx * dx + dy * dy)

    angle = math.atan2(dy, dx)

    px = thickness / 2 * math.sin(angle)
    py = thickness / 2 * math.cos(angle)

    p1 = (start[0] - px, start[1] + py)
    p2 = (start[0] + px, start[1] - py)
    p3 = (end[0] + px, end[1] - py)
    p4 = (end[0] - px, end[1] + py)

    pygame.draw.polygon(
        screen,
        color,
        [p1, p2, p3, p4]
    )

# -----------------------
# MAIN LOOP
# -----------------------

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(rgb)

    screen.fill((30, 30, 30))

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        # HEAD

        nose = smooth(
            "nose",
            (
                lm[0].x * WIDTH,
                lm[0].y * HEIGHT
            )
        )

        # LEFT ARM

        ls = smooth(
            "ls",
            (
                lm[11].x * WIDTH,
                lm[11].y * HEIGHT
            )
        )

        le = smooth(
            "le",
            (
                lm[13].x * WIDTH,
                lm[13].y * HEIGHT
            )
        )

        lw = smooth(
            "lw",
            (
                lm[15].x * WIDTH,
                lm[15].y * HEIGHT
            )
        )

        # RIGHT ARM

        rs = smooth(
            "rs",
            (
                lm[12].x * WIDTH,
                lm[12].y * HEIGHT
            )
        )

        re = smooth(
            "re",
            (
                lm[14].x * WIDTH,
                lm[14].y * HEIGHT
            )
        )

        rw = smooth(
            "rw",
            (
                lm[16].x * WIDTH,
                lm[16].y * HEIGHT
            )
        )

        # LEFT LEG

        lh = smooth(
            "lh",
            (
                lm[23].x * WIDTH,
                lm[23].y * HEIGHT
            )
        )

        lk = smooth(
            "lk",
            (
                lm[25].x * WIDTH,
                lm[25].y * HEIGHT
            )
        )

        la = smooth(
            "la",
            (
                lm[27].x * WIDTH,
                lm[27].y * HEIGHT
            )
        )

        # RIGHT LEG

        rh = smooth(
            "rh",
            (
                lm[24].x * WIDTH,
                lm[24].y * HEIGHT
            )
        )

        rk = smooth(
            "rk",
            (
                lm[26].x * WIDTH,
                lm[26].y * HEIGHT
            )
        )

        ra = smooth(
            "ra",
            (
                lm[28].x * WIDTH,
                lm[28].y * HEIGHT
            )
        )

        # BODY

        chest = (
            (ls[0] + rs[0]) / 2,
            (ls[1] + rs[1]) / 2
        )

        hip = (
            (lh[0] + rh[0]) / 2,
            (lh[1] + rh[1]) / 2
        )

        # DRAW ORDER

        # Legs

        draw_bone(
            lh,
            lk,
            18,
            (0, 100, 255)
        )

        draw_bone(
            lk,
            la,
            16,
            (0, 150, 255)
        )

        draw_bone(
            rh,
            rk,
            18,
            (0, 100, 255)
        )

        draw_bone(
            rk,
            ra,
            16,
            (0, 150, 255)
        )

        # Body

        draw_bone(
            chest,
            hip,
            50,
            (200, 200, 200)
        )

        # Arms

        draw_bone(
            ls,
            le,
            16,
            (0, 255, 100)
        )

        draw_bone(
            le,
            lw,
            14,
            (100, 255, 100)
        )

        draw_bone(
            rs,
            re,
            16,
            (0, 255, 100)
        )

        draw_bone(
            re,
            rw,
            14,
            (100, 255, 100)
        )

        # Head

        pygame.draw.circle(
            screen,
            (255, 220, 150),
            (
                int(nose[0]),
                int(nose[1] - 40)
            ),
            35
        )

        # Joints

        joints = [
            ls, rs,
            le, re,
            lw, rw,
            lh, rh,
            lk, rk,
            la, ra
        ]

        for p in joints:

            pygame.draw.circle(
                screen,
                (255, 255, 0),
                (
                    int(p[0]),
                    int(p[1])
                ),
                6
            )

    pygame.display.update()
    clock.tick(60)

cap.release()
pygame.quit()