import cv2
import mediapipe as mp
import pygame
import math
import os
import sys

# =========================
# INIT
# =========================

pygame.init()

WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# =========================
# LOAD SPRITES
# =========================

# Resolve paths relative to THIS FILE, not the current working directory --
# this way it works no matter where you launch the script from (terminal,
# VS Code "Run" button, double-click, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

def load(filename):
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        sys.exit(
            f"\nMissing asset: {path}\n"
            f"Make sure the 'assets' folder sits in the same directory as this script:\n"
            f"{BASE_DIR}\n"
        )
    return pygame.image.load(path).convert_alpha()

head = load("head.png")
torso = load("torso.png")

upper_arm = load("upper_arm.png")
lower_arm = load("lower_arm.png")

upper_leg = load("upper_leg.png")
lower_leg = load("lower_leg.png")

# joint-cap colors, matched to the sprite art so the cap blends in
# (skin tone for elbows, pants navy for knees -- see generate_character.py)
ELBOW_COLOR = (235, 181, 145)
KNEE_COLOR  = (52, 63, 96)

# =========================
# MATH
# =========================

def smooth_point(prev, new, alpha=0.75):
    if prev is None:
        return new
    return (
        prev[0] * alpha + new[0] * (1 - alpha),
        prev[1] * alpha + new[1] * (1 - alpha),
    )

# =========================
# CORE RENDER FUNCTIONS
# =========================

def draw_bone(img, start, end, thickness, pivot=(0.07, 0.5)):
    """
    Draw a limb sprite stretched + rotated between two joints.

    pivot=(0.07, 0.5) means the attachment point sits 7% in from the sprite's
    left edge, not right at the edge. That leaves a small "overlap nub" -- the
    first 7% of the sprite -- extending backward, underneath whatever piece
    is drawn before this one. That nub + the joint cap below is what removes
    the visible seam at the joint.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = math.hypot(dx, dy)
    if length < 2:
        return

    angle = math.degrees(math.atan2(dy, dx))

    img_scaled = pygame.transform.scale(img, (int(length), int(thickness)))
    rotated = pygame.transform.rotate(img_scaled, -angle)

    pivot_x = pivot[0] * img_scaled.get_width()
    pivot_y = pivot[1] * img_scaled.get_height()

    offset = pygame.math.Vector2(
        img_scaled.get_width() / 2 - pivot_x,
        img_scaled.get_height() / 2 - pivot_y
    )
    offset = offset.rotate(angle)

    rect = rotated.get_rect(center=(start[0] + offset.x, start[1] + offset.y))
    screen.blit(rotated, rect)


def draw_joint(pos, radius, color):
    """Small filled circle dropped exactly on a joint to hide the seam
    between two limb sprites (e.g. elbow between upper_arm and lower_arm)."""
    pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), int(radius))

# =========================
# MEDIAPIPE
# =========================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# =========================
# SMOOTH MEMORY
# =========================

mem = {}

def P(i, x, y):
    key = i
    new = (x, y)
    if key not in mem:
        mem[key] = new
    else:
        mem[key] = smooth_point(mem[key], new)
    return mem[key]

# =========================
# LOOP
# =========================

running = True

while running:

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    screen.fill((20, 20, 20))

    if res.pose_landmarks:

        lm = res.pose_landmarks.landmark

        # =========================
        # JOINTS
        # =========================

        ls = P(11, lm[11].x * WIDTH, lm[11].y * HEIGHT)
        rs = P(12, lm[12].x * WIDTH, lm[12].y * HEIGHT)

        le = P(13, lm[13].x * WIDTH, lm[13].y * HEIGHT)
        re = P(14, lm[14].x * WIDTH, lm[14].y * HEIGHT)

        lw = P(15, lm[15].x * WIDTH, lm[15].y * HEIGHT)
        rw = P(16, lm[16].x * WIDTH, lm[16].y * HEIGHT)

        lh = P(23, lm[23].x * WIDTH, lm[23].y * HEIGHT)
        rh = P(24, lm[24].x * WIDTH, lm[24].y * HEIGHT)

        lk = P(25, lm[25].x * WIDTH, lm[25].y * HEIGHT)
        rk = P(26, lm[26].x * WIDTH, lm[26].y * HEIGHT)

        la = P(27, lm[27].x * WIDTH, lm[27].y * HEIGHT)
        ra = P(28, lm[28].x * WIDTH, lm[28].y * HEIGHT)

        # =========================
        # BODY CENTERS
        # =========================

        chest = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

        shoulder_width = math.dist(ls, rs)
        torso_height = math.dist(chest, hip)

        scale = shoulder_width / 150

        arm_thickness = max(18, int(26 * scale))
        leg_thickness = max(20, int(32 * scale))

        # =========================
        # LIMBS
        # (drawn BEFORE torso/head on purpose -- the torso is drawn on top
        # afterward and naturally covers the shoulder/hip seams, so we only
        # need explicit joint caps at the elbows and knees below)
        # =========================

        # Left arm
        draw_bone(upper_arm, ls, le, arm_thickness)
        draw_bone(lower_arm, le, lw, arm_thickness)

        # Right arm
        draw_bone(upper_arm, rs, re, arm_thickness)
        draw_bone(lower_arm, re, rw, arm_thickness)

        # Left leg
        draw_bone(upper_leg, lh, lk, leg_thickness)
        draw_bone(lower_leg, lk, la, leg_thickness)

        # Right leg
        draw_bone(upper_leg, rh, rk, leg_thickness)
        draw_bone(lower_leg, rk, ra, leg_thickness)

        # =========================
        # JOINT CAPS (hide elbow/knee seams)
        # =========================

        draw_joint(le, arm_thickness * 0.42, ELBOW_COLOR)
        draw_joint(re, arm_thickness * 0.42, ELBOW_COLOR)
        draw_joint(lk, leg_thickness * 0.42, KNEE_COLOR)
        draw_joint(rk, leg_thickness * 0.42, KNEE_COLOR)

        # =========================
        # TORSO + HEAD (STABLE)
        # =========================

        torso_scaled = pygame.transform.scale(
            torso,
            (int(shoulder_width * 1.25), int(torso_height * 1.25))
        )

        torso_center = ((chest[0] + hip[0]) / 2, (chest[1] + hip[1]) / 2)

        dx = hip[0] - chest[0]
        dy = hip[1] - chest[1]
        body_angle = math.degrees(math.atan2(dy, dx)) - 90

        rotated_torso = pygame.transform.rotate(torso_scaled, -body_angle)
        torso_rect = rotated_torso.get_rect(center=torso_center)
        screen.blit(rotated_torso, torso_rect)

        head_size = int(shoulder_width * 0.9)
        head_scaled = pygame.transform.scale(head, (head_size, head_size))

        neck = chest
        head_center = (neck[0], neck[1] - head_size * 0.7)
        head_rect = head_scaled.get_rect(center=head_center)
        screen.blit(head_scaled, head_rect)

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
