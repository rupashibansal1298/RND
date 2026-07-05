import cv2
import mediapipe as mp
import pygame #rendering the animated character.
import math
import os
import sys #terminate program if assets are missing.

# =========================
# INIT
# =========================

pygame.init()

#This creates the animation window.
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock() #frame rate

# =========================
# LOAD SPRITES
# =========================
#This finds the directory containing the Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

def load(filename):
    path = os.path.join(ASSETS_DIR, filename) #exit if an asset is missing
    if not os.path.exists(path):
        sys.exit(
            f"\nMissing asset: {path}\n"
            f"Make sure the 'assets' folder sits in the same directory as this script:\n"
            f"{BASE_DIR}\n"
        )
    return pygame.image.load(path).convert_alpha()

#load the images
head = load("head.png") 
torso = load("torso.png")

upper_arm = load("upper_arm.png")
lower_arm = load("lower_arm.png")

upper_leg = load("upper_leg.png")
lower_leg = load("lower_leg.png")


# Used to draw circular joints connecting limbs.

ELBOW_COLOR = (235, 181, 145)
KNEE_COLOR  = (52, 63, 96) 

# =========================
# MATH
# =========================


# New Position = 75% previous + 25% current

def smooth_point(prev, new, alpha=0.75):
    if prev is None:
        return new
    return (
        prev[0] * alpha + new[0] * (1 - alpha),
        prev[1] * alpha + new[1] * (1 - alpha),
    )

def smooth_val(prev, new, alpha=0.75):  # for z value
    if prev is None:
        return new
    return prev * alpha + new * (1 - alpha)

# =========================
# CORE RENDER FUNCTIONS
# =========================

def draw_bone(img, start, end, thickness, pivot=(0.07, 0.5)):
    #It stretches the sprite between the two joints.

    dx = end[0] - start[0] # direction
    dy = end[1] - start[1]

    length = math.hypot(dx, dy)
    if length < 2: 
        return

    angle = math.degrees(math.atan2(dy, dx)) #calculate angle 

    img_scaled = pygame.transform.scale(img, (int(length), int(thickness))) # rescale
    rotated = pygame.transform.rotate(img_scaled, -angle) #roatate (arm points towards the elbow)


    pivot_x = pivot[0] * img_scaled.get_width()  # pivot correction
    pivot_y = pivot[1] * img_scaled.get_height()

    offset = pygame.math.Vector2(
        img_scaled.get_width() / 2 - pivot_x,
        img_scaled.get_height() / 2 - pivot_y
    )
    offset = offset.rotate(angle)

    rect = rotated.get_rect(center=(start[0] + offset.x, start[1] + offset.y))
    screen.blit(rotated, rect)


#Draws circles at elbows and knees.
def draw_joint(pos, radius, color):
    pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), int(radius))


def draw_arm(shoulder, elbow, wrist, thickness): #draw arm
    draw_bone(upper_arm, shoulder, elbow, thickness)
    draw_bone(lower_arm, elbow, wrist, thickness)
    draw_joint(elbow, thickness * 0.42, ELBOW_COLOR)


def draw_leg(hip, knee, ankle, thickness): #draw leg
    draw_bone(upper_leg, hip, knee, thickness)
    draw_bone(lower_leg, knee, ankle, thickness)
    draw_joint(knee, thickness * 0.42, KNEE_COLOR)

# =========================
# MEDIAPIPE
# =========================

#Creates the pose detection model.

mp_pose = mp.solutions.pose 
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# =========================
# SMOOTH MEMORY
# =========================

#store past poses
mem = {}       # smoothed (x, y) screen positions
mem_z = {}     # smoothed raw z depth values (per landmark)

def P(i, x, y):
    key = i
    new = (x, y)
    if key not in mem:
        mem[key] = new
    else:
        mem[key] = smooth_point(mem[key], new)
    return mem[key]

def Z(i, z):
    if i not in mem_z:
        mem_z[i] = z
    else:
        mem_z[i] = smooth_val(mem_z[i], z)
    return mem_z[i]

# now we get the smooth points

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

    frame = cv2.flip(frame, 1) #mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #media pipe requires RGB
    res = pose.process(rgb)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (WIDTH, HEIGHT))



    surface = pygame.surfarray.make_surface(frame.swapaxes(0,1))

    screen.blit(surface, (0,0))

    if res.pose_landmarks:

        lm = res.pose_landmarks.landmark

        # =========================
        # JOINTS (screen space)
        # =========================

        # convert normalized coords intpo screen pixels


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
        # DEPTH (z space) -- MediaPipe's z is roughly in the same
        # scale as x, with the origin around the hips. Smaller/negative
        # values are CLOSER to the camera.
        # =========================

        # smaller z--> closer to camera

        z_ls, z_rs = Z(11, lm[11].z), Z(12, lm[12].z)
        z_le, z_re = Z(13, lm[13].z), Z(14, lm[14].z)
        z_lw, z_rw = Z(15, lm[15].z), Z(16, lm[16].z)
        z_lh, z_rh = Z(23, lm[23].z), Z(24, lm[24].z)

        # =========================
        # BODY CENTERS
        # =========================

        chest = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

        # torso reference depth -- roughly the plane the torso sprite sits in
        torso_z = (z_ls + z_rs + z_lh + z_rh) / 4

        # scale the torso accordingle
        shoulder_width = math.dist(ls, rs)          # dist bw shoulders
        torso_height = math.dist(chest, hip)        # dist bw chest and hip

        scale = shoulder_width / 150

        #limb thickness

        arm_thickness = max(18, int(26 * scale))
        leg_thickness = max(20, int(32 * scale))

        # =========================
        # LEGS (always drawn behind the torso -- in a normal standing
        # pose the legs don't come in front of the chest/stomach)
        # =========================

        draw_leg(lh, lk, la, leg_thickness)
        draw_leg(rh, rk, ra, leg_thickness)

        # =========================
        # DECIDE ARM LAYERING PER ARM
        # Average the arm's own landmark depths and compare to the
        # torso's depth. If the arm is closer to the camera than the
        # torso, it should be drawn ON TOP of the torso (in front);
        # otherwise it stays behind it, like in real life.
        # =========================

        left_arm_z = (z_ls + z_le + z_lw) / 3
        right_arm_z = (z_rs + z_re + z_rw) / 3

        # small bias so an arm hanging naturally at the side (roughly
        # in the same plane as the torso) still renders behind it
        DEPTH_BIAS = 0.03

        left_arm_in_front = left_arm_z < (torso_z - DEPTH_BIAS)
        right_arm_in_front = right_arm_z < (torso_z - DEPTH_BIAS)

        # Draw the "behind" arms first (before torso/head)
        if not left_arm_in_front:
            draw_arm(ls, le, lw, arm_thickness)
        if not right_arm_in_front:
            draw_arm(rs, re, rw, arm_thickness)

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

        # Draw the "in front" arms last (after torso/head), so hands
        # raised toward the camera correctly overlap the body
        if left_arm_in_front:
            draw_arm(ls, le, lw, arm_thickness)
        if right_arm_in_front:
            draw_arm(rs, re, rw, arm_thickness)

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()