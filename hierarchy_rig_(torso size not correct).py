"""
A true skeletal rig must instead:
build a bone hierarchy (parent → child)
store local rotations + lengths
compute global transforms recursively
attach sprites to bones via pivots

"""

"""
int this code the upgrades are :
Now (real rig)
✔ bone tree (parent → child)
✔ rotation propagation
✔ skeletal transform chain
✔ single root drives body
✔ limbs follow hierarchy naturally



"""
"""
A correct skeletal rig has ONE source of truth:
Either:
✅ A) “Forward kinematics rig” (what you tried)
joints drive bones
hierarchy builds shape
OR:
✅ B) “Pose-driven absolute placement” (what MediaPipe gives you)
every bone is placed directly from joint-to-joint vectors
NO hierarchy needed for angles
👉 You are currently mixing BOTH
__ hence the op is not the desired op

therefore the final fixed version has:
This version:
removes broken hierarchy math
keeps clean joint-to-joint rigging
fixes alignment properly
gives stable human proportions


"""
"""
import cv2
import mediapipe as mp
import pygame
import math

# =========================================================
# INIT
# =========================================================

pygame.init()

WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# =========================================================
# LOAD SPRITES
# =========================================================

def load(name):
    return pygame.image.load(name).convert_alpha()

head_img = load("head.png")
torso_img = load("torso.png")

upper_arm_img = load("upper_arm.png")
lower_arm_img = load("lower_arm.png")

upper_leg_img = load("upper_leg2.png")
lower_leg_img = load("lower_leg2.png")

# =========================================================
# MATH HELPERS
# =========================================================

def dist(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def angle(a, b):
    return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))

def lerp(a, b, t):
    return a + (b - a) * t

# =========================================================
# SMOOTHING
# =========================================================

smooth_mem = {}

def smooth(key, p, alpha=0.7):
    if key not in smooth_mem:
        smooth_mem[key] = p
        return p

    ox, oy = smooth_mem[key]
    x = lerp(ox, p[0], 1-alpha)
    y = lerp(oy, p[1], 1-alpha)

    smooth_mem[key] = (x, y)
    return (x, y)

# =========================================================
# BONE SYSTEM
# =========================================================

class Bone:
    def __init__(self, name, length=100, image=None, parent=None, pivot=(0.5, 0.5)):
        self.name = name
        self.length = length
        self.image = image
        self.parent = parent
        self.children = []

        self.pivot = pivot

        self.local_angle = 0
        self.world_angle = 0

        self.start = (0, 0)
        self.end = (0, 0)

        if parent:
            parent.children.append(self)

    def update(self, origin, angle_deg):
        self.start = origin
        self.world_angle = angle_deg

        rad = math.radians(angle_deg)
        self.end = (
            origin[0] + math.cos(rad) * self.length,
            origin[1] + math.sin(rad) * self.length
        )

        for child in self.children:
            child.update(self.end, child.world_angle)

    def draw(self, surf):
        if self.image:
            img = pygame.transform.scale(self.image, (int(self.length), self.image.get_height()))
            rotated = pygame.transform.rotate(img, -self.world_angle)

            rect = rotated.get_rect(center=self.start)
            surf.blit(rotated, rect)

        for c in self.children:
            c.draw(surf)

# =========================================================
# MEDIAPIPE
# =========================================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# =========================================================
# BUILD SKELETON (HIERARCHY)
# =========================================================

root = Bone("root", length=0)

# torso chain
hip = Bone("hip", 120, torso_img, root)
chest = Bone("chest", 120, torso_img, hip)

# head
head = Bone("head", 60, head_img, chest)

# left arm
l_shoulder = Bone("ls", 80, upper_arm_img, chest)
l_elbow = Bone("le", 70, lower_arm_img, l_shoulder)
l_wrist = Bone("lw", 40, None, l_elbow)

# right arm
r_shoulder = Bone("rs", 80, upper_arm_img, chest)
r_elbow = Bone("re", 70, lower_arm_img, r_shoulder)
r_wrist = Bone("rw", 40, None, r_elbow)

# left leg
l_hip = Bone("lh", 90, upper_leg_img, hip)
l_knee = Bone("lk", 90, lower_leg_img, l_hip)
l_ankle = Bone("la", 40, None, l_knee)

# right leg
r_hip = Bone("rh", 90, upper_leg_img, hip)
r_knee = Bone("rk", 90, lower_leg_img, r_hip)
r_ankle = Bone("ra", 40, None, r_knee)

# =========================================================
# MAIN LOOP
# =========================================================

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
    result = pose.process(rgb)

    screen.fill((20, 20, 20))

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        # =====================================================
        # JOINTS (SMOOTHED)
        # =====================================================

        def P(i):
            return smooth(str(i), (lm[i].x * WIDTH, lm[i].y * HEIGHT))

        nose = P(0)

        ls, rs = P(11), P(12)
        le, re = P(13), P(14)
        lw, rw = P(15), P(16)

        lh, rh = P(23), P(24)
        lk, rk = P(25), P(26)
        la, ra = P(27), P(28)

        # =====================================================
        # COMPUTE GLOBAL ROOT
        # =====================================================

        hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        chest_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)

        root_pos = hip_mid

        # =====================================================
        # ANGLES (TRUE RIGGING)
        # =====================================================

        def set_angle(bone, a, b):
            bone.world_angle = angle(a, b)

        # spine
        set_angle(hip, hip_mid, chest_mid)
        set_angle(chest, chest_mid, nose)

        # head
        set_angle(head, chest_mid, nose)

        # left arm
        set_angle(l_shoulder, ls, le)
        set_angle(l_elbow, le, lw)

        # right arm
        set_angle(r_shoulder, rs, re)
        set_angle(r_elbow, re, rw)

        # left leg
        set_angle(l_hip, lh, lk)
        set_angle(l_knee, lk, la)

        # right leg
        set_angle(r_hip, rh, rk)
        set_angle(r_knee, rk, ra)

        # =====================================================
        # UPDATE HIERARCHY (CRITICAL PART)
        # =====================================================

        hip.update(root_pos, hip.world_angle)

        # =====================================================
        # DRAW
        # =====================================================

        hip.draw(screen)

        # =====================================================
        # DEBUG
        # =====================================================

        for p in [ls, rs, le, re, lw, rw, lh, rh, lk, rk, la, ra]:
            pygame.draw.circle(screen, (255, 200, 0), (int(p[0]), int(p[1])), 4)

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()




"""


import cv2
import mediapipe as mp
import pygame
import math

pygame.init()

WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# =========================
# LOAD
# =========================

def load(x):
    return pygame.image.load(x).convert_alpha()

head = load("head.png")
torso = load("torso.png")

upper_arm = load("upper_arm.png")
lower_arm = load("lower_arm.png")

upper_leg = load("upper_leg2.png")
lower_leg = load("lower_leg2.png")

# =========================
# MATH
# =========================

def angle(a, b):
    return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))

def length(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

# =========================
# SMOOTHING
# =========================

mem = {}

def smooth(k, p, a=0.75):
    if k not in mem:
        mem[k] = p
        return p
    ox, oy = mem[k]
    x = ox*a + p[0]*(1-a)
    y = oy*a + p[1]*(1-a)
    mem[k] = (x, y)
    return (x, y)

# =========================
# DRAW LIMB (IMPORTANT FIX)
# =========================

def draw_limb(img, start, end, pivot=(0, 0.2)):
    """
    pivot = (x,y) in normalized sprite space:
    (0,0)=top-left, (1,1)=bottom-right

    For arms:
    - shoulder: pivot near top (0.5, 0.1)
    - elbow: pivot near top again for lower arm
    """

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    dist = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx))

    # scale to bone length
    scaled = pygame.transform.scale(
        img,
        (max(2, int(dist)), img.get_height())
    )

    # pivot in pixel space
    pivot_x = int(scaled.get_width() * pivot[0])
    pivot_y = int(scaled.get_height() * pivot[1])

    rotated = pygame.transform.rotate(scaled, -angle_deg)

    # offset correction (THIS IS THE KEY FIX)
    rot_pivot = pygame.math.Vector2(pivot_x, pivot_y).rotate(angle_deg)

    rect = rotated.get_rect()
    rect.center = (start[0] + rot_pivot.x, start[1] + rot_pivot.y)

    screen.blit(rotated, rect)

# =========================
# MEDIAPIPE
# =========================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

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

    screen.fill((20,20,20))

    if res.pose_landmarks:

        lm = res.pose_landmarks.landmark

        def P(i):
            return smooth(i, (lm[i].x*WIDTH, lm[i].y*HEIGHT))

        ls, rs = P(11), P(12)
        le, re = P(13), P(14)
        lw, rw = P(15), P(16)

        lh, rh = P(23), P(24)
        lk, rk = P(25), P(26)
        la, ra = P(27), P(28)

        nose = P(0)

        chest = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hip = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)

        # =========================
        # LIMBS (CORRECT SYSTEM)
        # =========================

        # draw_limb(upper_arm, ls, le)
        # draw_limb(lower_arm, le, lw)

        # draw_limb(upper_arm, rs, re)
        # draw_limb(lower_arm, re, rw)

        # draw_limb(upper_leg, lh, lk)
        # draw_limb(lower_leg, lk, la)

        # draw_limb(upper_leg, rh, rk)
        # draw_limb(lower_leg, rk, ra)

        # specify the correct pivot *****IMPORTANT
        # upper arm (shoulder pivot)
        draw_limb(upper_arm, ls, le, pivot=(0.5, 0.1))
        draw_limb(upper_arm, rs, re, pivot=(0.5, 0.1))

        # lower arm (elbow pivot)
        draw_limb(lower_arm, le, lw, pivot=(0.5, 0.1))
        draw_limb(lower_arm, re, rw, pivot=(0.5, 0.1))

        # upper leg (hip pivot)
        draw_limb(upper_leg, lh, lk, pivot=(0.5, 0.1))
        draw_limb(upper_leg, rh, rk, pivot=(0.5, 0.1))

        # lower leg (knee pivot)
        draw_limb(lower_leg, lk, la, pivot=(0.5, 0.1))
        draw_limb(lower_leg, rk, ra, pivot=(0.5, 0.1))

        # torso (simple stable placement)
        torso_scaled = pygame.transform.scale(torso, (120, 160))
        screen.blit(torso_scaled, torso_scaled.get_rect(center=hip))

        head_scaled = pygame.transform.scale(head, (80, 80))
        screen.blit(head_scaled, head_scaled.get_rect(center=(chest[0], chest[1]-80)))

        # debug joints
        for p in [ls,rs,le,re,lw,rw,lh,rh,lk,rk,la,ra]:
            pygame.draw.circle(screen, (255,200,0), (int(p[0]), int(p[1])), 4)

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
