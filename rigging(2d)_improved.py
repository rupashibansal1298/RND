
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

old_points = {}              # to smoothen the movements

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

        def smooth(name,new_point):

            if name not in old_points:
                old_points[name]=new_point

            x = int(0.8*old_points[name][0] + 0.2*new_point[0])
            y = int(0.8*old_points[name][1] + 0.2*new_point[1])

            old_points[name]=(x,y)

            return (x,y)

        ls = smooth("ls",ls)
        rs = smooth("rs",rs)
        le = smooth("le",le)
        re = smooth("re",re)
        lw = smooth("lw",lw)
        rw = smooth("rw",rw)
        lh = smooth("lh",lh)
        rh = smooth("rh",rh)
        lk = smooth("lk",lk)
        rk = smooth("rk",rk)
        la = smooth("la",la)
        ra = smooth("ra",ra)

        #pygame.draw.line(screen,(255,255,255),ls,rs,5)

        # pygame.draw.line(screen,(0,255,0),ls,le,5)
        # pygame.draw.line(screen,(0,255,0),le,lw,5)

        # pygame.draw.line(screen,(0,255,0),rs,re,5)
        # pygame.draw.line(screen,(0,255,0),re,rw,5)

        # pygame.draw.line(screen,(255,0,0),lh,rh,5)

        # pygame.draw.line(screen,(0,0,255),lh,lk,5)
        # pygame.draw.line(screen,(0,0,255),lk,la,5)

        # pygame.draw.line(screen,(0,0,255),rh,rk,5)
        # pygame.draw.line(screen,(0,0,255),rk,ra,5)

        # for p in [ls,rs,le,re,lw,rw,lh,rh,lk,rk,la,ra]:
        #     pygame.draw.circle(screen,(255,255,0),p,8)

        # instead of just this, we want a more realistic looking skeleton

                # ---------------------------
        # HEAD
        # ---------------------------

        head_center = (
            (ls[0] + rs[0]) // 2,
            min(ls[1], rs[1]) - 40
        )

        pygame.draw.circle(
            screen,
            (255,220,180),
            head_center,
            30
        )

        # eyes

        pygame.draw.circle(
            screen,
            (0,0,0),
            (head_center[0]-10, head_center[1]-5),
            3
        )

        pygame.draw.circle(
            screen,
            (0,0,0),
            (head_center[0]+10, head_center[1]-5),
            3
        )

        # ---------------------------
        # NECK
        # ---------------------------

        neck = (
            (ls[0]+rs[0])//2,
            (ls[1]+rs[1])//2
        )

        pygame.draw.line(
            screen,
            (255,255,255),
            head_center,
            neck,
            6
        )

        # ---------------------------
        # TORSO
        # ---------------------------

        hip_center = (
            (lh[0]+rh[0])//2,
            (lh[1]+rh[1])//2
        )

        pygame.draw.line(
            screen,
            (100,200,255),
            neck,
            hip_center,
            10
        )

        # ---------------------------
        # SHOULDER LINE
        # ---------------------------

        pygame.draw.line(
            screen,
            (255,255,255),
            ls,
            rs,
            8
        )

        # ---------------------------
        # LEFT ARM
        # ---------------------------

        pygame.draw.line(
            screen,
            (0,255,0),
            ls,
            le,
            8
        )

        pygame.draw.line(
            screen,
            (0,180,0),
            le,
            lw,
            8
        )

        # ---------------------------
        # RIGHT ARM
        # ---------------------------

        pygame.draw.line(
            screen,
            (0,255,0),
            rs,
            re,
            8
        )

        pygame.draw.line(
            screen,
            (0,180,0),
            re,
            rw,
            8
        )

        # ---------------------------
        # HIP LINE
        # ---------------------------

        pygame.draw.line(
            screen,
            (255,255,255),
            lh,
            rh,
            8
        )

        # ---------------------------
        # LEFT LEG
        # ---------------------------

        pygame.draw.line(
            screen,
            (0,100,255),
            lh,
            lk,
            8
        )

        pygame.draw.line(
            screen,
            (0,0,255),
            lk,
            la,
            8
        )

        # ---------------------------
        # RIGHT LEG
        # ---------------------------

        pygame.draw.line(
            screen,
            (0,100,255),
            rh,
            rk,
            8
        )

        pygame.draw.line(
            screen,
            (0,0,255),
            rk,
            ra,
            8
        )

        # ---------------------------
        # JOINTS
        # ---------------------------

        for p in [
            neck,
            ls,rs,
            le,re,
            lw,rw,
            lh,rh,
            lk,rk,
            la,ra
        ]:

            pygame.draw.circle(
                screen,
                (255,255,0),
                p,
                10
            )

    pygame.display.update()
    clock.tick(60)

cap.release()
pygame.quit()