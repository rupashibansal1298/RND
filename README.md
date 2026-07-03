# 2D Character Rigging Using MediaPipe and Pygame

## Overview

This project demonstrates the complete workflow of building a **2D character rigging system** controlled by real-time human pose estimation. Starting from the basics of pose detection and angle calculation, the project gradually evolves into a fully rigged 2D character whose body parts rotate naturally according to the user's movements captured through a webcam.

The project was developed step-by-step to understand the concepts behind skeletal animation rather than directly implementing the final solution.

---

## Features

- Real-time pose detection using MediaPipe
- Joint angle calculation
- Skeleton visualization
- Rectangle-based body representation
- Image-based 2D character rigging
- Rotation of body parts around custom pivot points
- Improved sprite alignment using transparent padding
- Real-time animation using Pygame

---

## Project Workflow

The repository contains multiple Python files, each representing a stage in the development process.

first, download the required libraries and assets

FOR THE BG screen:
# screen.fill((30,30,30))__ 
this gives us the figure on a black bg, not the actual webcam feed

    #this gives us the feed view:

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (WIDTH, HEIGHT))



    surface = pygame.surfarray.make_surface(frame.swapaxes(0,1))

    screen.blit(surface, (0,0))
    

### 1. Pose Detection

**File:** `pose_extraction.py`

- Detects human body landmarks using MediaPipe Pose.
- Draws body landmarks and connections.
- Forms the foundation for all later stages.

**Concepts learned**

- Pose estimation
- Landmark extraction
- Coordinate systems

---

### 2. Angle Tracking

**File:** `angle_tracker.py`

Calculates joint angles using three body landmarks.

These angles are later used to rotate body parts naturally.

**Concepts learned**

- Vector mathematics
- Joint angle calculation
- Shoulder and elbow rotation

---

### 3. Skeleton Construction
**File:** `draw_each_frame.py`




The detected landmarks are connected using lines to create a stick-figure skeleton.

This verifies that landmark positions and body hierarchy are correct before introducing graphics.

**Concepts learned**

- Parent-child relationships
- Body hierarchy
- Skeleton visualization

---

### 4. Head and Torso Integration
**File:** `draw_each_frame_improved.py`

Simple graphical representations of the head and torso are added onto the skeleton.

This stage bridges the gap between a stick figure and a graphical character.

---

### 5. Rectangle-Based Rigging

**File:** `rectangular_rig.py`

Each body segment is represented by a rectangle.

The rectangles rotate according to the computed joint angles, allowing the rigging logic to be tested without worrying about image alignment.

**Concepts learned**

- Rotation around pivot points
- Transformation hierarchy
- Segment positioning

---

### 6. Image-Based Character Rigging

**File:** `hierarchy_rig_(torso size not correct).py`
(the pivot locations can be changed according to you_)
Rectangles are replaced with individual image cutouts representing:

- Head
- Torso
- Upper arms
- Lower arms
- Upper legs
- Lower legs

Each sprite rotates around carefully chosen pivot points to simulate realistic movement.

**Concepts learned**

- Sprite transformations
- Pivot-based rotation
- Image alignment

---

### 7. Improved Rigging

**File:** `motion_character.py`

The final improvement involves adding transparent padding around body-part images.

This allows pivot points to remain inside the image while preventing clipping during rotation, resulting in smoother and more realistic animations.

**File:** `depth_final.py`

this file also uses the d coordinate from pose estimation to decide whether the arm is in front of the torso or the back, and gives the output accordingly
---

## Folder Structure

```
project/
│
├── assets/
│   └── images/
│       ├── head.png
│       ├── torso.png
│       ├── upper_arm.png
│       ├── lower_arm.png
│       ├── upper_leg.png
│       ├── lower_leg.png
│       └── ...
│
├── pose_extraction.py
├── angle_tracker.py
├── rectangular_rig.py
├── actual_2d_rig.py
├── rigging(2d)_improved.py
├── motion_character.py
├── hierarchy_rig.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Pygame
- NumPy

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

Move into the project directory

```bash
cd your-repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run any stage of the project

```bash
python actual_2d_rig.py
```

or

```bash
python rigging(2d)_improved.py
```

---

## Future Improvements

- Full inverse kinematics
- Better sprite interpolation
- Finger tracking
- Facial expression animation
- Character scaling based on camera distance
- Multiple character support
- Custom avatar loading
- Physics-based secondary motion

---

## Learning Outcomes

This project helped in understanding:

- Human pose estimation
- MediaPipe Pose landmarks
- Joint angle computation
- Hierarchical transformations
- Pivot-based image rotation
- 2D skeletal animation
- Real-time graphics rendering using Pygame
- Character rigging fundamentals

---

## Acknowledgements

- MediaPipe for real-time pose estimation
- OpenCV for webcam processing
- Pygame for rendering and animation

---

## Author

Developed as a learning project to explore the complete pipeline of **2D character rigging driven by real-time human pose estimation**, progressing from basic pose detection to a fully animated image-based character.
