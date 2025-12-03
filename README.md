# Face Hero – A Real-Time Face & Gesture Controlled Rhythm Game

Face Hero is a real-time rhythm game inspired by Beat Saber and Guitar Hero —  
but instead of using a keyboard or VR controllers, **your face and hands ARE the controller**.

Built using Python, OpenCV, and MediaPipe, Face Hero turns head movements and hand gestures into gameplay actions.  
All motion is tracked live using custom face-tracking pipelines, smoothing filters, head-pose estimation, and gesture detection.

This repo showcases a complete real-time vision-powered game system.

**Video Demo:** [Click to watch](https://drive.google.com/file/d/1pyu8bpDnGsdfxoNMzEo-Ct6Bdwmroypk/view?usp=drive_link)


---

## 🚀 What Is Face Hero?

Face Hero is a rhythm-action game where:
- Your **head movement** controls the directional actions
- Your **head pose** (yaw/pitch) controls turning/tilting actions
- Your **hands** trigger gestures like wave, clap, hands-up
- You play to music beats, with prompts appearing in real time
- You gain XP, level-up, build combos, and activate ULT bursts
- Visual effects, UI, and audio feedback respond instantly to your motions

Think of it like Beat Saber —  
but instead of lightsabers, **you play with your face**.

---

## 🧠 How It Works (CV Pipeline)

### 1. **Face Detection (MediaPipe)**
- Detects faces each detection cycle  
- Converts relative bounding boxes to pixel space  
- Filters unstable tiny/edge detections

### 2. **Tracking**
Face Hero uses a hybrid detection + tracking pipeline:

#### **CentroidTracker**
- Lightweight ID tracker  
- Good baseline for stable faces  
- Used for comparisons

#### **CorrelationTrackerManager (CSRT/KCF)**
 
*The system runs:*
- CSRT / KCF correlation trackers between detection frames
- Reinitializes trackers on detection cycles
- Maintains stable bounding boxes & trails
- Performs IoU matching and lost-frame management

This produces smooth real-time bounding boxes even when MediaPipe isn't running every frame.

### 3. **Motion Extraction**
*From each tracked face:*
- **dx** → horizontal movement → MOVE LEFT/RIGHT  
- **dz** → size change (depth) → LEAN IN/OUT  
- Both smoothed using exponential moving averages

### 4. **Head Pose Estimation**
*Based on MediaPipe keypoints:*
- **yaw** → TURN LEFT/RIGHT  
- **pitch** → TILT UP/DOWN  

### 5. **Hand Gesture Recognition**
*Using MediaPipe Hands:*
- **WAVE** → lateral wrist movement  
- **CLAP** → wrists close together  
- **HANDS UP** → wrists above head region

*Gestures override face movement when triggered.*

---

## 🎮 Gameplay & Features

### Core Mechanics
- Beat-based prompts  
- Perfect hits, misses, reaction timing  
- Combo multiplier  
- XP bar and level-up  
- ULT mode when high combo achieved  

### Visual Effects
- Face aura around tracked faces  
- Beat ripples  
- Floating score text  
- Neon-style HUD  
- Prompt lane with beat pops  

### Audio
- Background music  
- Hit/miss sound effects  
- Voice prompts for actions  

### Data Logging
*Every frame logs:*
- face ID  
- centroid  
- bounding box  
- dx, dz  
- yaw, pitch  
- gesture  
- reaction time  
- success/miss  

Output: `face_logs.csv`

### Offline Analysis
`plot_trajectories.py` visualizes face trajectories over time.

---

## 🧩 Installation

### 1. Create and activate virtual environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependancies
```bash
pip install -r requirements.txt
```

#### 3. Run the Game
```bash
python main.py
```

*What to expect:*

- your webcam feed
- face boxes with glowing FX
- HUD with prompts
- beat cycles
- SFX and level-ups
- real-time motion tracking driving gameplay

#### 4. Plot Trajectories
```bash
python plot_trajectories.py
```

*This reads face_logs.csv and visualizes:*

face movement
ID consistency
smoothness of the tracker
motion over time

---

**📝 Notes**

- Works best in good lighting
- Webcam required
- Windows/macOS/Linux supported
- Python 3.9+ recommended

