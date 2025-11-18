# Airport Unattended Item Tracking System
<p align="center">
  <img src="ChatGPT Image Aug 12, 2025, 02_53_36 PM.png" width="500" />
</p>


## 📌 Overview
This project is an **AI-powered surveillance tool** designed for airport security monitoring.  
It automatically detects **handbags, backpacks, and suitcases**, tracks them across frames, and analyzes their movement to determine whether an item is **moving** or **stationary**.

The system uses:
- **YOLO Object Detection** (converted to **OpenVINO** for faster inference)
- **BoT-SORT Multi-Object Tracking** (with ReID embedding model)
- **Movement Analysis Algorithms** (lateral, depth, and direction)
- **Real-Time Video Processing** with visual annotations and detailed logs

---

## 🎯 Use Case
Airports face the challenge of unattended baggage, which can pose **security risks**.  
Our tracker:
- Detects baggage in surveillance footage
- Tracks items consistently even when they leave and re-enter the scene
- Determines whether they have been moved
- Alerts/logs suspicious activity

---

## ⚙️ System Architecture

**Flow Diagram**:

Video Input
↓
Frame Extraction (VideoPlayerOffline)
↓
YOLO Detection (OpenVINO Model)
↓
Class Filtering (Only Airport Items: handbag/backpack/suitcase)
↓
BoT-SORT Tracking (Persistent IDs)
↓
Tracked Objects Database (Bounding Boxes per Frame)
↓
FinalMovementAnalyzer
├─ Lateral Movement Detection
├─ Depth Movement Detection
├─ Main Direction Logging
└─ False Positive Filtering
↓
Visualization & Logging
↓
Display Output / Save Logs
