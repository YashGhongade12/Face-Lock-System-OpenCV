# Face Lock System using OpenCV

A simple face-lock project built with Python and OpenCV.  
The system captures the user's face on first run and saves it as an authorized image.  
On the next run, it compares the live webcam face with the saved face using pixel-based similarity.  
If both faces match, the Desktop folder unlocks; otherwise, access is denied.

## Features
- Face detection using Haar Cascade
- Face registration (first-time setup)
- Face matching using grayscale pixel comparison
- Unlocks Desktop folder on successful match

## Technologies
- Python
- OpenCV
- NumPy

## How to Run
1. Run the program the first time → press **S** to register your face.
2. Run again → system verifies your face.
3. If matched → Desktop folder opens.

## Controls
- **S** → Save/Register face
- **Q** → Quit
