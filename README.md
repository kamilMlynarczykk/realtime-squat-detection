
# Automatic Squat Counting with Voice Commands

This project uses Python libraries to create an application that counts squats using computer vision and voice recognition.

## Features
- Counts squats using body pose estimation from two camera angles (front and side).
- Provides visual feedback on proper form, including arm position and knee angle.
- Uses voice commands to start, repeat, and stop workouts.
- Displays the remaining number of squats in the workout.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- SpeechRecognition
- NumPy
- PyAudio, Vosk (offline speech recognition)

## Installation

First, install the required libraries using the following command:

```bash
pip install opencv-python mediapipe speechrecognition numpy pyaudio vosk
```

Make sure you have two cameras connected to your computer, can be laptop webcam and phone camera connected via cable or external software like IVCams.

## Instructions

1. Connect two cameras to your computer (one for the front view, one for the side view).
2. Run the application by executing the following command:

```bash
python squat_counter.py
```

## How It Works

- The application initializes two camera captures (front and side views) and uses MediaPipe for pose estimation.
- It listens for voice commands using the SpeechRecognition library.

### Voice Commands:
- **Start workout (number) squats:** Begins a new workout with the specified number of squats.
- **Repeat workout:** Restarts the current workout without changing the squat count.
- **Stop workout:** Stops the current workout session.

## Visual Feedback

- A combined image of both camera views (front and side) is displayed.
- Body landmarks are drawn on top of the camera images.
- The current squat count and the number of remaining squats are displayed.
- Visual cues are provided to ensure proper arm position and knee angle during squats.

## Limitations
- The user’s body must be clearly visible from both the front and side camera angles for accurate squat counting.
- The accuracy of the squat count may be affected by lighting conditions and clothing.
- This is a basic implementation that can be enhanced with additional features like form correction or exercise history tracking.

## Possible future upgrades
- Create more visually appealing UI

## Disclaimer
- Offline recognition is heavilly dependent on model and machine used to run the program.
- While testing online model (recognize_google) was vastly more accurate, maybe having a better microphone would close the gap between these two methods
- Download models used in testing: https://alphacephei.com/vosk/models
