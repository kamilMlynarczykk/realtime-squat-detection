import json
import time
import re
import cv2
import mediapipe as mp
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

KEY_POINTS_FRONT = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

KEY_POINTS_SIDE = [
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX
]

# cameras
front_camera = 0
side_camera = 1
microphone = 3

# cv2 text params
line_thickness = 2
font_scale = 1

# path to offline model
model_path = "{DIRECTORY_WITH_VOSK_MODELS}\\{VOSK-MODEL}}"  #for example: "Offline-Vosk-Models\\vosk-model-small-en-us-0.15"

class SquatCounterApp:
    def __init__(self):
        self.wrongCommand = False
        self.command = None
        self.no_number_in_command = None
        self.countdown = 3
        self.arm_to_torso_position = None
        self.arm_position_correct = None
        self.side_visible = None
        self.front_visible = None
        self.results_front = None
        self.results_side = None
        self.front_frame = None
        self.side_frame = None
        self.lock = threading.Lock()
        self.counter = 0
        self.notLowEnough = False
        self.stage = None
        self.frames_below_90 = 0
        self.debounce_frames = 5
        self.start_time = None
        self.listening_for_start = False
        self.speech_thread = None
        self.start_speech_command_received = False
        self.repeat_speech_command_received = False
        self.stop_speech_command_received = False
        self.cap_front_camera = None
        self.pose_front_camera = None
        self.cap_side_camera = None
        self.pose_side_camera = None
        self.squats_todo = 2
        self.camera_thread_active = False

    def reset(self):
        self.counter = 0
        self.notLowEnough = False
        self.stage = None
        self.frames_below_90 = 0
        self.debounce_frames = 5
        self.start_time = None
        self.listening_for_start = True
        self.speech_thread = None
        self.start_speech_command_received = False
        self.repeat_speech_command_received = False

    def start(self):
        try:
            # Initialize webcam capture if it's not already initialized
            if self.cap_front_camera is None:
                print("initializing front cap...")
                self.cap_front_camera = cv2.VideoCapture(front_camera)
                if not self.cap_front_camera.isOpened():
                    raise Exception("Failed to open the camera.")

            # Initialize MediaPipe pose if necessary
            if self.pose_front_camera is None:
                print("initializing front pose...")
                self.pose_front_camera = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            if self.cap_side_camera is None:
                print("initializing side cap...")
                self.cap_side_camera = cv2.VideoCapture(side_camera)
                if not self.cap_side_camera.isOpened():
                    raise Exception("Failed to open the camera.")

            # Initialize MediaPipe pose if necessary
            if self.pose_side_camera is None:
                print("initializing side pose...")
                self.pose_side_camera = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            self.listening_for_start = True
            self.start_speech_thread()
            if not self.camera_thread_active:
                self.open_camera()
        except Exception as e:
            print(f"An error occurred during startup: {str(e)}")
            self.reset()
            exit(1)

    def start_speech_thread(self):
        if self.speech_thread is None or not self.speech_thread.is_alive():
            print("Starting speech thread.")
            self.speech_thread = threading.Thread(target=self.listen_for_command)
            self.speech_thread.daemon = True
            self.speech_thread.start()

    def start_cameras_threads(self):
        front_camera_thread = threading.Thread(target=self.process_front_camera)
        side_camera_thread = threading.Thread(target=self.process_side_camera)

        front_camera_thread.start()
        side_camera_thread.start()

    def open_camera(self):
        self.start_cameras_threads()
        self.camera_thread_active = True
        while True:
            if self.front_frame is not None and self.side_frame is not None:
                # self.image.flags.writeable = True
                image = self.concat_two_images(self.front_frame, self.side_frame)
                image = self.draw_top_rectangle(image, 200)

                self.ensure_user_is_visible(image)

                # Display remaining squats
                remaining_squats = self.squats_todo - self.counter
                if self.countdown <= 0 and not self.listening_for_start:
                    if remaining_squats > 0:
                        cv2.putText(image, f'Squats remaining: {remaining_squats}', (200, 100),
                                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), line_thickness, cv2.LINE_AA)
                    # Workout complete notification
                    if remaining_squats <= 0:
                        cv2.putText(image, "Workout complete! Say 'repeat workout' or 'stop workout'",
                                    (10, 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)
                        self.start_speech_thread()
                        self.wrong_command_heard(image)

                cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('test', image)

                if cv2.waitKey(10) & 0xFF == ord('q') or self.stop_speech_command_received:
                    self.stop()

    def ensure_user_is_visible(self, image):
        with self.lock:
            if self.front_visible and self.side_visible:
                if self.start_time is None and self.listening_for_start:
                    if self.start_speech_command_received:
                        self.start_time = time.time()
                        self.start_speech_command_received = False
                        self.listening_for_start = False
                    else:
                        cv2.putText(image, "Say 'START WORKOUT (number) squats' to begin", (10, 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)
                        if self.no_number_in_command and not self.wrongCommand:
                            cv2.putText(image,
                                        "No number found in the start workout command.",
                                        (10, 130),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)
                        self.wrong_command_heard(image)
                elif self.start_time:
                    elapsed_time = time.time() - self.start_time
                    self.countdown = max(0, 3 - int(elapsed_time))
                    if self.countdown > 0:
                        cv2.putText(image, f'Starting in: {self.countdown} seconds...', (200, 100),
                                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), line_thickness, cv2.LINE_AA)

    def wrong_command_heard(self, image):
        if self.wrongCommand:
            cv2.putText(image, "Heard Command:", (10, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)
            cv2.putText(image, f"{self.command}", (10, 170),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), line_thickness, cv2.LINE_AA)

    def front_squat_logic(self, front_image):
        try:
            landmarks = self.results_front.pose_landmarks.landmark
            self.front_visible = all(landmarks[point.value].visibility > 0.5 for point in KEY_POINTS_FRONT)
            front_image.flags.writeable = True
            with self.lock:
                if not self.front_visible:
                    self.start_time = None
                    cv2.putText(front_image, "Ensure full body is visible", (90, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)
                elif self.side_visible and self.countdown <= 0 and not self.listening_for_start:
                    left_knee_angle, right_knee_angle = self.calculateKneesAngle(landmarks)
                    self.display_up_down_text(front_image, left_knee_angle, right_knee_angle)
                    self.display_knee_angles(front_image, left_knee_angle, right_knee_angle)
                    self.countRepetition(front_image, left_knee_angle, right_knee_angle)
        except Exception as e:
            cv2.putText(front_image, "Ensure full body is visible", (90, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)

    def side_squat_logic(self, side_image):
        try:
            landmarks = self.results_side.pose_landmarks.landmark
            self.side_visible = all(landmarks[point.value].visibility > 0.5 for point in KEY_POINTS_SIDE)
            side_image.flags.writeable = True
            with self.lock:
                if not self.side_visible:
                    self.start_time = None
                    cv2.putText(side_image, "Ensure full body is visible", (0, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)
                else:
                    self.display_arm_placement_text(landmarks, side_image)

        except Exception as e:
            cv2.putText(side_image, "Ensure full body is visible", (0, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness, cv2.LINE_AA)

    def display_arm_placement_text(self, landmarks, side_image):
        arms_straight_in_elbows = self.check_elbow_straight(landmarks)
        arms_parallel_to_feet = self.check_arm_feet_parallel(landmarks)
        if not arms_straight_in_elbows and not arms_parallel_to_feet:
            cv2.putText(side_image, "Straighten your arms!", (0, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)
            cv2.putText(side_image, "Place your arms in front!", (0, 200),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)

        elif not arms_parallel_to_feet:
            cv2.putText(side_image, "Place your arms in front!", (0, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)
        elif not arms_straight_in_elbows:
            cv2.putText(side_image, "Straighten your arms!", (0, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)

    def process_front_camera(self):
        while True:
            ret_f, frame_f = self.cap_front_camera.read()
            if not ret_f:
                break

            image_front = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
            image_front = self.draw_top_rectangle(image_front, 50)
            cv2.putText(image_front, "Front Camera", (170, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness,
                        cv2.LINE_AA)
            self.results_front = self.pose_front_camera.process(image_front)
            image_front.flags.writeable = True
            image_front = cv2.cvtColor(image_front, cv2.COLOR_RGB2BGR)

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image_front, self.results_front.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 26, 230), thickness=2, circle_radius=2))

            self.front_squat_logic(image_front)
            with self.lock:
                self.front_frame = image_front

    def process_side_camera(self):
        while True:
            ret_s, frame_s = self.cap_side_camera.read()
            if not ret_s:
                break

            image_side = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
            image_side = cv2.rotate(image_side, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_side = self.draw_top_rectangle(image_side, 50)
            cv2.putText(image_side, "Side Camera", (150, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), line_thickness,
                        cv2.LINE_AA)
            self.results_side = self.pose_side_camera.process(image_side)
            image_side.flags.writeable = True
            image_side = cv2.cvtColor(image_side, cv2.COLOR_RGB2BGR)

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image_side, self.results_side.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            with self.lock:
                self.side_frame = image_side

            self.side_squat_logic(image_side)

    def concat_two_images(self, img1, img2):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        if h1 > h2:
            img2 = cv2.resize(img2, (int(w2 * h1 / h2), h1))
        elif h2 > h1:
            img1 = cv2.resize(img1, (int(w1 * h2 / h1), h2))

        # print(img1.shape[:2])
        return np.concatenate((img1, img2), axis=1)

    def draw_top_rectangle(self, image, rect_height):
        rect_width = image.shape[1]
        rectangle = np.zeros((rect_height, rect_width, 3), dtype=np.uint8)
        rectangle[:] = (0, 0, 0)
        return np.vstack((rectangle, image))

    def draw_left_rectangle(self, image, rect_width):
        rect_height = image.shape[0]
        rectangle = np.zeros((rect_height, rect_width, 3), dtype=np.uint8)
        rectangle[:] = (0, 0, 0)
        return np.hstack((image, rectangle))

    def stop_audio_input(self, stream, mic_source):
        stream.stop_stream()
        stream.close()
        mic_source.terminate()


    def listen_for_command(self):

        print("Initializing model...")
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)

        try:
            # Initialize the microphone
            microphone_source = pyaudio.PyAudio()
            stream = microphone_source.open(format=pyaudio.paInt16,channels=1, rate=16000, input=True, frames_per_buffer=8192, input_device_index=3)
            print(f"Microphone initialized")
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            return

        # Terminate the PyAudio object
        print("Listening...")
        while True:
            try:
                audio = stream.read(2048)
                if recognizer.AcceptWaveform(audio):
                    result = json.loads(recognizer.Result())
                    self.command = result['text']
                else:
                    continue
                print(f"Heard command: {self.command}")

                if 'start work out' in self.command:
                    self.wrongCommand = False
                    # Extract the number of squats from the command
                    match = re.search(r'start work out (\d+) squats', self.command)
                    print(match)
                    if match:
                        self.no_number_in_command = False
                        self.squats_todo = int(match.group(1))
                        print(self.squats_todo)
                        self.start_speech_command_received = True
                        self.stop_audio_input(stream, microphone_source)
                        break
                    else:
                        self.no_number_in_command = True
                        self.start_speech_command_received = False
                elif 'repeat work out' in self.command:
                    self.wrongCommand = False
                    print("Repeat workout command received.")
                    self.restart_workout()
                    self.stop_audio_input(stream, microphone_source)
                    break
                elif 'stop work out' in self.command:
                    self.wrongCommand = False
                    self.stop_speech_command_received = True
                    self.stop_audio_input(stream, microphone_source)
                    break
                else:
                    self.wrongCommand = True
                print(self.wrongCommand)


            except Exception as e:
                print(e)

    def restart_workout(self):
        """Restart the workout by resetting and reinitializing."""
        print("Restarting workout...")
        self.reset()  # Reset internal state
        print("Stoping speech thread")
        self.stop_speech_thread()
        print("Starting workout...")
        self.start()  # Start the workout again

    def calculateKneesAngle(self, landmarks):
        points = {
            'left': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
            'right': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
        }

        angles = {}
        for side, [hip, knee, ankle] in points.items():
            angles[side] = self.calculate_angle(
                [landmarks[hip.value].x, landmarks[hip.value].y],
                [landmarks[knee.value].x, landmarks[knee.value].y],
                [landmarks[ankle.value].x, landmarks[ankle.value].y]
            )

        return angles['left'], angles['right']

    def countRepetition(self, image, left_knee_angle, right_knee_angle):
        # If both knees are extended above 160 degrees, the user is standing ("up" position)

        if left_knee_angle > 160 and right_knee_angle > 160:
            self.stage = "up"
            self.frames_below_90 = 0
            self.notLowEnough = False  # Reset after notification

        # If both knees are below 100 degrees, the user is in the squatting position
        if left_knee_angle < 100 and right_knee_angle < 100:
            self.frames_below_90 += 1
            self.notLowEnough = False  # No need for notification, user went low enough

        else:
            self.frames_below_90 = 0
            self.notLowEnough = True  # Flag for when user didn't squat low enough

        # Increment the counter when user goes from "up" to "down" position after reaching below 90 degrees
        if self.frames_below_90 >= self.debounce_frames and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.frames_below_90 = 0

    def display_up_down_text(self, image, left_knee_angle, right_knee_angle):
        remaining_squats = self.squats_todo - self.counter
        if remaining_squats > 0:
            if self.notLowEnough:
                if left_knee_angle > 100 and right_knee_angle > 100:
                    cv2.putText(image, "Squat down!", (20, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)
            else:
                cv2.putText(image, "Squat up!", (20, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)

    def display_knee_angles(self, image, left_knee_angle, right_knee_angle):
        cv2.putText(image, f"Left knee angle: {int(left_knee_angle)}°", (20, 450),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)
        cv2.putText(image, f"Right knee angle: {int(right_knee_angle)}°", (20, 500),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), line_thickness, cv2.LINE_AA)

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def check_elbow_straight(self, landmarks):
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        elbow_angle = self.calculate_angle(right_wrist, right_elbow, right_shoulder)
        return elbow_angle >= 160

    def check_arm_feet_parallel(self, landmarks):
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        a1 = self.calculate_linear_function(right_wrist, right_shoulder)
        a2 = self.calculate_linear_function(right_heel, right_foot_index)
        return np.fabs(a1 - a2) < 0.3

    def calculate_linear_function(self, one_point, second_point):
        x1, y1 = one_point[:2]
        x2, y2 = second_point[:2]

        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
        else:
            raise ValueError("Cannot calculate slope: x1 and x2 are the same, resulting in a vertical line.")
        return m

    def stop_speech_thread(self):
        self.listening_for_start = False

        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2)

    def stop(self):
        print("Stopping workout and releasing resources...")
        self.cap_front_camera.release()
        self.cap_side_camera.release()
        cv2.destroyAllWindows()
        exit(0)


def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


def list_cameras():
    print([(cv2.VideoCapture(index)).read()[0] for index in range(10)])


if __name__ == "__main__":
    #list_microphones()
    list_cameras()
    app = SquatCounterApp()
    app.start()
    # Open the microphone stream

    #model_path = "vosk-model-en-us-0.42-gigaspeech"
    # Initialize the model with model-path

    #model = Model(model_path)
    #rec = KaldiRecognizer(model, 16000)
    #p = pyaudio.PyAudio()
    #stream = p.open(format=pyaudio.paInt16,
    #                channels=1,
    #                rate=16000,
    #                input=True,
    #                frames_per_buffer=8192,
    #                input_device_index=3)
    #print("Listening...")
    #recognized_text = 0
    #while True:
    #    data = stream.read(4096)  # read in chunks of 4096 bytes
    #    if rec.AcceptWaveform(data):  # accept waveform of input voice
    #        # Parse the JSON result and get the recognized text
    #        result = json.loads(rec.Result())
    #        recognized_text = result['text']
    #        print(f"Recognized text: \"{recognized_text}\"")
#
    #    if recognized_text != 0 and "terminate" in recognized_text.lower():
    #        print("Termination keyword detected. Stopping...")
    #        break
#
    #stream.stop_stream()
    #stream.close()
#
    ## Terminate the PyAudio object
    #p.terminate()