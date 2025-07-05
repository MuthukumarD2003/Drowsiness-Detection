import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import dlib
import imutils
import threading
import time
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import os
import json
from twilio.rest import Client

# Constants
CONFIG_FILE = "user_config.json"
SHAPE_PREDICTOR = "models/shape_predictor_68_face_landmarks.dat"
WARNING_SOUND = "music.wav"  # First alert sound file (mild alert) - plays from 5-10 seconds
DANGER_SOUND = "warning_alarm.mp3"  # Second alert sound file (urgent alert) - plays after 10 seconds

# Default thresholds
EYE_AR_THRESH = 0.25
WARNING_TIME = 5  # seconds for first alert
DANGER_TIME = 10  # seconds for second alert and SMS
FPS_ESTIMATE = 30


class DrowsinessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        # Create a notebook with tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Setup tabs
        self.setup_registration_tab()
        self.setup_settings_tab()
        self.setup_sound_tab()  # New tab for sound settings

        # Load user data if exists
        self.load_user_data()

        # Initialize detection variables
        self.is_running = False
        self.detection_thread = None

        # Initialize mixer for different sounds
        mixer.init()
        self.current_sound = None

    def setup_registration_tab(self):
        reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(reg_frame, text="User Registration")

        # User details section
        ttk.Label(reg_frame, text="Driver Information", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                         pady=20)

        ttk.Label(reg_frame, text="Full Name:").grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.name_var, width=40).grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(reg_frame, text="Age:").grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.age_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.age_var, width=10).grid(row=2, column=1, sticky="w", pady=5)

        ttk.Label(reg_frame, text="Emergency Contact Name:").grid(row=3, column=0, sticky="w", padx=20, pady=5)
        self.emergency_name_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.emergency_name_var, width=40).grid(row=3, column=1, sticky="w", pady=5)

        ttk.Label(reg_frame, text="Emergency Contact Number:").grid(row=4, column=0, sticky="w", padx=20, pady=5)
        self.emergency_number_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.emergency_number_var, width=20).grid(row=4, column=1, sticky="w", pady=5)

        ttk.Label(reg_frame, text="Relation:").grid(row=5, column=0, sticky="w", padx=20, pady=5)
        self.relation_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.relation_var, width=20).grid(row=5, column=1, sticky="w", pady=5)

        # Twilio SMS API credentials (for sending SMS alerts)
        ttk.Label(reg_frame, text="Twilio Account SID:").grid(row=6, column=0, sticky="w", padx=20, pady=5)
        self.twilio_sid_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.twilio_sid_var, width=40).grid(row=6, column=1, sticky="w", pady=5)

        ttk.Label(reg_frame, text="Twilio Auth Token:").grid(row=7, column=0, sticky="w", padx=20, pady=5)
        self.twilio_token_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.twilio_token_var, width=40, show="*").grid(row=7, column=1, sticky="w",
                                                                                          pady=5)

        ttk.Label(reg_frame, text="Twilio Phone Number:").grid(row=8, column=0, sticky="w", padx=20, pady=5)
        self.twilio_phone_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.twilio_phone_var, width=20).grid(row=8, column=1, sticky="w", pady=5)

        # Save button
        ttk.Button(reg_frame, text="Save Information", command=self.save_user_data).grid(row=9, column=0, columnspan=2,
                                                                                         pady=20)

        # Start detection button
        self.start_btn = ttk.Button(reg_frame, text="Start Drowsiness Detection", command=self.toggle_detection)
        self.start_btn.grid(row=10, column=0, columnspan=2, pady=10)

    def setup_settings_tab(self):
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        ttk.Label(settings_frame, text="Detection Settings", font=("Arial", 16, "bold")).grid(row=0, column=0,
                                                                                              columnspan=2, pady=20)

        ttk.Label(settings_frame, text="Eye Aspect Ratio Threshold:").grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.ear_threshold_var = tk.StringVar(value=str(EYE_AR_THRESH))
        ttk.Entry(settings_frame, textvariable=self.ear_threshold_var, width=10).grid(row=1, column=1, sticky="w",
                                                                                      pady=5)

        ttk.Label(settings_frame, text="Warning Alert Time (seconds):").grid(row=2, column=0, sticky="w", padx=20,
                                                                             pady=5)
        self.warning_time_var = tk.StringVar(value=str(WARNING_TIME))
        ttk.Entry(settings_frame, textvariable=self.warning_time_var, width=10).grid(row=2, column=1, sticky="w",
                                                                                     pady=5)

        ttk.Label(settings_frame, text="Danger Alert Time (seconds):").grid(row=3, column=0, sticky="w", padx=20,
                                                                            pady=5)
        self.danger_time_var = tk.StringVar(value=str(DANGER_TIME))
        ttk.Entry(settings_frame, textvariable=self.danger_time_var, width=10).grid(row=3, column=1, sticky="w", pady=5)

        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_user_data).grid(row=4, column=0,
                                                                                           columnspan=2, pady=20)

    def setup_sound_tab(self):
        sound_frame = ttk.Frame(self.notebook)
        self.notebook.add(sound_frame, text="Alert Sounds")

        ttk.Label(sound_frame, text="Alert Sound Settings", font=("Arial", 16, "bold")).grid(row=0, column=0,
                                                                                             columnspan=3, pady=20)

        # Warning sound selection
        ttk.Label(sound_frame, text="Warning Sound (5-10s):").grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.warning_sound_var = tk.StringVar(value=WARNING_SOUND)
        ttk.Entry(sound_frame, textvariable=self.warning_sound_var, width=30).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Button(sound_frame, text="Test", command=lambda: self.test_sound(self.warning_sound_var.get())).grid(
            row=1, column=2, padx=5, pady=5)

        # Danger sound selection
        ttk.Label(sound_frame, text="Danger Sound (>10s):").grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.danger_sound_var = tk.StringVar(value=DANGER_SOUND)
        ttk.Entry(sound_frame, textvariable=self.danger_sound_var, width=30).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Button(sound_frame, text="Test", command=lambda: self.test_sound(self.danger_sound_var.get())).grid(
            row=2, column=2, padx=5, pady=5)

        # SMS message customization
        ttk.Label(sound_frame, text="Custom SMS Alert Message:").grid(row=3, column=0, sticky="w", padx=20, pady=5)
        self.sms_message_var = tk.StringVar(
            value="URGENT ALERT: {name} appears to be falling asleep while driving! This message is from {relation}.")
        ttk.Entry(sound_frame, textvariable=self.sms_message_var, width=50).grid(row=3, column=1, columnspan=2,
                                                                                 sticky="w", pady=5)

        ttk.Label(sound_frame,
                  text="Note: Use {name} for driver's name and {relation} for emergency contact relation").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=20, pady=5)

        # Save sound settings button
        ttk.Button(sound_frame, text="Save Sound Settings", command=self.save_user_data).grid(
            row=5, column=0, columnspan=3, pady=20)

    def test_sound(self, sound_file):
        try:
            # Stop any currently playing sound
            mixer.music.stop()
            # Load and play the selected sound
            mixer.music.load(sound_file)
            mixer.music.play()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to play sound file: {str(e)}")

    def save_user_data(self):
        # Check if emergency number is provided
        if not self.emergency_number_var.get():
            messagebox.showerror("Error", "Emergency contact number is required!")
            return

        # Gather all user data
        user_data = {
            "name": self.name_var.get(),
            "age": self.age_var.get(),
            "emergency_name": self.emergency_name_var.get(),
            "emergency_number": self.emergency_number_var.get(),
            "relation": self.relation_var.get(),
            "twilio_sid": self.twilio_sid_var.get(),
            "twilio_token": self.twilio_token_var.get(),
            "twilio_phone": self.twilio_phone_var.get(),
            "ear_threshold": float(self.ear_threshold_var.get() or EYE_AR_THRESH),
            "warning_time": int(self.warning_time_var.get() or WARNING_TIME),
            "danger_time": int(self.danger_time_var.get() or DANGER_TIME),
            "warning_sound": self.warning_sound_var.get(),
            "danger_sound": self.danger_sound_var.get(),
            "sms_message": self.sms_message_var.get()
        }

        # Save to file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(user_data, f)

        messagebox.showinfo("Success", "User information saved successfully!")

    def load_user_data(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    user_data = json.load(f)

                # Set values in UI
                self.name_var.set(user_data.get("name", ""))
                self.age_var.set(user_data.get("age", ""))
                self.emergency_name_var.set(user_data.get("emergency_name", ""))
                self.emergency_number_var.set(user_data.get("emergency_number", ""))
                self.relation_var.set(user_data.get("relation", ""))
                self.twilio_sid_var.set(user_data.get("twilio_sid", ""))
                self.twilio_token_var.set(user_data.get("twilio_token", ""))
                self.twilio_phone_var.set(user_data.get("twilio_phone", ""))
                self.ear_threshold_var.set(str(user_data.get("ear_threshold", EYE_AR_THRESH)))
                self.warning_time_var.set(str(user_data.get("warning_time", WARNING_TIME)))
                self.danger_time_var.set(str(user_data.get("danger_time", DANGER_TIME)))

                # Load sound settings if available
                if "warning_sound" in user_data:
                    self.warning_sound_var.set(user_data["warning_sound"])
                if "danger_sound" in user_data:
                    self.danger_sound_var.set(user_data["danger_sound"])
                if "sms_message" in user_data:
                    self.sms_message_var.set(user_data["sms_message"])

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load user data: {str(e)}")

    def toggle_detection(self):
        if not self.is_running:
            # Check if required fields are filled
            if not self.emergency_number_var.get():
                messagebox.showerror("Error", "Emergency contact number is required before starting!")
                return

            # Start detection
            self.is_running = True
            self.start_btn.config(text="Stop Detection")
            self.detection_thread = threading.Thread(target=self.run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
        else:
            # Stop detection
            self.is_running = False
            self.start_btn.config(text="Start Drowsiness Detection")

    def send_sms_alert(self, driver_name):
        try:
            # Your Twilio credentials (provided by you)
            account_sid = '#######ef8f2bcd66b6ac7c6f3bafced2af'
            auth_token = '#########995a93b58e48c41329198caba'
            from_number = '+102222217983'  # Your Twilio number
            to_number = '+9179000000064'  # Emergency number (recipient)

            # SMS message content
            message_body = f"URGENT ALERT: {driver_name} appears to be falling asleep while driving. Call him immediately!"

            # Send SMS using Twilio
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body=message_body,
                from_=from_number,
                to=to_number
            )

            print(f"SMS alert sent successfully. SID: {message.sid}")
            return True

        except Exception as e:
            print(f"Failed to send SMS: {str(e)}")
            return False

    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = distance.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def play_sound(self, sound_file):
        try:
            # Force stop any currently playing sound
            mixer.music.stop()

            # Make sure the sound file exists
            if not os.path.exists(sound_file):
                print(f"Warning: Sound file not found: {sound_file}")
                return

            # Load and play the sound file
            print(f"Playing sound: {sound_file}")
            mixer.music.load(sound_file)
            mixer.music.play(-1)  # Loop the sound (-1 means loop indefinitely)
            self.current_sound = sound_file
        except Exception as e:
            print(f"Failed to play sound: {str(e)}")

    def stop_sound(self):
        try:
            if mixer.music.get_busy():
                mixer.music.stop()
            self.current_sound = None
            print("Sound stopped")
        except Exception as e:
            print(f"Error stopping sound: {str(e)}")

    def run_detection(self):
        # Load user settings
        with open(CONFIG_FILE, 'r') as f:
            user_data = json.load(f)

        thresh = float(user_data.get("ear_threshold", EYE_AR_THRESH))
        warning_time = int(user_data.get("warning_time", WARNING_TIME))  # Default 5 seconds
        danger_time = int(user_data.get("danger_time", DANGER_TIME))  # Default 10 seconds
        driver_name = user_data.get("name", "Driver")

        # Get sound file paths - ensure using the correct filenames
        warning_sound = "music.wav"  # Always use music.wav for first alert (5s)
        danger_sound = "warning_alarm.mp3"  # Always use warning_alarm.mp3 for second alert (10s)

        # Print sound file paths for debugging
        print(f"Warning sound (5s): {warning_sound}")
        print(f"Danger sound (10s): {danger_sound}")

        # Verify sound files exist
        if not os.path.exists(warning_sound):
            print(f"WARNING: Sound file not found: {warning_sound}")
        if not os.path.exists(danger_sound):
            print(f"WARNING: Sound file not found: {danger_sound}")

        # Calculate frame thresholds
        warning_frames = warning_time * FPS_ESTIMATE  # Frames for 5 seconds
        danger_frames = danger_time * FPS_ESTIMATE  # Frames for 10 seconds

        # Initialize face detection
        detect = dlib.get_frontal_face_detector()

        try:
            predict = dlib.shape_predictor(SHAPE_PREDICTOR)
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Failed to load shape predictor: {str(e)}\nMake sure the file exists at: {SHAPE_PREDICTOR}")
            self.is_running = False
            self.start_btn.config(text="Start Drowsiness Detection")
            return

        # Get facial landmarks indices
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        # Initialize video capture
        cap = cv2.VideoCapture(0)

        flag = 0
        sms_sent = False
        current_alert_level = 0  # 0=none, 1=warning, 2=danger

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)

            if len(subjects) > 0:  # Make sure a face is detected
                for subject in subjects:
                    shape = predict(gray, subject)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    # Draw eye contours
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # Display current EAR value
                    cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Check if eyes are closed (EAR below threshold)
                    if ear < thresh:
                        flag += 1
                        # Calculate approximate seconds based on frame count
                        seconds_closed = flag / FPS_ESTIMATE

                        # Print debug info
                        if flag % 30 == 0:  # Print every second
                            print(f"Eyes closed for {seconds_closed:.1f} seconds (frames: {flag})")

                        # Display info based on alert level
                        if flag < warning_frames:
                            # Normal state - eyes just closed but less than 5 seconds
                            cv2.putText(frame, f"Eyes Closed: {seconds_closed:.1f}s", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            # Keep alert level at 0 - no sound playing

                        elif flag >= warning_frames and flag < danger_frames:
                            # Warning level - between 5-10 seconds - PLAY MUSIC.WAV
                            cv2.putText(frame, "WARNING: Eyes Closed Too Long!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            cv2.putText(frame, f"Time: {seconds_closed:.1f}s", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                            # Switch to warning alert level if we haven't already
                            if current_alert_level != 1:
                                print(f"ALERT LEVEL 1: Playing warning sound {warning_sound}")
                                self.stop_sound()  # Stop any current sound
                                self.play_sound(warning_sound)  # Play music.wav (first alert)
                                current_alert_level = 1  # Set alert level to warning

                        elif flag >= danger_frames:
                            # Danger level - 10+ seconds - PLAY WARNING_ALARM.MP3
                            cv2.putText(frame, "****************DANGER!****************", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, f"Eyes closed for {seconds_closed:.1f}s", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # Switch to danger alert level if we haven't already
                            if current_alert_level != 2:
                                print(f"ALERT LEVEL 2: Playing danger sound {danger_sound}")
                                self.stop_sound()  # Stop any current sound
                                self.play_sound(danger_sound)  # Play warning_alarm.mp3 (second alert)
                                current_alert_level = 2  # Set alert level to danger

                            # Send SMS alert if not already sent
                            if not sms_sent and seconds_closed >= danger_time:
                                sms_success = self.send_sms_alert(driver_name)
                                sms_text = "SMS Alert Sent!" if sms_success else "Failed to send SMS!"
                                cv2.putText(frame, sms_text, (10, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                sms_sent = True
                    else:
                        # Eyes are open, reset everything
                        if flag > 0:
                            print("Eyes opened - resetting detection")

                        flag = 0
                        sms_sent = False

                        # If we were playing a sound, stop it
                        if current_alert_level > 0:
                            self.stop_sound()

                        current_alert_level = 0

            # Display additional program information
            cv2.putText(frame, f"Warning: {warning_time}s | Danger: {danger_time}s", (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow("Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not self.is_running:
                break

        cv2.destroyAllWindows()
        cap.release()
        self.stop_sound()
        self.is_running = False
        self.start_btn.config(text="Start Drowsiness Detection")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectionApp(root)
    root.mainloop()