import cv2
import pyautogui
import webbrowser
import subprocess
import threading
import time
import json
import random
import os

from imutils.video import FPS
from flask import Flask, Response, render_template

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

import pyaudio
import wave
import pyttsx3
import speech_recognition as sr

import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt
import serial
time.sleep(10)

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)   # using BCM numbering

# NOTE: if your hardware is still on 12 / 21, change back here:
readbutton = 21
light = 20

GPIO.setup(readbutton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(light, GPIO.OUT)

# ---------------- RELAY SETUP ----------------
relay1 = 26   # Relay 1
relay2 = 19   # Relay 2

GPIO.setup(relay1, GPIO.OUT)
GPIO.setup(relay2, GPIO.OUT)

# Active LOW â†’ OFF initially
GPIO.output(relay1, GPIO.HIGH)
GPIO.output(relay2, GPIO.HIGH)

# ---------------- MQTT SETUP (ROBOSMS) ----------------
MQTT_SERVER = "konnect.robosap.co.in"
MQTT_PORT = 1883
MQTT_USER = "robosaptwo"
MQTT_PASSWORD = "337adxl2023"
MQTT_TOPIC_SMS = "robosms"

mqtt_client = mqtt.Client()

def setup_mqtt():
    try:
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        mqtt_client.connect(MQTT_SERVER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("[MQTT] Connected to broker")
    except Exception as e:
        print("[MQTT] Connection failed:", e)

def send_help_alert(source="gesture"):
 
    try:
        payload = json.dumps({
            "number": "+919902071516",
            "msg": "help required by disabled person please attend."
        })
        mqtt_client.publish(MQTT_TOPIC_SMS, payload, qos=1)
        print(f"[MQTT] HELP SMS sent from {source}: {payload}")

        # Also speak help for confirmation
        engine.say("Help request sent")
        engine.runAndWait()

    except Exception as e:
        print("[MQTT] Failed to send HELP SMS:", e)


# ---------------- TTS ENGINE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ---------------- LSTM MODEL ----------------
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# ----- Model / Encoder / Scaler -----
input_size = 84
hidden_size = 32

label_encoder_classes = np.load('/home/pi/Desktop/signlangugae/label_encoder_classes.npy', allow_pickle=True)
output_size = len(label_encoder_classes)

model = GestureLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('/home/pi/Desktop/signlangugae/best_gesture_modelrpi.pth', map_location=torch.device('cpu')))
model.eval()

label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

scaler = torch.load('/home/pi/Desktop/signlangugae/scaler.pth', map_location=torch.device('cpu'))

# ---------------- Mediapipe Hands ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ---------------- Prediction Helper ----------------
def predict_gesture(model, data, scaler, label_encoder):
    data = scaler.transform(data)
    data = torch.tensor(data, dtype=torch.float32).reshape(1, 1, data.shape[1])

    with torch.no_grad():
        outputs = model(data)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    gesture = label_encoder.inverse_transform([preds.item()])
    return gesture[0], confidence.item()

# ---------------- Audio Recording ----------------
def recordaudio():
    form_1 = pyaudio.paInt16
    chans = 1
    samp_rate = 16000
    chunk = 4096
    record_secs = 5
    dev_index = 1
    wav_output_filename = 'test1.wav'

    audio = pyaudio.PyAudio()
    stream = audio.open(format=form_1, rate=samp_rate, channels=chans,
                        input_device_index=dev_index, input=True,
                        frames_per_buffer=chunk)

    print("recording")
    GPIO.output(light, GPIO.HIGH)

    frames = []
    for ii in range(0, int((samp_rate / chunk) * record_secs)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)

    print("finished recording")
    GPIO.output(light, GPIO.LOW)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wavefile = wave.open(wav_output_filename, 'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

# ---------------- Speech Recognition ----------------
def recognizea(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)

    response = {"success": True, "error": None, "transcription": None}

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

# ---------------- Globals for Gesture Logic ----------------
prediction_buffer = []
sentence = []
BUFFER_SIZE = 10
COOLDOWN_PERIOD = 5
last_gesture_time = time.time()

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(3.0)

opened = False
recognizer = sr.Recognizer()

engine.say("Sign  Bridge-Advanced sign Translator")
engine.runAndWait()

# Get IP for Flask host/open
lss1 = subprocess.getoutput('hostname -I')
lss1 = lss1.strip().split()
lss = lss1[0]
print(lss)
time.sleep(2)

outputFrame = None
frame = None
lock = threading.Lock()

# --------------- Flask App ---------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frameCount):
    global outputFrame, lock, opened, frame
    global prediction_buffer, sentence, BUFFER_SIZE, COOLDOWN_PERIOD, last_gesture_time

    total = 0

    while True:
        bread = GPIO.input(readbutton)
        # print("button:", bread)

        if bread == 0:
            # SIGN MODE
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (600, 600))
            x, y, c = frame.shape

            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks:
                left_hand_lmks = [0] * (input_size // 2)
                right_hand_lmks = [0] * (input_size // 2)

                for i, handslms in enumerate(result.multi_hand_landmarks):
                    if i == 0:
                        for lm_idx, lm in enumerate(handslms.landmark):
                            left_hand_lmks[lm_idx * 2] = int(lm.x * x)
                            left_hand_lmks[lm_idx * 2 + 1] = int(lm.y * y)
                    else:
                        for lm_idx, lm in enumerate(handslms.landmark):
                            right_hand_lmks[lm_idx * 2] = int(lm.x * x)
                            right_hand_lmks[lm_idx * 2 + 1] = int(lm.y * y)

                combined_lmks = left_hand_lmks + right_hand_lmks

                if len(combined_lmks) == input_size:
                    gesture, confidence = predict_gesture(
                        model, np.array(combined_lmks).reshape(1, -1),
                        scaler, label_encoder)

                    prediction_buffer.append(gesture)
                    if len(prediction_buffer) > BUFFER_SIZE:
                        prediction_buffer.pop(0)

                    if len(prediction_buffer) == BUFFER_SIZE and all(
                        pred == prediction_buffer[0] for pred in prediction_buffer
                    ):
                        current_time = time.time()
                        if current_time - last_gesture_time > COOLDOWN_PERIOD:

                            # We use display_label to show "help" instead of "6"
                            display_label = gesture

                            if gesture == 'speak':
                                final_sentence = ''.join(sentence)
                                cv2.putText(frame, f'Sentence: {final_sentence}', (10, 90),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                print(final_sentence)
                                engine.say(final_sentence)
                                engine.runAndWait()

                            elif gesture == 'wipe':
                                sentence = []

                            elif gesture == 'clear':
                                if sentence:
                                    sentence.pop()

                            elif gesture == 'space':
                                sentence.append(' ')

                            # -------- RELAY CONTROL ----------
                            elif gesture == '3':
                                GPIO.output(relay1, GPIO.LOW)
                                print("Relay 1 ON")

                            elif gesture == '4':
                                GPIO.output(relay1, GPIO.HIGH)
                                print("Relay 1 OFF")

                            elif gesture == '5':
                                GPIO.output(relay2, GPIO.LOW)
                                print("Relay 2 ON")

                            elif gesture == '6':
                                # Treat "6" as HELP gesture
                                display_label = 'help'
                                print("HELP gesture detected (6)")
                                # Optional: also turn relay2 OFF if you still want that
                                GPIO.output(relay2, GPIO.HIGH)
                                print("Relay 2 OFF (on HELP)")
                                # Send MQTT SMS via robosms topic
                                send_help_alert(source="gesture_model")

                            # ---------------------------------
                            else:
                                sentence.append(gesture)

                            last_gesture_time = current_time

                        # Use display_label for showing text (maps 6 â†’ help)
                        if gesture == '6':
                            display_label = 'help'
                        else:
                            display_label = gesture

                        display_text = f'Gesture: {display_label} ({confidence:.2f})'
                    else:
                        display_text = "Invalid"
                        display_label = "Invalid"

                    cv2.putText(frame, display_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, str(display_label), (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                for handslms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

        else:
            # VOICE MODE  (old behaviour retained)
            print("outside (voice mode)")
            frame = cv2.imread("/home/pi/Desktop/signlangugae/nothing.jpg")
            engine.say("Please Say the Sign to Convey")
            engine.runAndWait()
            try:
                recordaudio()
                ffile = sr.AudioFile('test1.wav')
                guess = recognizea(recognizer, ffile)

                if guess["transcription"]:
                    engine.say(guess["transcription"])
                    engine.runAndWait()
                    print("You said:", guess["transcription"])

                    txt = guess["transcription"].lower().strip()

                    if txt == "good":
                        frame = cv2.imread('/home/pi/Desktop/signlangugae/good.jpg')
                    elif txt in ["hai", "hi"]:
                        frame = cv2.imread('/home/pi/Desktop/signlangugae/hii.jpg')
                    elif txt == "water":
                        frame = cv2.imread('/home/pi/Desktop/signlangugae/water.jpg')
                    elif txt == "victory":
                        frame = cv2.imread('/home/pi/Desktop/signlangugae/victory.jpg')
                    elif txt == "help":
                        frame = cv2.imread('/home/pi/Desktop/signlangugae/help.jpg')
                        # Also send HELP alert when help spoken
                        send_help_alert(source="voice_mode")

                    time.sleep(3)
            except Exception as e:
                print("Recording error:", e)

        # ------------- OPEN BROWSER (KIOSK FULLSCREEN) ---------------
        if not opened:
            link = "http://" + str(lss) + ":8000/"
            webbrowser.open(link)
            time.sleep(3)
            pyautogui.press('f11')
            print("opening browser")
            opened = True

        total += 1
        with lock:
            if frame is not None:
                outputFrame = frame.copy()

# ---------------- ARDUINO SERIAL READER ----------------
def read_arduino_serial():
    """
    Read JSON lines like {"gesture":"C"} from /dev/ttyUSB0 (Arduino Nano).
    Speak out the gesture and send HELP SMS when help gesture is received.
    """
    try:
        ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
        time.sleep(2)
        print("[SERIAL] Connected to /dev/ttyUSB0")
    except Exception as e:
        print("[SERIAL] Error opening /dev/ttyUSB0:", e)
        return

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print("[SERIAL] Raw:", line)

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # ignore non-JSON lines
                continue

            if "gesture" in data:
                g = str(data["gesture"])
                print(f"[SERIAL] Gesture from Arduino: {g}")

                # Speak out the gesture (you can map letters to words if needed)
                try:
                    engine.say(g)
                    engine.runAndWait()
                except Exception as e:
                    print("[TTS] error while speaking Arduino gesture:", e)

                # Help condition from Arduino side too
                if g.lower() == "help" or g == "6":
                    print("[SERIAL] HELP gesture from Arduino")
                    send_help_alert(source="arduino_serial")

        except Exception as e:
            print("[SERIAL] Read error:", e)
            time.sleep(1)

# ---------------- FLASK VIDEO STREAM ----------------
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_sentence")
def get_sentence():
    global sentence
    return {"sentence": "".join(sentence)}

# --------------- MAIN ---------------
if __name__ == '__main__':
    try:
        setup_mqtt()

        t_cam = threading.Thread(target=detect_motion, args=(32,))
        t_cam.daemon = True
        t_cam.start()

        t_serial = threading.Thread(target=read_arduino_serial)
        t_serial.daemon = True
        t_serial.start()

        app.static_folder = 'static'
        app.run(host=lss, port=8000, debug=True,
                threaded=True, use_reloader=False)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        try:
            mqtt_client.loop_stop()
        except:
            pass

