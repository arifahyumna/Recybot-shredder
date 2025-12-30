import os, sys, random, datetime as dt
import cv2
import torch
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import RPi.GPIO as GPIO
import threading
import time
from pathlib import Path
import paho.mqtt.client as mqtt
import json
from utils.augmentations import letterbox
import numpy as np
import tensorflow as tf

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# ---------------- variabel global ----------------
suhu_update = None
xs = []
ys = []
SSR_PIN = 17

running = True
worker_running = True

CLASS_NAMES = ["plastik", "non plastik", "hand"]

#------------setup path and device--------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(ROOT))
tflite_model_path = str(ROOT / 'best (2)-fp16.tflite')

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def camera_loop():
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)
            
#------------------setup GPIO------------------
GPIO_AVAILABLE = True
try:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SSR_PIN, GPIO.OUT)
    GPIO.output(SSR_PIN, GPIO.LOW)
    GPIO_AVAILABLE = True
except RuntimeError as e:
    print("RPi.GPIO runtime error:", e)
    GPIO_AVAILABLE = False
except Exception as e:
    print("Error in setting up GPIO:", e)
    GPIO_AVAILABLE = False

#--------------setup tkinter gui---------------
root = tk.Tk()
root.title("Smart Monitoring Machine")

frame_1 = tk.Frame(root, relief="groove", borderwidth=2)
frame_2 = tk.Frame(root, relief="groove", borderwidth=2)

frame_1.pack(side="left", fill=tk.BOTH, expand=True, padx=10, pady=10)
frame_2.pack(side="right", fill=tk.BOTH, expand=True, padx=10, pady=10)

# Frame 1 - PENCACAH
label_p1 = tk.Label(frame_1, text="PENCACAH", font=("Segoe UI", 12, "bold"))
label_p1.pack(pady=5)

frame_daya1 = tk.Frame(frame_1)
label_daya1 = tk.Label(frame_daya1, text="Daya")
label_unitd1 = tk.Label(frame_daya1, text="0 Watt", relief="ridge", width=10)
label_daya1.pack(side="left", padx=5)
label_unitd1.pack(side="left", padx=5)
frame_daya1.pack(pady=5)

frame_vision = tk.Frame(frame_1, bg="gray", width=400, height=300)
label_v = tk.Label(frame_vision, text="Kamera")
frame_vision.pack(pady=5)
frame_vision.pack_propagate(False)
label_v.pack(expand=True)

frame_klas = tk.Frame(frame_1)
label_klas = tk.Label(frame_klas, text="Klasifikasi:")
label_result = tk.Label(frame_klas, text="--", relief="ridge", width=10)
frame_klas.pack(pady=10)
label_klas.pack(side="left", padx=5)
label_result.pack(side="left", padx=5)

frame_process = tk.Frame(frame_1)
label_process = tk.Label(frame_process, text="--", relief="ridge", width=40)
frame_process.pack(pady=15)
label_process.pack(side="left")

# Frame 2 - PENCETAK
label_p2 = tk.Label(frame_2, text="PENCETAK", font=("Segoe UI", 12, "bold"))
label_p2.pack(pady=5)

frame_daya2 = tk.Frame(frame_2)
label_daya2 = tk.Label(frame_daya2, text="Daya:")
label_unitd2 = tk.Label(frame_daya2, text="0 Watt", relief="ridge", width=10)
frame_daya2.pack(pady=5)
label_daya2.pack(side="left", padx=5)
label_unitd2.pack(side="left", padx=5)

frame_grafik = tk.Frame(frame_2, bg="gray", width=400, height=300)
#label_graf = tk.Label(frame_grafik, text="Grafik Suhu")
frame_grafik.pack(pady=5)
frame_grafik.pack_propagate(False)
#label_graf.pack(expand=True)

frame_suhu = tk.Frame(frame_2)
label_suhu = tk.Label(frame_suhu, text="Suhu:")
label_units = tk.Label(frame_suhu, text="0 ℃", relief="ridge", width=10)
frame_suhu.pack(pady=10)
label_suhu.pack(side="left", padx=5)
label_units.pack(side="left", padx=5)

frame_alert = tk.Frame(frame_2)
label_alert = tk.Label(frame_alert, text="--", relief="ridge", width=40)
frame_alert.pack(pady=15)
label_alert.pack(side="left")

# ---------------- Grafik setup ----------------
fig, ax = plt.subplots(figsize=(4, 3))
canvas = FigureCanvasTkAgg(fig, master=frame_grafik)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=5)

ax.set_title("Monitoring Suhu Pencetak", fontsize=10, pad=10)
ax.set_xlabel("Waktu", fontsize=5)
ax.set_ylabel("Suhu (°C)", fontsize=5)
ax.grid(True, linestyle="--", alpha=0.6)

def update_plot(value):
    # 'value' bisa None jika belum ada update suhu
    xs.append(dt.datetime.now().strftime('%H:%M:%S'))
    ys.append(value if value is not None else 0)

    # hanya simpan 10 data terakhir
    xs[:] = xs[-10:]
    ys[:] = ys[-10:]

    # draw
    ax.clear()
    ax.plot(xs, ys, marker='o')
    ax.set_title("Monitoring Suhu Press Heater")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Suhu (°C)")
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    canvas.draw()

def graf_update():
    # gunakan global suhu_update
    global suhu_update
    update_plot(suhu_update)
    # panggil lagi tiap 5000 ms
    root.after(5000, graf_update)

# ---------------- MQTT callbacks ----------------
def on_connect(client, userdata, flags, rc):
	global flag_connected
	flag_connected = 1
	client_subscription(client)
	print("Connected to MQTT server")
    
def on_disconnect(client, userdata, rc):
	global flag_connected
	flag_connected = 0
	print("Disconncted From MQTT server")
    
def client_subscription(client):
	client.subscribe("esp32/#")

def on_message(client, userdata, msg):
    global suhu_update
    topic = msg.topic
    payload = msg.payload.decode()
    
    print(f"[DEBUG] Topic: {topic}, Payload: {payload}")

    if topic == "esp32/suhu":
        try:
            data = json.loads(payload)
            suhu_val = data["temp_c"]
            suhu_update = suhu_val
            label_units.config(text=f"{suhu_val: .2f} ℃")
        except Exception:
            label_units.config(text=f"Error: {payload}")
            print("error suhu: ", e)

    elif topic == "esp32/daya":
        try:
            data = json.loads(payload)
            daya_val = data.get("daya_watt", 0.0)
            label_unitd2.config(text=f"{daya_val} Watt")
        except Exception:
            label_unitd2.config(text=f"Error: {payload}")
            print("error daya: ", e)

    elif topic == "esp32/alert":
        try:
            data = json.loads(payload)
            alert_val = data.get("alert", payload)
            label_alert.config(text=f"{alert_val}")
        except Exception:
            label_alert.config(text=f"{payload}")
            print("error alert: ", e)
        

#--------------------Frame YOLO--------------------------
import queue

frame_queue = queue.Queue(maxsize=1)

annotated_frame = None
class_text = "--"
process_text = "--"

worker_running = True
worker_lock = threading.Lock()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak terdeteksi")
else:
    print("Kamera terdeteksi")
    
def worker():
    global annotated_frame, class_text, process_text, worker_running

    while worker_running:
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # ===== PREPROCESS =====
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # ===== INFERENCE =====
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            best_conf = 0
            best_class = None

            for det in output:
                if det[4] < 0.3:
                    continue

                class_scores = det[5:]
                cid = int(np.argmax(class_scores))
                conf = det[4] * class_scores[cid]

                if conf > best_conf:
                    best_conf = conf
                    best_class = cid

            # ===== LOGIKA SHREDDER =====
            if best_class is not None:
                label = f"{CLASS_NAMES[best_class]} ({best_conf:.2f})"

                if CLASS_NAMES[best_class] == "plastik":
                    status = "SHREDDER ON"
                    if GPIO_AVAILABLE:
                        GPIO.output(SSR_PIN, GPIO.HIGH)
                else:
                    status = "SHREDDER OFF"
                    if GPIO_AVAILABLE:
                        GPIO.output(SSR_PIN, GPIO.LOW)
            else:
                label = "--"
                status = "SHREDDER OFF"
                if GPIO_AVAILABLE:
                    GPIO.output(SSR_PIN, GPIO.LOW)

            # ===== ANNOTASI =====
            annotated = frame.copy()
            cv2.putText(
                annotated, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )

            # ===== SHARE KE GUI =====
            with worker_lock:
                annotated_frame = annotated
                class_text = label
                process_text = status

        except Exception as e:
            print("Worker exception:", e)


def update_camera():
    with worker_lock:
        af = annotated_frame
        ct = class_text
        pt = process_text

    if af is not None:
        af_rgb = cv2.cvtColor(af, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(
            Image.fromarray(af_rgb).resize((400, 300))
        )
        label_v.configure(image=imgtk)
        label_v.imgtk = imgtk

    label_result.config(text=ct)
    label_process.config(
        text=pt,
        fg="green" if pt == "SHREDDER ON" else "red"
    )

    label_v.after(30, update_camera)

    
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()
	
#-------------------jalankan GUI------------------------------
graf_update()
update_camera()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), GPIO.cleanup(), root.destroy()))
root.mainloop()
