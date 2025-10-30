import threading, time, math
from collections import deque
from io import BytesIO
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from flask import Flask, Response, render_template, send_file, jsonify, request

# ---------------- CONFIG ----------------
ENABLE_SYSTEM_VOLUME = True
audio_endpoint = None
_pycaw_available = False

if ENABLE_SYSTEM_VOLUME:
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        audio_endpoint = cast(interface, POINTER(IAudioEndpointVolume))
        _pycaw_available = True
        print("[INFO] Pycaw initialized — system volume control ENABLED.")
    except Exception as e:
        print("[WARN] Pycaw unavailable:", e)

# ---------------- GLOBALS ----------------
app = Flask(__name__)
lock = threading.Lock()
cap = None
frame = None
distance_val = 0.0
distances = deque(maxlen=600)
gesture = "NONE"
volume = 0
smooth_vol = 0.0

DEFAULT_MIN, DEFAULT_MAX = 20.0, 200.0
calib_min, calib_max = DEFAULT_MIN, DEFAULT_MAX
calib_applied = False

last_plot_time = 0.0
plot_cache_bytes = None
running = True

mp_hands = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)


# ---------------- HELPERS ----------------
def set_system_volume(percent):
    if _pycaw_available and audio_endpoint:
        try:
            scalar = float(np.clip(percent / 100.0, 0.0, 1.0))
            audio_endpoint.SetMasterVolumeLevelScalar(scalar, None)
        except Exception as e:
            print("[WARN] System volume:", e)


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------- CAMERA THREAD ----------------
def camera_loop():
    global cap, frame, distance_val, gesture, volume, smooth_vol
    global calib_min, calib_max, calib_applied

    print("[INFO] Opening camera…")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return
    print("[INFO] Camera opened successfully.")

    last_time = time.time()
    frame_interval = 1 / 20.0  # target 20 FPS (lighter load)

    while running:
        ret, img = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        d = 0.0
        gest = "NONE"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            draw_utils.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            lm4, lm8 = hand.landmark[4], hand.landmark[8]
            d = euclid((lm4.x * w, lm4.y * h), (lm8.x * w, lm8.y * h))
            distance_val = d
            distances.append(d)

            wrist = hand.landmark[0]
            tips = [hand.landmark[i] for i in (8, 12, 16, 20)]
            avg = np.mean([euclid((wrist.x * w, wrist.y * h), (t.x * w, t.y * h)) for t in tips])
            gest = "PINCH" if d < 40 else "CLOSED" if avg < 120 else "OPEN"

        min_d = calib_min if calib_applied else DEFAULT_MIN
        max_d = calib_max if calib_applied else DEFAULT_MAX
        vol = np.interp(d, [min_d, max_d], [0, 100])
        smooth_vol = 0.85 * smooth_vol + 0.15 * vol
        volume = int(np.clip(smooth_vol, 0, 100))
        if _pycaw_available:
            set_system_volume(volume)
        gesture = gest

        with lock:
            frame = img

        # Maintain stable FPS
        elapsed = time.time() - last_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_time = time.time()

    cap.release()
    print("[INFO] Camera thread stopped.")


# ---------------- FLASK ROUTES ----------------

@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/detect")
def detect_page():
    return render_template("detect.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                if frame is None:
                    time.sleep(0.02)
                    continue
                ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1 / 20)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify({
        "distance": distance_val,
        "volume": volume,
        "gesture": gesture,
        "calib_min": calib_min,
        "calib_max": calib_max,
        "calib_applied": calib_applied,
    })



@app.route("/calib/apply_manual", methods=["POST"])
def calib_apply_manual():
    global calib_min, calib_max, calib_applied, last_plot_time, plot_cache_bytes
    data = request.get_json() or {}
    calib_min = float(data.get("min", DEFAULT_MIN))
    calib_max = float(data.get("max", DEFAULT_MAX))
    if calib_max <= calib_min:
        calib_max = calib_min + 1.0
    calib_applied = True
    plot_cache_bytes = None
    last_plot_time = 0.0
    return jsonify({"calib_min": calib_min, "calib_max": calib_max})

@app.route("/plot.png")
def plot_png():
    global distances

    import matplotlib.pyplot as plt
    from io import BytesIO
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=100)  # Keep same graph box size

    # Prepare data
    y = list(distances)
    if len(y) == 0:
        y = [0]
    x = list(range(len(y)))

    # Plot line
    ax.plot(x, y, color="#58a6ff", linewidth=2, label="Distance")

    # Add min/max markers
    ymin, ymax = min(y), max(y)
    ax.scatter(y.index(ymin), ymin, color="#f87171", s=35, label=f"Min: {ymin:.1f}")
    ax.scatter(y.index(ymax), ymax, color="#34d399", s=35, label=f"Max: {ymax:.1f}")

    # Set background
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Add X/Y labels
    ax.set_xlabel("Frame Count", color="#eaeaea", fontsize=8, labelpad=4)
    ax.set_ylabel("Distance (px)", color="#eaeaea", fontsize=8, labelpad=4)
    ax.set_title("Live Gesture Distance", color="#58a6ff", fontsize=9, pad=6)

    # Show grid + frame
    ax.grid(True, linestyle="--", linewidth=0.6, color="#58a6ff", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("#58a6ff")
        spine.set_linewidth(0.8)

    # Ticks styling
    ax.tick_params(axis='x', colors='#eaeaea', labelsize=7)
    ax.tick_params(axis='y', colors='#eaeaea', labelsize=7)

    # Axis range with 10% margin
    y_range = ymax - ymin if ymax != ymin else 1
    ax.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)
    ax.margins(x=0.05)

    # Legend
    ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#58a6ff", labelcolor="#eaeaea", loc="upper right")

    fig.tight_layout(pad=1.0)
    plt.close(fig)

    output = BytesIO()
    fig.savefig(output, format="png", facecolor="#0d1117", transparent=False)
    output.seek(0)
    return send_file(output, mimetype="image/png")



@app.route("/camera/start", methods=["POST"])
def start_camera():
    global cap, running
    if cap is None or not cap.isOpened():
        print("[INFO] Starting camera...")
        running = True
        threading.Thread(target=camera_loop, daemon=True).start()
        time.sleep(1)  # allow initialization
    else:
        print("[INFO] Camera already running.")
    return jsonify({"status": "started"})


@app.route("/camera/pause", methods=["POST"])
def pause_camera():
    """Pause or resume hand detection."""
    global running
    running = not running
    if running:
        print("[INFO] Resuming detection...")
        threading.Thread(target=camera_loop, daemon=True).start()
    else:
        print("[INFO] Detection paused.")
    return jsonify({"paused": not running})


@app.route("/camera/stop", methods=["POST"])
def stop_camera():
    """Stop the webcam and release it properly."""
    global cap, running, frame
    print("[INFO] Stopping camera...")
    running = False
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # show black frame after stop
    cv2.putText(frame, "Camera stopped. Click 'Start' to resume.",
                (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    print("[INFO] Camera stopped successfully.")
    return jsonify({"status": "stopped"})



# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("[INFO] Flask server running at http://127.0.0.1:5000")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
