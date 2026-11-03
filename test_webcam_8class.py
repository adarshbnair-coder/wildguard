"""
WildGuard 8-class webcam detector.

Camera location is obtained on THIS laptop through a small local browser page.
Open the page that appears and allow location permission. The webcam process
reads the coordinates from the loopback-only local service and uploads each
new animal detection to the authenticated Render API.

Required environment variable:
    WILDGUARD_CAMERA_API_KEY
"""

from ultralytics import YOLO
import cv2
import requests
import os
import time
import json
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_PATH = r"runs\detect\runs\detect\wildlife_8class_30epochs\weights\best.pt"
API_URL = "https://wildguard-2ddm.onrender.com/detections"

CAMERA_API_KEY = os.getenv("WILDGUARD_CAMERA_API_KEY", "").strip()
CAMERA_ID = "laptop-webcam-01"

WEBCAM_INDEX = 0
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25

LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8765
LOCAL_LOCATION_URL = f"http://{LOCAL_HOST}:{LOCAL_PORT}/location"

# Reject a location reading if the reported accuracy radius is worse than this.
MAX_LOCATION_ACCURACY_METERS = 1000

# Retry an upload/location read after this many seconds if it has not succeeded.
RETRY_INTERVAL_SECONDS = 5


# ---------------------------------------------------------------------------
# LOCAL LAPTOP LOCATION SERVICE
# ---------------------------------------------------------------------------

location_lock = threading.Lock()
camera_location = {
    "available": False,
    "latitude": None,
    "longitude": None,
    "accuracy": None,
    "timestamp": None,
}


LOCATION_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WildGuard Camera Location</title>
<style>
body{margin:0;padding:28px;background:#0b1220;color:#fff;font:16px Arial,sans-serif}
main{max-width:680px;margin:30px auto;background:#142238;padding:26px;border-radius:12px}
h1{color:#39d6a0}#status{padding:14px;background:#0b1a2c;border-radius:8px;line-height:1.6}
.small{color:#aabbd0;line-height:1.6}
</style>
</head>
<body>
<main>
<h1>WILDGUARD — Camera Location</h1>
<p>This page supplies location to the WildGuard webcam program running on
this same laptop. Keep this page open while animal detection is running.</p>
<div id="status">Requesting laptop location permission...</div>
<p class="small">Select Allow when your browser asks for location permission.
The reading is an estimate of the laptop's location, not the exact position
of an animal. A reading with accuracy worse than 1 km will not be used.</p>
</main>
<script>
const statusBox = document.getElementById("status");

function sendLocation(position) {
    const c = position.coords;
    const accuracy = Number(c.accuracy);
    const payload = {
        latitude: Number(c.latitude),
        longitude: Number(c.longitude),
        accuracy: accuracy,
        timestamp: new Date(position.timestamp).toISOString()
    };

    fetch("/location", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    }).then(response => {
        if (!response.ok) throw new Error("Local location service rejected the reading");
        if (accuracy > 1000) {
            statusBox.textContent =
                "Location received, but accuracy is too low (" +
                Math.round(accuracy) + " metres). Waiting for a better reading.";
        } else {
            statusBox.textContent =
                "Laptop location ready. Latitude: " + payload.latitude.toFixed(6) +
                " | Longitude: " + payload.longitude.toFixed(6) +
                " | Accuracy: " + Math.round(accuracy) + " metres. " +
                "Keep this page open.";
        }
    }).catch(error => {
        statusBox.textContent = "Could not send location to the local webcam program: " + error;
    });
}

function locationError(error) {
    const messages = {
        1: "Location permission was denied. Allow location access for this local page in browser settings.",
        2: "The laptop location is currently unavailable. Check location services and try again.",
        3: "Location request timed out. Waiting for another reading."
    };
    statusBox.textContent = messages[error.code] || "Unable to obtain laptop location.";
}

if (!navigator.geolocation) {
    statusBox.textContent = "This browser does not support location services.";
} else {
    navigator.geolocation.watchPosition(
        sendLocation,
        locationError,
        {enableHighAccuracy: true, maximumAge: 5000, timeout: 20000}
    );
}
</script>
</body>
</html>
"""


class LocationHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = LOCATION_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/location":
            with location_lock:
                current = dict(camera_location)
            self._send_json(200, current)
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        # Only accept browser updates originating from this local helper page.
        origin = self.headers.get("Origin")
        allowed_origins = {
            f"http://{LOCAL_HOST}:{LOCAL_PORT}",
            f"http://localhost:{LOCAL_PORT}",
        }
        if origin not in allowed_origins:
            self._send_json(403, {"error": "Invalid local page origin"})
            return

        if self.path != "/location":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            latitude = float(payload["latitude"])
            longitude = float(payload["longitude"])
            accuracy = float(payload["accuracy"])
            timestamp = str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat())

            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude is out of range")
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude is out of range")
            if accuracy < 0:
                raise ValueError("Accuracy cannot be negative")

            with location_lock:
                camera_location.update({
                    "available": True,
                    "latitude": latitude,
                    "longitude": longitude,
                    "accuracy": accuracy,
                    "timestamp": timestamp,
                })

            self._send_json(200, {"message": "Laptop location updated"})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})

    def log_message(self, format_string, *args):
        # Keep the webcam console output readable.
        return


def start_location_service():
    server = ThreadingHTTPServer((LOCAL_HOST, LOCAL_PORT), LocationHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    page_url = f"http://{LOCAL_HOST}:{LOCAL_PORT}/"
    print("\nWildGuard laptop location page:", page_url)
    print("Allow location permission in the browser window that opens.")
    webbrowser.open(page_url)
    return server


def get_camera_location():
    """Return a sufficiently accurate laptop location, otherwise None."""
    try:
        response = requests.get(LOCAL_LOCATION_URL, timeout=2)
        response.raise_for_status()
        data = response.json()

        if not data.get("available"):
            return None

        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        accuracy = float(data["accuracy"])

        if accuracy > MAX_LOCATION_ACCURACY_METERS:
            print(
                f"Camera location accuracy is too low ({accuracy:.0f} m); "
                "waiting for a better reading."
            )
            return None

        return {
            "latitude": latitude,
            "longitude": longitude,
            "location_accuracy_m": round(accuracy, 1),
            "location_timestamp": data.get("timestamp"),
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        print("Laptop location is not ready:", error)
        return None


# ---------------------------------------------------------------------------
# MAIN YOLO WEBCAM LOOP
# ---------------------------------------------------------------------------

def main():
    if not CAMERA_API_KEY:
        raise SystemExit(
            "\nERROR: WILDGUARD_CAMERA_API_KEY is not set.\n"
            "Set it in your Windows environment, then reopen the terminal."
        )

    print("Loading WildGuard 8-class YOLO model...")
    model = YOLO(MODEL_PATH)

    location_server = start_location_service()
    time.sleep(1)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    print("Webcam opened:", cap.isOpened())

    if not cap.isOpened():
        location_server.shutdown()
        raise SystemExit("ERROR: Could not open webcam. Check camera index and permissions.")

    # A species is uploaded once while it remains continuously visible.
    # Failed uploads/location reads are retried every few seconds.
    uploaded_animals = set()
    last_attempt_time = {}

    headers = {"X-Camera-API-Key": CAMERA_API_KEY}

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            result = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )[0]

            display = result.plot()
            current_animals = {}

            for box in result.boxes:
                class_id = int(box.cls[0])
                species = str(model.names[class_id])

                # Ignore person detections.
                if species.lower() == "person":
                    continue

                confidence = float(box.conf[0])

                # Keep highest confidence for each species in this frame.
                if (
                    species not in current_animals
                    or confidence > current_animals[species]
                ):
                    current_animals[species] = confidence

            # Reset upload state after a species disappears.
            for species in list(uploaded_animals):
                if species not in current_animals:
                    uploaded_animals.discard(species)
                    last_attempt_time.pop(species, None)

            for species, confidence in current_animals.items():
                if species in uploaded_animals:
                    continue

                now = time.time()
                if now - last_attempt_time.get(species, 0) < RETRY_INTERVAL_SECONDS:
                    continue

                last_attempt_time[species] = now

                # Do not send a detection without camera coordinates.
                location = get_camera_location()
                if location is None:
                    print(
                        f"{species} detected, but no sufficiently accurate "
                        "laptop coordinates are available. Will retry."
                    )
                    continue

                event = {
                    "species": species,
                    "confidence": round(confidence, 3),
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "location_accuracy_m": location["location_accuracy_m"],
                    "location_timestamp": location["location_timestamp"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "video": None,
                    "camera_id": CAMERA_ID,
                }

                try:
                    response = requests.post(
                        API_URL,
                        json=event,
                        headers=headers,
                        timeout=15
                    )

                    if response.status_code == 200:
                        uploaded_animals.add(species)
                        print(
                            "Uploaded to WildGuard:",
                            species,
                            f"confidence={confidence:.3f}",
                            f"latitude={location['latitude']:.6f}",
                            f"longitude={location['longitude']:.6f}",
                        )
                    elif response.status_code in (401, 403):
                        print(
                            "AUTHENTICATION FAILED. Check that the camera API key "
                            "on this laptop matches the Render environment variable."
                        )
                    else:
                        print("Upload failed:", response.status_code, response.text)

                except requests.RequestException as error:
                    print("API connection error:", error)

            cv2.imshow("WildGuard Animal Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        location_server.shutdown()
        location_server.server_close()
        print("WildGuard webcam stopped.")


if __name__ == "__main__":
    main()
