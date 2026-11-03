from fastapi import FastAPI
from fastapi.responses import FileResponse
from datetime import datetime
from pathlib import Path

app = FastAPI(title="WildGuard API")

detection_events = []
current_location = {
    "latitude": None,
    "longitude": None
}

WEBSITE_FILE = Path(__file__).parent / "wildguard_monitor.html"


@app.get("/")
def home():
    return FileResponse(WEBSITE_FILE)


@app.post("/location")
def update_location(data: dict):
    current_location["latitude"] = data.get("latitude")
    current_location["longitude"] = data.get("longitude")

    return {
        "message": "Location updated",
        "location": current_location
    }


@app.post("/detections")
def receive_detection(data: dict):
    event = {
        "id": len(detection_events) + 1,
        "species": data.get("species"),
        "confidence": data.get("confidence"),
        "latitude": data.get("latitude") or current_location["latitude"],
        "longitude": data.get("longitude") or current_location["longitude"],
        "timestamp": data.get("timestamp") or datetime.now().isoformat(),
        "video": data.get("video")
    }

    detection_events.append(event)

    return {
        "message": "Detection received successfully",
        "event": event
    }


@app.get("/detections")
def get_detections():
    return detection_events