"""
WildGuard FastAPI backend.

POST /detections is reserved for the authenticated laptop camera client.
GET /detections is public and read-only for the dashboard.
The old public POST /location endpoint is intentionally not available.
"""

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="WildGuard API")

# Keep the current in-memory storage behaviour of the existing prototype.
# Note: Render restarts/redeployments can clear in-memory detections.
detection_events = []

WEBSITE_FILE = Path(__file__).parent / "wildguard_monitor.html"


class DetectionIn(BaseModel):
    species: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timestamp: Optional[str] = None
    video: Optional[str] = None
    location_accuracy_m: Optional[float] = Field(default=None, ge=0.0)
    location_timestamp: Optional[str] = None
    camera_id: Optional[str] = Field(default="laptop-webcam-01", max_length=100)


def verify_camera_key(provided_key: Optional[str]) -> None:
    expected_key = os.getenv("WILDGUARD_CAMERA_API_KEY", "").strip()

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Camera API authentication is not configured on the server."
        )

    if not provided_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing camera API key."
        )


@app.get("/")
def home():
    if not WEBSITE_FILE.is_file():
        raise HTTPException(status_code=500, detail="Dashboard HTML file is missing.")
    return FileResponse(WEBSITE_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "service": "WildGuard API"}


@app.post("/detections")
def receive_detection(
    data: DetectionIn,
    x_camera_api_key: Optional[str] = Header(default=None, alias="X-Camera-API-Key")
):
    # Only the trusted webcam client with the secret key can write detections.
    verify_camera_key(x_camera_api_key)

    event = {
        "id": len(detection_events) + 1,
        "species": data.species,
        "confidence": data.confidence,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "location_accuracy_m": data.location_accuracy_m,
        "location_timestamp": data.location_timestamp,
        "timestamp": data.timestamp or datetime.now(timezone.utc).isoformat(),
        "video": data.video,
        "camera_id": data.camera_id,
    }

    detection_events.append(event)

    return {
        "message": "Detection received successfully",
        "event": event
    }


@app.get("/detections")
def get_detections():
    # Public read-only endpoint used by the dashboard.
    return detection_events


@app.post("/location")
def reject_public_location_updates():
    # Location is accepted only as part of an authenticated camera detection.
    raise HTTPException(
        status_code=405,
        detail="Public location updates are disabled. Camera coordinates come from authenticated detections."
    )
