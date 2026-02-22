# Facial Recognition System (FRS) – Advanced Demo

## 📌 Project Overview

This project is a **robust, web-based Facial Recognition System (FRS)** built for advanced demonstration purposes.
It allows users to:

* Manually register faces via upload or webcam
* Automatically detect and register faces via RTSP camera streams
* Verify or identify users using 1:1 and 1:N facial recognition
* Perform real-time auto-identification with bounding boxes and emotion detection
* Track history of verifications with a dedicated Match Logs dashboard
* Store facial embeddings and raw image data securely in a PostgreSQL database
* Display results through a modern TailwindCSS-based UI

The system uses **DeepFace** for facial recognition/analysis, **OpenCV** for stream processing, and **FastAPI** as the backend framework.

---

## 🏗️ Tech Stack

### Backend

* **FastAPI** – High-performance Python web framework
* **DeepFace** – Facial recognition, embedding extraction, & emotion analysis
* **OpenCV (`cv2`)** – Processing RTSP IP Camera streams and Webcam feeds
* **PostgreSQL** – Persistent storage for users, image blobs, match logs, & embeddings
* **SQLAlchemy** – ORM for database interaction
* **Uvicorn** – ASGI server

### Frontend

* **Jinja2 Templates** – Server-side HTML rendering
* **TailwindCSS** – Clean and responsive modern UI
* Vanilla JS & JS Canvas – Drawing real-time bounding boxes via polling

---

## 🧠 System Architecture

```
User (Browser/Camera) OR RTSP Stream
             ↓
    FastAPI Endpoints / OpenCV Loop
             ↓
    DeepFace (Embeddings & Emotion Analysis)
             ↓
    PostgreSQL (Store/Retrieve Images, Compare Embeddings, Log Matches)
             ↓
    Response Overlay on UI (Bounding Box, Match Status, Logs Table)
```

---

### 1️⃣ Face Registration (Manual & Auto)
* Upload an image, snap from a live webcam, or supply an RTSP stream.
* Extract face embedding using DeepFace.
* Store the user name, actual image byte data, and vector embedding persistently in the database.
* Auto-detection from RTSP assigns sequential names (e.g., 001, 002) to unknown faces.

### 2️⃣ Face Verification & Auto-Identification
* Single Upload: Upload a new image to generate an embedding and compare it with the stored database.
* Real-time WebCam Auto-Track: Tracks faces live, drawing green (match) or red (unknown) bounding boxes and analyzing dominant emotion.
* RTSP Auto-Verify: Continuously processes frames to detect known users passing by.

### 3️⃣ Match Logging & Auditing
* Every time a face is verified or identified via streams, it adds an entry to the `MatchLog` database table.
* The web UI features a dedicated Logs dashboard to review recent verifications with confidence scores.

### 4️⃣ User Management
* Complete dashboard to search, view, and manage registered profiles.
* Update names or re-register new photos for existing users dynamically.

---

## 🗄️ Database Design (PostgreSQL)

### Table: `users`
| Column     | Type        | Description       |
| ---------- | ----------- | ----------------- |
| id         | SERIAL (PK) | User ID           |
| name       | VARCHAR     | Person name       |
| image_path | TEXT        | Temp path string  |
| image_data | BYTEA       | Actual image blob |
| embedding  | JSONB       | Face vector (float array) |
| created_at | TIMESTAMP   | Record time       |

### Table: `match_logs`
| Column           | Type        | Description          |
| ---------------- | ----------- | -------------------- |
| id               | SERIAL (PK) | Log ID               |
| user_id          | INTEGER     | Foreign Key to Users |
| timestamp        | TIMESTAMP   | Match time           |
| confidence_score | FLOAT       | Distance metric      |
| source           | VARCHAR     | RTSP or Webcam       |

---

## 📂 Suggested Project Structure

```
frs_app/
├── main.py
├── database.py
├── models.py
├── schemas.py
├── services/
│   └── face_service.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── add_faces.html
│   ├── verification.html
│   ├── users.html
│   └── logs.html
├── static/
│   └── uploads/     # Temporary ephemeral processing folder
└── requirements.txt
```

---

## 🔍 DeepFace Usage Flow

1. Load image
2. Detect face
3. Generate embedding
4. Compare embeddings using cosine similarity

Example logic:

```python
from deepface import DeepFace

embedding = DeepFace.represent(
    img_path="image.jpg",
    model_name="Facenet",
    enforce_detection=False
)
```

---

## 🎨 UI Design (Tailwind)

Pages:

* Home Page Dashboard
* Manage Users (Search & Edit Data)
* Add Faces (Webcam & RTSP)
* Verification (Upload, Webcam Auto-Track, & RTSP)
* Match Logs

Use:

* Clean card layout
* JS Canvas overlays for Bounding Boxes & Emotion Text
* Modals for editing profiles
* Success / failure color indicators

---

## 🔐 Security Considerations (Important Even for Demo)

* Store images securely
* Validate uploaded file types
* Limit image size
* Do not expose raw embeddings in API response
* Use HTTPS in production

---

## 🚀 How to Run

```bash
./run.sh
```

Or manually:
```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
PostgreSQL should be running locally or linked via environment variables.

---

## 🎯 Purpose

This project demonstrates:

* Backend-driven UI (No React/Django)
* Facial recognition integration
* Embedding storage in relational DB
* FastAPI template rendering
* Clean UI with Tailwind
 