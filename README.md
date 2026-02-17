# Facial Recognition System (FRS) – Demo Project

## 📌 Project Overview

This project is a **web-based Facial Recognition System (FRS)** built for demonstration purposes.
It allows users to:

* Register faces
* Store facial embeddings in a PostgreSQL database
* Verify or identify users using facial recognition
* Display results through a simple TailwindCSS-based UI

The system uses **DeepFace** for facial recognition and **FastAPI** as the backend framework.

---

## 🏗️ Tech Stack

### Backend

* **FastAPI** – High-performance Python web framework
* **DeepFace** – Facial recognition & embedding extraction
* **PostgreSQL** – Persistent storage for users & embeddings
* **SQLAlchemy** – ORM for database interaction
* **Uvicorn** – ASGI server

### Frontend

* **Jinja2 Templates** – Server-side rendering
* **TailwindCSS** – Clean and responsive UI
* HTML5 (Camera Upload via input or JS)

---

## 🧠 System Architecture

```
User (Browser)
      ↓
FastAPI Routes
      ↓
DeepFace (Generate Embeddings)
      ↓
PostgreSQL (Store & Compare Embeddings)
      ↓
Response to UI (Match / Not Match)
```

---

## ⚙️ Core Features

### 1️⃣ Face Registration

* Upload image
* Extract face embedding using DeepFace
* Store:

  * User Name
  * Image path
  * Face embedding (vector)
  * Timestamp

### 2️⃣ Face Verification (1:1)

* Upload new image
* Generate embedding
* Compare with stored embedding
* Return match score

### 3️⃣ Face Identification (1:N)

* Upload image
* Compare embedding with all stored embeddings
* Return best match

---

## 🗄️ Database Design (PostgreSQL)

### Table: `users`

| Column     | Type        | Description       |
| ---------- | ----------- | ----------------- |
| id         | SERIAL (PK) | User ID           |
| name       | VARCHAR     | Person name       |
| image_path | TEXT        | Stored image path |
| embedding  | TEXT / JSON | Face vector       |
| created_at | TIMESTAMP   | Record time       |

> For demo purposes, embeddings can be stored as JSON or TEXT.

---

## 📂 Suggested Project Structure

```
frs_app/
│
├── main.py
├── database.py
├── models.py
├── schemas.py
├── services/
│   └── face_service.py
├── templates/
│   ├── base.html
│   ├── register.html
│   ├── verify.html
│   └── result.html
├── static/
│   └── uploads/
├── requirements.txt
└── README.md
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

* Home Page
* Register Face
* Verify Face
* Identification Result Page

Use:

* Clean card layout
* Image preview
* Match percentage display
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
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload
```

PostgreSQL should be running locally.

---

## 📈 Future Enhancements

* Live webcam capture
* Face liveness detection
* Docker containerization
* Redis caching
* Vector database (FAISS) for faster comparison
* JWT Authentication
* Role-based access control

---

## 🎯 Purpose

This project demonstrates:

* Backend-driven UI (No React/Django)
* Facial recognition integration
* Embedding storage in relational DB
* FastAPI template rendering
* Clean UI with Tailwind
 