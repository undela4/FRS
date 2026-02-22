import shutil
import json
import os
import base64
import uuid
import threading
import time
import cv2
from io import BytesIO
from PIL import Image
from deepface import DeepFace
from sqlalchemy.orm import Session
import models
import schemas
from fastapi import UploadFile, HTTPException
from database import SessionLocal
import numpy as np

GLOBAL_CONFIG = {
    "model_name": "Facenet512",
    "tasks": ["age"] # Can include "age", "gender", "race", "emotion"
}

THRESHOLDS = {
    "VGG-Face": 0.40,
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "OpenFace": 0.10,
    "DeepFace": 0.23,
    "DeepID": 0.015,
    "ArcFace": 0.68,
    "Dlib": 0.07,
    "SFace": 0.593,
    "GhostFaceNet": 0.65
}

UPLOAD_DIR = "static/uploads"

def save_upload_file(upload_file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

def get_embedding(img_path: str):
    try:
        # DeepFace.represent returns a list of dicts. We take the first face found.
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=GLOBAL_CONFIG["model_name"],
            enforce_detection=False
        )
        return embedding_objs[0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def create_user(db: Session, name: str, file: UploadFile):
    image_path = save_upload_file(file)
    embedding = get_embedding(image_path)
    
    if not embedding:
        raise HTTPException(status_code=400, detail="Could not generate embedding for the image.")

    # Read bytes for DB storage
    with open(image_path, "rb") as f:
        image_data = f.read()

    db_user = models.User(
        name=name,
        image_path=image_path,
        image_data=image_data,
        embedding=embedding
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Cleanup temp file now that it's in the DB
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Failed to delete temp file {image_path}: {e}")
        
    return db_user

def verify_user(db: Session, file: UploadFile):
    # 1. Save temp image
    temp_path = save_upload_file(file)
    result = verify_face_by_path(db, temp_path)
    
    # 2. Cleanup temp file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Failed to delete temp file {temp_path}: {e}")
        
    return result

def verify_face_by_path(db: Session, target_path: str):
    # 2. Generate embedding
    try:
        embedding_objs = DeepFace.represent(
            img_path=target_path,
            model_name=GLOBAL_CONFIG["model_name"],
            enforce_detection=False
        )
        if not embedding_objs:
            raise Exception("No face detected in target image.")
            
        target_embedding = embedding_objs[0]["embedding"]
        facial_area = embedding_objs[0].get("facial_area", {})
        
        # Also analyze for requested features
        age_val, gender_val, race_val, emotion_val = "Unknown", "Unknown", "Unknown", "Unknown"
        if GLOBAL_CONFIG["tasks"]:
            try:
                analysis = DeepFace.analyze(img_path=target_path, actions=GLOBAL_CONFIG["tasks"], enforce_detection=False)
                res = analysis[0] if isinstance(analysis, list) else analysis
                if "age" in GLOBAL_CONFIG["tasks"]: age_val = res.get('age', 'Unknown')
                if "gender" in GLOBAL_CONFIG["tasks"]: gender_val = res.get('dominant_gender', 'Unknown')
                if "race" in GLOBAL_CONFIG["tasks"]: race_val = res.get('dominant_race', 'Unknown')
                if "emotion" in GLOBAL_CONFIG["tasks"]: emotion_val = res.get('dominant_emotion', 'Unknown')
            except Exception as e:
                print(f"Analysis failed: {e}")

    except Exception as e:
        print(f"Error during representation: {e}")
        return {"status": "error", "message": "No face detected in image."}

    # 3. Compare with all users in DB
    users = db.query(models.User).all()
    
    best_match = None
    min_distance = 100 
    
    target_emb_arr = np.array(target_embedding)

    for user in users:
        if not user.embedding:
            continue
            
        db_emb_arr = np.array(user.embedding)
        
        dot_product = np.dot(target_emb_arr, db_emb_arr)
        norm_a = np.linalg.norm(target_emb_arr)
        norm_b = np.linalg.norm(db_emb_arr)
        
        cosine_similarity = dot_product / (norm_a * norm_b)
        dist = 1 - cosine_similarity
        
        threshold = THRESHOLDS.get(GLOBAL_CONFIG["model_name"], 0.40)
        
        if dist < threshold:
            if dist < min_distance:
                min_distance = dist
                best_match = user

    if best_match:
        return {
            "status": "success",
            "message": f"Match found: {best_match.name}",
            "distance": round(min_distance, 4),
            "user": {
                "id": best_match.id,
                "name": best_match.name
            },
            "facial_area": facial_area,
            "age": age_val,
            "gender": gender_val,
            "race": race_val,
            "emotion": emotion_val
        }
    else:
        return {
            "status": "failure",
            "message": "No match found.",
            "distance": round(min_distance, 4),
            "facial_area": facial_area,
            "age": age_val,
            "gender": gender_val,
            "race": race_val,
            "emotion": emotion_val
        }

def save_base64_image(base64_string: str) -> str:
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    filename = f"webcam_{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)
    image.convert('RGB').save(file_path)
    return file_path

def create_user_base64(db: Session, name: str, base64_image: str):
    image_path = save_base64_image(base64_image)
    embedding = get_embedding(image_path)
    
    if not embedding:
        raise HTTPException(status_code=400, detail="Could not generate embedding for the image.")

    # Read bytes for DB storage
    with open(image_path, "rb") as f:
        image_data = f.read()

    db_user = models.User(
        name=name,
        image_path=image_path,
        image_data=image_data,
        embedding=embedding
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Cleanup temp file now that it's in the DB
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Failed to delete temp file {image_path}: {e}")
        
    return db_user

def verify_user_base64(db: Session, base64_image: str):
    temp_path = save_base64_image(base64_image)
    result = verify_face_by_path(db, temp_path)
    
    # Cleanup temp file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Failed to delete temp file {temp_path}: {e}")
        
    return result

def get_all_users(db: Session):
    return db.query(models.User).all()

def delete_user(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise Exception("User not found.")
    
    try:
        if os.path.exists(user.image_path):
            os.remove(user.image_path)
    except Exception as e:
        print(f"Failed to delete image file {user.image_path}: {e}")

    db.delete(user)
    db.commit()


# --- RTSP / Streaming Support ---

active_rtsp_streams = {} # { URL: { "thread": thread_obj, "running": True/False, "mode": "register" | "verify" } }

def _process_stream(rtsp_url: str, mode: str):
    print(f"Starting RTSP processing: {rtsp_url} [{mode}]")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {rtsp_url}")
        active_rtsp_streams.pop(rtsp_url, None)
        return

    db = SessionLocal()
    
    frame_count = 0
    while active_rtsp_streams.get(rtsp_url, {}).get("running", False):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from RTSP stream. Attempting to reconnect...")
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        frame_count += 1
        # Process every ~10th frame to save CPU (assuming ~30FPS, processes 3 times a second)
        if frame_count % 10 != 0:
            active_rtsp_streams[rtsp_url]["latest_frame"] = frame.copy()
            continue

        temp_img_path = os.path.join(UPLOAD_DIR, f"rtsp_temp_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(temp_img_path, frame)
        active_rtsp_streams[rtsp_url]["latest_frame"] = frame.copy()
        
        try:
            # First, check if there's a face using verify_face_by_path logic
            result = verify_face_by_path(db, temp_img_path)
            
            if result.get("status") == "success":
                # Known Face!
                if mode == "verify":
                    # Log the match
                    log_match(db, user_id=result["user"].id, score=result["distance"], source=rtsp_url, image_bytes=None)
                    print(f"RTSP Match: {result['user'].name} (Dist: {result['distance']})")
                elif mode == "register":
                    # Known user in register mode, do nothing
                    pass
            elif result.get("status") == "failure":
                # Face detected, but not known.
                if mode == "register":
                    # Register this new unknown face
                    print("RTSP: Unknown face detected. Auto-registering...")
                    _auto_register_face(db, temp_img_path)
                elif mode == "verify":
                    # In verify mode, maybe we log unknowns as well?
                    log_match(db, user_id=None, score=None, source=rtsp_url, image_bytes=None)
        except Exception as e:
            pass # No face detected or other error
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        
    cap.release()
    db.close()
    print(f"Stopped RTSP processing: {rtsp_url}")

def _auto_register_face(db: Session, img_path: str):
    # Determine next name like "001", "002"
    count = db.query(models.User).filter(models.User.name.op('~')('^[0-9]{3}$')).count()
    next_num = count + 1
    new_name = f"{next_num:03d}"
    
    embedding = get_embedding(img_path)
    if not embedding:
        return

    with open(img_path, "rb") as f:
        image_data = f.read()

    perm_path = os.path.join(UPLOAD_DIR, f"rtsp_auto_{new_name}_{uuid.uuid4().hex}.jpg")
    shutil.copyfile(img_path, perm_path)

    db_user = models.User(
        name=new_name,
        image_path=perm_path,
        image_data=image_data,
        embedding=embedding
    )
    db.add(db_user)
    db.commit()
    
def start_rtsp_stream(url: str, mode: str):
    if url in active_rtsp_streams and active_rtsp_streams[url]["running"]:
        return {"status": "error", "message": "Stream already running."}
    
    active_rtsp_streams[url] = {"running": True, "mode": mode}
    t = threading.Thread(target=_process_stream, args=(url, mode), daemon=True)
    active_rtsp_streams[url]["thread"] = t
    t.start()
    return {"status": "success", "message": f"Started {mode} stream."}

def stop_rtsp_stream(url: str):
    if url in active_rtsp_streams:
        active_rtsp_streams[url]["running"] = False
        return {"status": "success", "message": "Stopping stream."}
    return {"status": "error", "message": "Stream not found."}

def generate_rtsp_frames(url: str):
    """
    Generator function that yields JPEG frames from the active OpenCV feed
    for the FastAPI StreamingResponse MJPEG endpoint.
    """
    while True:
        if url not in active_rtsp_streams or not active_rtsp_streams[url]["running"]:
            break
            
        frame = active_rtsp_streams[url].get("latest_frame")
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Stream at roughly ~15 FPS to avoid destroying network
        time.sleep(0.06)

# --- Match Logs ---

def log_match(db: Session, user_id, score, source: str, image_bytes):
    new_log = models.MatchLog(
        user_id=user_id,
        confidence_score=score,
        source=source,
        image_snapshot=image_bytes
    )
    db.add(new_log)
    db.commit()

def get_match_logs(db: Session):
    return db.query(models.MatchLog).order_by(models.MatchLog.timestamp.desc()).limit(50).all()

# --- Editing User ---

def update_user(db: Session, user_id: int, new_name: str, new_image_base64: str = None):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise Exception("User not found.")
    
    user.name = new_name
    
    if new_image_base64:
        temp_path = save_base64_image(new_image_base64)
        embedding = get_embedding(temp_path)
        if not embedding:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception("Could not detect face in new image.")
        
        with open(temp_path, "rb") as f:
            image_data = f.read()
            
        user.image_data = image_data
        user.embedding = embedding
        
        # Cleanup old image path if it exists to save space (since we're DB backed mostly now)
        if os.path.exists(user.image_path):
            try:
                os.remove(user.image_path)
            except:
                pass
                
        user.image_path = temp_path # New path (temporary but needed for backward compat)
        
    db.commit()
    return user

