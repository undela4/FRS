import shutil
import json
import os
from deepface import DeepFace
from sqlalchemy.orm import Session
import models
import schemas
from fastapi import UploadFile, HTTPException

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
            model_name="Facenet",
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

    db_user = models.User(
        name=name,
        image_path=image_path,
        embedding=embedding  # SQLAlchemy will handle JSON serialization
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def verify_user(db: Session, file: UploadFile):
    # 1. Save temp image
    temp_path = save_upload_file(file)
    
    # 2. Generate embedding
    target_embedding = get_embedding(temp_path)
    if not target_embedding:
        return {"status": "error", "message": "No face detected in uploaded image."}

    # 3. Compare with all users in DB (Naive approach for demo)
    # Ideally use a Vector DB or optimized search
    users = db.query(models.User).all()
    
    best_match = None
    min_distance = 100 # Initialize with high value
    threshold = 10 # Cosine distance threshold (DeepFace defaults usually around 0.4 for Facenet with cosine)
    
    # NOTE: DeepFace.verify() handles thresholds internally better, but requires two image paths.
    # Since we stored embeddings, we must manually calculate distance or use DeepFace.find if we had a DB folder.
    # For this demo, let's use a simple cosine similarity check or just iterate.
    
    # Actually, let's use DeepFace's verification logic if possible, but we only have embeddings.
    # We have to calculate cosine distance manually.
    
    import numpy as np

    match_found = False
    matched_user = None

    # Convert target embedding to numpy array
    target_emb_arr = np.array(target_embedding)

    for user in users:
        # Stored embedding is a list of floats
        if not user.embedding:
            continue
            
        db_emb_arr = np.array(user.embedding)
        
        # Calculate Cosine Distance
        # Cosine Distance = 1 - Cosine Similarity
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        
        dot_product = np.dot(target_emb_arr, db_emb_arr)
        norm_a = np.linalg.norm(target_emb_arr)
        norm_b = np.linalg.norm(db_emb_arr)
        
        cosine_similarity = dot_product / (norm_a * norm_b)
        dist = 1 - cosine_similarity
        
        # Threshold for Facenet is typically around 0.40
        if dist < 0.40:
            if dist < min_distance:
                min_distance = dist
                best_match = user
                match_found = True

    if match_found:
        return {
            "status": "success",
            "message": f"Match found: {best_match.name}",
            "distance": round(min_distance, 4),
            "user": best_match
        }
    else:
        return {
            "status": "failure",
            "message": "No match found.",
            "distance": round(min_distance, 4)
        }
