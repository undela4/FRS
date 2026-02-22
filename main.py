from fastapi import FastAPI, Request, File, UploadFile, Depends, Form, HTTPException, Response
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
import models
import schemas
import database
from services import face_service

# Initialize Database
models.Base.metadata.create_all(bind=database.engine)
from passlib.context import CryptContext

app = FastAPI(title="Facial Recognition System")

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def post_register(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        user = face_service.create_user(db, name, file)
        message = f"Successfully registered {user.name}!"
        return templates.TemplateResponse("register.html", {"request": request, "message": message, "message_type": "success"})
    except Exception as e:
        return templates.TemplateResponse("register.html", {"request": request, "message": str(e), "message_type": "error"})

@app.get("/verify", response_class=HTMLResponse)
async def get_verify(request: Request):
    return templates.TemplateResponse("verify.html", {"request": request})

@app.post("/verify", response_class=HTMLResponse)
async def post_verify(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        result = face_service.verify_user(db, file)
        return templates.TemplateResponse("result.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse("verify.html", {"request": request, "message": str(e), "message_type": "error"})

@app.get("/users", response_class=HTMLResponse)
async def get_users(request: Request, db: Session = Depends(get_db)):
    users = face_service.get_all_users(db)
    return templates.TemplateResponse("users.html", {"request": request, "users": users})

@app.get("/users/{user_id}/image")
async def get_user_image(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.image_data:
        return Response(content=user.image_data, media_type="image/jpeg")
    
    # Fallback for old records without image_data
    import os
    if os.path.exists(user.image_path):
        with open(user.image_path, "rb") as f:
            return Response(content=f.read(), media_type="image/jpeg")
            
    raise HTTPException(status_code=404, detail="Image data not available")

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    try:
        face_service.delete_user(db, user_id)
        return {"status": "success", "message": "User deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class WebcamImage(BaseModel):
    image: str

class WebcamRegister(BaseModel):
    name: str
    image: str

@app.get("/webcam", response_class=HTMLResponse)
async def get_webcam(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})

@app.post("/webcam/verify")
async def verify_webcam(data: WebcamImage, db: Session = Depends(get_db)):
    try:
        result = face_service.verify_user_base64(db, data.image)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/webcam/register")
async def register_webcam(data: WebcamRegister, db: Session = Depends(get_db)):
    try:
        user = face_service.create_user_base64(db, data.name, data.image)
        return {"status": "success", "message": f"Successfully registered {user.name}!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class RTSPStart(BaseModel):
    url: str
    mode: str # "register" or "verify"

@app.post("/rtsp/start")
async def start_rtsp(data: RTSPStart):
    if data.mode not in ["register", "verify"]:
        raise HTTPException(status_code=400, detail="Invalid mode.")
    return face_service.start_rtsp_stream(data.url, data.mode)

@app.post("/rtsp/stop")
async def stop_rtsp(data: RTSPStart):
    return face_service.stop_rtsp_stream(data.url)

@app.get("/rtsp/feed")
async def rtsp_feed(url: str):
    """
    HTTP MJPEG streaming endpoint for the frontend.
    Yields JPEG frames continuously.
    """
    if not url:
        raise HTTPException(status_code=400, detail="Missing RTSP URL.")
    
    return StreamingResponse(
        face_service.generate_rtsp_frames(url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/logs/matches")
async def get_logs(db: Session = Depends(get_db)):
    logs = face_service.get_match_logs(db)
    return logs

class AppConfig(BaseModel):
    model_name: str
    tasks: list[str]

@app.get("/config")
async def get_config():
    return face_service.GLOBAL_CONFIG

@app.put("/config")
async def update_config(data: AppConfig):
    face_service.GLOBAL_CONFIG["model_name"] = data.model_name
    face_service.GLOBAL_CONFIG["tasks"] = data.tasks
    return {"status": "success", "message": "Configuration updated successfully!"}

class UserUpdate(BaseModel):
    name: str
    image: str = None

@app.put("/users/{user_id}")
async def update_user(user_id: int, data: UserUpdate, db: Session = Depends(get_db)):
    try:
        user = face_service.update_user(db, user_id, data.name, data.image)
        return {"status": "success", "message": f"Successfully updated user {user.name}."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/add_faces", response_class=HTMLResponse)
async def get_add_faces(request: Request):
    return templates.TemplateResponse("add_faces.html", {"request": request})

@app.get("/verification", response_class=HTMLResponse)
async def get_verification(request: Request):
    return templates.TemplateResponse("verification.html", {"request": request})

@app.get("/logs", response_class=HTMLResponse)
async def get_logs_page(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request})
