from fastapi import FastAPI, Request, File, UploadFile, Depends, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import models
import schemas
import database
from services import face_service

# Initialize Database
models.Base.metadata.create_all(bind=database.engine)

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
