from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    name: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    image_path: str
    created_at: datetime
    # We might not want to return the full embedding in API responses by default
    
    class Config:
        orm_mode = True
