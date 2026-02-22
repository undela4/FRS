from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, LargeBinary, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    image_path = Column(Text, nullable=False)
    image_data = Column(LargeBinary, nullable=True) # Added for DB storage
    # Storing embedding as a JSON array of floats
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    match_logs = relationship("MatchLog", back_populates="user")

class MatchLog(Base):
    __tablename__ = "match_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Nullable for unknown faces seen in stream
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    confidence_score = Column(Float, nullable=True)
    source = Column(String, index=True) # e.g. "Webcam", "RTSP - Camera 1"
    image_snapshot = Column(LargeBinary, nullable=True) # snapshot of the match

    user = relationship("User", back_populates="match_logs")
