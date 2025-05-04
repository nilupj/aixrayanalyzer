import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.types import LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database URL from environment
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create engine
engine = create_engine(DATABASE_URL)

# Create base class
Base = declarative_base()

# Define User model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)  # Store hashed passwords only
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

# Define Analysis model for storing X-ray analysis results
class Analysis(Base):
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    image_path = Column(String(255))  # Path to saved image file
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Metadata extracted from DICOM (if available)
    patient_id = Column(String(50), nullable=True)
    study_date = Column(String(50), nullable=True)
    modality = Column(String(50), nullable=True)
    body_part = Column(String(50), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    predictions = relationship("Prediction", back_populates="analysis")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, timestamp='{self.timestamp}')>"

# Define Prediction model for storing individual condition predictions
class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'))
    condition = Column(String(100), nullable=False)
    probability = Column(Float, nullable=False)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(condition='{self.condition}', probability={self.probability})>"

# Define function to get database session
def get_db_session():
    """Create and return a database session"""
    Session = sessionmaker(bind=engine)
    return Session()

# Create tables in the database
def init_db():
    """Initialize the database by creating all tables"""
    Base.metadata.create_all(engine)
    print("Database tables created.")

if __name__ == "__main__":
    # Initialize database when this script is run directly
    init_db()