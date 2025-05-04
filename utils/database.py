import os
import datetime
import hashlib
from models import get_db_session, User, Analysis, Prediction

def save_analysis_result(user_id, image_path, predictions, metadata=None):
    """
    Save analysis results to the database
    
    Args:
        user_id (int): User ID (or None for anonymous user)
        image_path (str): Path to the saved image file
        predictions (dict): Dictionary of condition names and probabilities
        metadata (dict): Dictionary of DICOM metadata (if available)
        
    Returns:
        Analysis: The saved Analysis object
    """
    # Create a database session
    session = get_db_session()
    
    try:
        # Create a new Analysis record
        analysis = Analysis(
            user_id=user_id,
            image_path=image_path,
            timestamp=datetime.datetime.utcnow()
        )
        
        # Add metadata if available
        if metadata:
            analysis.patient_id = metadata.get('PatientID', None)
            analysis.study_date = metadata.get('StudyDate', None)
            analysis.modality = metadata.get('Modality', None)
            analysis.body_part = metadata.get('BodyPartExamined', None)
        
        # Add the analysis to the session
        session.add(analysis)
        session.commit()
        
        # Create prediction records for each condition
        for condition, probability in predictions.items():
            prediction = Prediction(
                analysis_id=analysis.id,
                condition=condition,
                probability=probability
            )
            session.add(prediction)
        
        # Commit the changes
        session.commit()
        
        return analysis
    except Exception as e:
        # Roll back any changes if an error occurs
        session.rollback()
        raise e
    finally:
        # Close the session
        session.close()

def get_user_analyses(user_id, limit=10):
    """
    Get a list of analyses for a specific user
    
    Args:
        user_id (int): User ID
        limit (int): Maximum number of analyses to return
        
    Returns:
        list: List of Analysis objects
    """
    session = get_db_session()
    try:
        analyses = session.query(Analysis).filter(
            Analysis.user_id == user_id
        ).order_by(Analysis.timestamp.desc()).limit(limit).all()
        
        return analyses
    finally:
        session.close()

def get_analysis_with_predictions(analysis_id):
    """
    Get an analysis and its predictions
    
    Args:
        analysis_id (int): Analysis ID
        
    Returns:
        tuple: (Analysis object, list of Prediction objects)
    """
    session = get_db_session()
    try:
        analysis = session.query(Analysis).filter(
            Analysis.id == analysis_id
        ).first()
        
        if not analysis:
            return None, []
        
        predictions = session.query(Prediction).filter(
            Prediction.analysis_id == analysis_id
        ).order_by(Prediction.probability.desc()).all()
        
        return analysis, predictions
    finally:
        session.close()

def create_user(username, email, password):
    """
    Create a new user
    
    Args:
        username (str): Username
        email (str): Email address
        password (str): Plain text password (will be hashed)
        
    Returns:
        User: The created user object
    """
    session = get_db_session()
    try:
        # Hash the password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Create a new user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            created_at=datetime.datetime.utcnow(),
            is_active=True
        )
        
        # Add the user to the session
        session.add(user)
        session.commit()
        
        return user
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def authenticate_user(username, password):
    """
    Authenticate a user
    
    Args:
        username (str): Username
        password (str): Plain text password
        
    Returns:
        User: User object if authentication is successful, None otherwise
    """
    session = get_db_session()
    try:
        # Hash the password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Find the user
        user = session.query(User).filter(
            User.username == username,
            User.password_hash == password_hash,
            User.is_active == True
        ).first()
        
        return user
    finally:
        session.close()