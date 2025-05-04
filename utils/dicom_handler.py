import numpy as np
import io
import pydicom
import SimpleITK as sitk
from PIL import Image

def read_dicom(file_obj):
    """
    Read DICOM file and convert to a numpy array
    
    Args:
        file_obj: File-like object containing DICOM data
        
    Returns:
        numpy.ndarray: Image data as a numpy array
    """
    try:
        # First try using SimpleITK
        return read_dicom_with_sitk(file_obj)
    except Exception:
        # Fall back to pydicom
        return read_dicom_with_pydicom(file_obj)

def read_dicom_with_sitk(file_obj):
    """
    Read DICOM file using SimpleITK
    
    Args:
        file_obj: File-like object containing DICOM data
        
    Returns:
        numpy.ndarray: Image data as a numpy array
    """
    # Save the file to a temporary buffer
    buffer = io.BytesIO(file_obj.read())
    file_obj.seek(0)  # Reset file pointer
    
    # Create a SimpleITK ImageFileReader
    reader = sitk.ImageFileReader()
    reader.SetFileName(buffer.getvalue())
    reader.ReadImageInformation()
    
    # Read the image
    image = reader.Execute()
    
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Normalize to 0-255 range for visualization
    array = array.astype(np.float32)
    array = (array - array.min()) / (array.max() - array.min()) * 255.0
    array = array.astype(np.uint8)
    
    # Handle 3D images (take first slice)
    if array.ndim > 2:
        array = array[0]
    
    return array

def read_dicom_with_pydicom(file_obj):
    """
    Read DICOM file using pydicom
    
    Args:
        file_obj: File-like object containing DICOM data
        
    Returns:
        numpy.ndarray: Image data as a numpy array
    """
    # Read DICOM file
    dcm = pydicom.dcmread(file_obj)
    
    # Extract image data
    array = dcm.pixel_array
    
    # Normalize to 0-255 range for visualization
    array = array.astype(np.float32)
    array = (array - array.min()) / (array.max() - array.min()) * 255.0
    array = array.astype(np.uint8)
    
    return array

def dicom_to_pil(dicom_array):
    """
    Convert DICOM numpy array to PIL Image
    
    Args:
        dicom_array (numpy.ndarray): DICOM image array
        
    Returns:
        PIL.Image: PIL Image object
    """
    # Make sure array is 8-bit
    if dicom_array.dtype != np.uint8:
        dicom_array = ((dicom_array - dicom_array.min()) / 
                       (dicom_array.max() - dicom_array.min()) * 255).astype(np.uint8)
    
    # Create PIL image
    if len(dicom_array.shape) == 2:
        # Grayscale image
        img = Image.fromarray(dicom_array, mode='L')
        return img.convert('RGB')  # Convert to RGB
    else:
        # Already a color image
        return Image.fromarray(dicom_array)

def extract_dicom_metadata(file_obj):
    """
    Extract metadata from DICOM file
    
    Args:
        file_obj: File-like object containing DICOM data
        
    Returns:
        dict: Dictionary of DICOM metadata
    """
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(file_obj)
        file_obj.seek(0)  # Reset file pointer
        
        # Extract basic metadata
        metadata = {
            'PatientID': getattr(dcm, 'PatientID', 'Unknown'),
            'StudyDate': getattr(dcm, 'StudyDate', 'Unknown'),
            'Modality': getattr(dcm, 'Modality', 'Unknown'),
            'BodyPartExamined': getattr(dcm, 'BodyPartExamined', 'Unknown'),
            'StudyDescription': getattr(dcm, 'StudyDescription', 'Unknown'),
            'ImageSize': f"{getattr(dcm, 'Rows', 'Unknown')}x{getattr(dcm, 'Columns', 'Unknown')}"
        }
        
        return metadata
    except Exception as e:
        return {'Error': str(e)}
