import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image): Input X-ray image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define preprocessing steps similar to training data
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Apply preprocessing
    input_tensor = preprocess(image)
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor

def is_valid_image(image):
    """
    Check if the image appears to be a valid X-ray
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        bool: True if appears to be valid X-ray, False otherwise
    """
    # Convert to grayscale for analysis
    gray_img = image.convert('L')
    np_img = np.array(gray_img)
    
    # Check image characteristics typical of X-rays
    # 1. Check for proper contrast
    contrast = np.std(np_img)
    if contrast < 20:  # Low contrast images are unlikely to be X-rays
        return False
    
    # 2. Check for reasonable histogram distribution (X-rays have specific histogram patterns)
    hist = cv2.calcHist([np_img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # X-rays typically have strong peaks in darker and lighter areas
    dark_ratio = np.sum(hist[:85])
    light_ratio = np.sum(hist[170:])
    mid_ratio = np.sum(hist[85:170])
    
    # X-rays usually have more dark and light areas than middle areas
    if mid_ratio > 0.7:
        return False
    
    # 3. Check if the image has reasonable dimensions
    if min(image.size) < 100:  # Very small images are unlikely to be X-rays
        return False
    
    return True

def adjust_contrast(image, factor=1.5):
    """
    Adjust the contrast of an image
    
    Args:
        image (PIL.Image): Input image
        factor (float): Contrast adjustment factor
        
    Returns:
        PIL.Image: Contrast-adjusted image
    """
    # Convert to numpy array
    np_img = np.array(image).astype(float)
    
    # Apply contrast adjustment
    mean = np.mean(np_img)
    adjusted = (np_img - mean) * factor + mean
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    # Convert back to PIL image
    return Image.fromarray(adjusted)

def resize_image(image, size=(512, 512)):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image (PIL.Image): Input image
        size (tuple): Target size (width, height)
        
    Returns:
        PIL.Image: Resized image
    """
    # Calculate aspect ratio
    width, height = image.size
    aspect = width / height
    
    # Determine new dimensions
    if aspect > 1:
        # Image is wider than tall
        new_width = size[0]
        new_height = int(new_width / aspect)
    else:
        # Image is taller than wide
        new_height = size[1]
        new_width = int(new_height * aspect)
    
    # Resize the image
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with target size
    new_img = Image.new('RGB', size, (0, 0, 0))
    
    # Paste resized image at center
    paste_x = (size[0] - new_width) // 2
    paste_y = (size[1] - new_height) // 2
    new_img.paste(resized, (paste_x, paste_y))
    
    return new_img
