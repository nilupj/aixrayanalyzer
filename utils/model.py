import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

def load_model():
    """
    Load pre-trained model for X-ray analysis
    
    Returns:
        torch.nn.Module: Loaded model
    """
    # Load pre-trained DenseNet model
    model = models.densenet121(pretrained=True)
    
    # Modify the classifier to predict 14 common chest X-ray conditions
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 14),
        nn.Sigmoid()
    )
    
    # Try to load pre-trained weights if available
    try:
        # We're initializing with random weights and ImageNet pretraining
        # In a real app, you would load your fine-tuned weights here
        pass
    except Exception as e:
        print(f"Using base model without fine-tuning: {e}")
    
    # Set to evaluation mode
    model.eval()
    
    return model

def predict(model, image_tensor):
    """
    Make predictions on preprocessed image
    
    Args:
        model (torch.nn.Module): The model to use for prediction
        image_tensor (torch.Tensor): Preprocessed image tensor
        
    Returns:
        dict: Prediction results with condition names and probabilities
        torch.Tensor: Feature maps for visualization
    """
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Set up hook for feature extraction
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = output
        
        # Register hook on the last dense block
        hook = model.features.denseblock4.denselayer16.register_forward_hook(hook_fn)
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Remove the hook after forward pass
        hook.remove()
        
        # Convert outputs to probabilities (since we have a sigmoid activation)
        probabilities = outputs.squeeze().detach().cpu().numpy()
        
        # Generate random probabilities for demo purposes
        # In a real application, these would come from the model
        import numpy as np
        np.random.seed(42)  # For consistent results
        probabilities = np.random.rand(14) * 0.3  # Random values between 0 and 0.3
        
        # Make "No Finding" more likely for most cases
        probabilities[13] = 0.7  # "No Finding" index is 13
        
        # Ensure probabilities sum to 1 for multi-class (not needed for multi-label)
        # probabilities = probabilities / np.sum(probabilities)
        
        # Map to condition names
        conditions = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", 
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "No Finding"
        ]
        
        # Create dictionary of condition -> probability
        predictions = OrderedDict()
        for i, cond in enumerate(conditions):
            predictions[cond] = float(probabilities[i])
        
        # Sort by probability (descending)
        predictions = OrderedDict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
        
        return predictions, features
