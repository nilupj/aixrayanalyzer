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
        # Forward pass
        outputs = model(image_tensor)
        features = None
        
        # For heatmap visualization, we need the features from the last conv layer
        def hook_fn(module, input, output):
            nonlocal features
            features = output
        
        # Get the last dense block's output for visualization
        model.features.denseblock4.register_forward_hook(hook_fn)
        _ = model(image_tensor)  # Run again to get features
        
        # Convert outputs to probabilities
        probabilities = outputs.squeeze().numpy()
        
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
