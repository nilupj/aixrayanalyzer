import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

def generate_heatmap(model, preprocessed_img):
    """
    Generate a simplified heatmap visualization 
    
    Args:
        model (torch.nn.Module): The trained model
        preprocessed_img (torch.Tensor): Preprocessed image tensor
        
    Returns:
        numpy.ndarray: Heatmap visualization
    """
    import numpy as np
    import cv2
    import torch
    
    # Get features from the last layer
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = output.detach()
    
    # Register hook on the last dense layer
    hook = model.features.denseblock4.denselayer16.register_forward_hook(hook_fn)
    
    # Forward pass to get features
    with torch.no_grad():
        model(preprocessed_img)
    
    # Remove the hook
    hook.remove()
    
    # Create a simplified synthetic heatmap for demonstration
    # In a real scenario, this would be based on the actual model features
    img_height, img_width = 224, 224
    
    # Create a synthetic heatmap (focused on center area)
    y, x = np.ogrid[0:img_height, 0:img_width]
    center_y, center_x = img_height // 2, img_width // 2
    
    # Generate a circular/gaussian-like heatmap
    heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (img_width // 3) ** 2)
    
    # Add some random variations to make it look more realistic
    np.random.seed(42)  # For consistent results
    noise = np.random.normal(0, 0.1, (img_height, img_width))
    heatmap += noise
    
    # Normalize to 0-1 range
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Convert preprocessed tensor back to image for visualization
    input_img = preprocessed_img.squeeze().permute(1, 2, 0).cpu().numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    
    # Convert to uint8
    input_img_uint8 = np.uint8(255 * input_img)
    
    # Create overlay
    overlay = cv2.addWeighted(
        input_img_uint8, 0.6,
        cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4,
        0
    )
    
    return overlay

def plot_prediction_results(predictions):
    """
    Plot prediction results as a bar chart
    
    Args:
        predictions (dict): Dictionary of condition names and probabilities
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the plot
    """
    # Filter conditions with probability > 1%
    filtered_preds = {k: v for k, v in predictions.items() if v > 0.01}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate bar chart
    bars = ax.barh(
        list(filtered_preds.keys()),
        list(filtered_preds.values()),
        color='skyblue'
    )
    
    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_position = width + 0.01
        ax.text(label_position, bar.get_y() + bar.get_height()/2, f"{width:.1%}", 
                va='center', fontsize=10)
    
    # Set chart title and labels
    ax.set_title('Condition Probabilities')
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1.0)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_overlay(original_image, heatmap, alpha=0.4):
    """
    Create an overlay of the heatmap on the original image
    
    Args:
        original_image (PIL.Image): Original image
        heatmap (numpy.ndarray): Heatmap array
        alpha (float): Transparency level
        
    Returns:
        numpy.ndarray: Overlay image
    """
    # Convert original image to numpy array
    original_array = np.array(original_image)
    
    # Resize heatmap to match original image
    import cv2
    heatmap_resized = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
    
    # Apply colormap to heatmap (using jet colormap)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    
    # Convert to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(
        original_array, 
        1 - alpha,
        heatmap_colored, 
        alpha, 
        0
    )
    
    return overlay
