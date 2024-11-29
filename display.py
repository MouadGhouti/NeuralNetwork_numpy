import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform

def visNetwork(network, error):
    plt.ion()  # Enable interactive mode
    
    # Filter for dense and dropout layers only
    network = list(filter(lambda x: x.__class__.__name__ == 'Dense' or x.__class__.__name__ == 'Dropout', network)) 

    # Initialize subplots if they don't already exist
    if not hasattr(visNetwork, 'fig'):
        visNetwork.fig, visNetwork.axes = plt.subplots(1, len(network) + 1, figsize=(15, 5))
        visNetwork.images = []
        visNetwork.loss_values = []  # List to store loss values over time
        # Initialize plots for each layer's weights
        for i, (layer, ax) in enumerate(zip(network, visNetwork.axes[:-1])):
            color = list(plt.cm._colormaps)[i*2+1]
            img = ax.matshow(layer.weights, cmap=color)
            ax.set_title(f"{layer.__class__.__name__}")
            ax.axis('off')
            visNetwork.images.append(img)
        
        # Initialize the loss plot in the last subplot
        visNetwork.loss_ax = visNetwork.axes[-1]
        visNetwork.loss_line, = visNetwork.loss_ax.plot([], [], 'r-')  # Red line for loss
        visNetwork.loss_ax.set_title("Loss Over Time")
        visNetwork.loss_ax.set_xlabel("Iteration")
        visNetwork.loss_ax.set_ylabel("Loss")
        visNetwork.error_text = visNetwork.loss_ax.text(0.5, 1.2, '', 
                                                        transform=visNetwork.loss_ax.transAxes,
                                                        ha='center', color='black', fontsize=12,
                                                        fontweight='bold')
    # Update each layer's weights
    for i, (layer, img) in enumerate(zip(network, visNetwork.images)):
        img.set_data(layer.weights)  # Update weight data
    
    # Update the loss plot
    visNetwork.loss_values.append(error)
    visNetwork.loss_line.set_data(range(len(visNetwork.loss_values)), visNetwork.loss_values)
    visNetwork.loss_ax.relim()  # Recompute the limits of the loss plot
    visNetwork.loss_ax.autoscale_view()  # Rescale to fit new data
    visNetwork.error_text.set_text(f"Current Error: {error:.4f}")

    
    # Redraw the canvas
    visNetwork.fig.canvas.draw()
    visNetwork.fig.canvas.flush_events()

    plt.tight_layout()
