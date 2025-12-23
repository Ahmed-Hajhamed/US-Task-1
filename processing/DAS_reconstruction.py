import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def reconstruct_image(rf_data, fs, pitch, c, z_start=0, z_end=0.08, width=0.04, dz=None, dx=None):
    """
    Performs Delay-and-Sum (SAFT) beamforming on raw pulse-echo data.
    """
    # 1. Define the Reconstruction Grid (The "Pixels")
    if dz is None: dz = c / fs / 2  # Step size close to axial resolution
    if dx is None: dx = pitch       # Step size matching element pitch
    
    z_axis = np.arange(z_start, z_end, dz)
    x_axis = np.arange(-width/2, width/2, dx)
    
    # Create the output image grid (zeros)
    image = np.zeros((len(z_axis), len(x_axis)))
    
    # 2. Geometry Setup
    num_samples, num_elements = rf_data.shape
    element_positions = np.arange(num_elements) * pitch
    element_positions -= np.mean(element_positions) # Center the array at 0
    
    # 3. The DAS Loop (Iterate over every pixel in the output image)
    # Note: Vectorized for speed, but logic is: find delay for every pixel -> sum
    
    print(f"Reconstructing image grid: {image.shape}...")
    
    # Grid matrices for vectorized calculation
    Z_grid, X_grid = np.meshgrid(z_axis, x_axis, indexing='ij')
    
    for i, x_elem in enumerate(element_positions):
        # Distance from this specific element to every pixel in the grid
        # Since it's pulse-echo on one element: Distance = 2 * (Element -> Pixel)
        dist_grid = 2 * np.sqrt((X_grid - x_elem)**2 + Z_grid**2)
        
        # Convert distance to time samples
        sample_grid = np.round(dist_grid / c * fs).astype(int)
        
        # Valid samples only (inside the recorded RF trace)
        valid_mask = (sample_grid >= 0) & (sample_grid < num_samples)
        
        # Add the signal from this element to the image
        # We take the signal at the calculated time index for each pixel
        # rf_data[:, i] is the trace for element i
        signal_contrib = np.zeros_like(image)
        signal_contrib[valid_mask] = rf_data[sample_grid[valid_mask], i]
        
        image += signal_contrib
        
    return image, z_axis, x_axis

# --- TEST BLOCK ---
# Load one file to test
try:
    data = np.load('rf_data\\rf_data_3.0MHz_40mm.npz') # Adjust path if needed
    rf_data = data['rf_data']
    fs = data['fs']
    pitch = data['pitch']
    c = data['c']
    
    # Run Reconstruction
    bf_image, z, x = reconstruct_image(rf_data, fs, pitch, c, z_start=0.03, z_end=0.05)
    
    # Quick Envelope to check
    envelope = np.abs(hilbert(bf_image, axis=0))
    
    # Visualization check
    plt.figure(figsize=(6,6))
    plt.imshow(20*np.log10(envelope/np.max(envelope) + 1e-12), 
               extent=[x[0]*1000, x[-1]*1000, z[-1]*1000, z[0]*1000], 
               cmap='gray', vmin=-60, vmax=0)
    plt.title("Reconstructed Point Scatterer (B-Mode)")
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Axial (mm)")
    plt.show()

except FileNotFoundError:
    print("File not found. Make sure the .npz files are in the directory.")