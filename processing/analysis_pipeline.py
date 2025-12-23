import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import glob
import pandas as pd
import os

# --- 1. CORE FUNCTIONS ---

def reconstruct_image(rf_data, fs, pitch, c, z_start=0, z_end=0.08, width=0.04, dz=None, dx=None):
    """Reconstructs the B-mode image from RF data (same as before)."""
    if dz is None: dz = c / fs / 2
    if dx is None: dx = pitch
    
    z_axis = np.arange(z_start, z_end, dz)
    x_axis = np.arange(-width/2, width/2, dx)
    
    # Grid setup for vectorized calculation
    Z_grid, X_grid = np.meshgrid(z_axis, x_axis, indexing='ij')
    image = np.zeros_like(Z_grid)
    
    num_samples, num_elements = rf_data.shape
    element_positions = np.arange(num_elements) * pitch
    element_positions -= np.mean(element_positions) 
    
    print(f"  -> Reconstructing {image.shape} grid...")
    
    for i, x_elem in enumerate(element_positions):
        dist_grid = 2 * np.sqrt((X_grid - x_elem)**2 + Z_grid**2)
        sample_grid = np.round(dist_grid / c * fs).astype(int)
        
        valid_mask = (sample_grid >= 0) & (sample_grid < num_samples)
        
        signal_contrib = np.zeros_like(image)
        signal_contrib[valid_mask] = rf_data[sample_grid[valid_mask], i]
        image += signal_contrib
        
    return image, z_axis, x_axis

def calculate_fwhm(profile, axis_values):
    """
    Calculates Full-Width-at-Half-Maximum (FWHM) with sub-pixel interpolation.
    Input:
        profile: 1D array of signal intensity (linear scale)
        axis_values: 1D array of physical distances (mm) corresponding to profile
    """
    # 1. Normalize
    profile = np.abs(profile)
    peak_val = np.max(profile)
    half_max = peak_val / 2.0
    
    # 2. Find indices above half max
    # We look for crossings where signal goes from < half_max to > half_max
    above_threshold = np.where(profile >= half_max)[0]
    
    if len(above_threshold) < 2:
        return 0.0 # Error or too narrow
    
    left_idx = above_threshold[0]
    right_idx = above_threshold[-1]
    
    # 3. Linear Interpolation for precision
    # Left crossing
    y1 = profile[left_idx - 1]
    y2 = profile[left_idx]
    x1 = axis_values[left_idx - 1]
    x2 = axis_values[left_idx]
    x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    
    # Right crossing
    if right_idx + 1 < len(profile):
        y1 = profile[right_idx]
        y2 = profile[right_idx + 1]
        x1 = axis_values[right_idx]
        x2 = axis_values[right_idx + 1]
        x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        x_right = axis_values[right_idx]

    return abs(x_right - x_left)

# --- 2. MAIN PIPELINE ---

def process_all_files(input_folder="rf_data"):
    files = glob.glob(os.path.join(input_folder, "*.npz"))
    results = []
    
    print(f"Found {len(files)} files to process.")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        
        # Load Data
        data = np.load(filepath)
        rf_data = data['rf_data']
        fs = float(data['fs'])
        pitch = float(data['pitch'])
        c = float(data['c'])
        fc_label = f"{data['fc']/1e6:.1f} MHz"
        
        depth_str = filename.split('_')[3].replace('mm.npz','')
        target_depth_m = float(depth_str) / 1000.0
        
        # Define Reconstruction Window
        z_start = max(0, target_depth_m - 0.015) 
        z_end = target_depth_m + 0.015           
        
        # --- THE FIX IS HERE ---
        # We explicitly set dx to be much smaller than the pitch (High Resolution Grid)
        # pitch / 10 ensures we have plenty of pixels across the main lobe
        high_res_dx = pitch / 10  
        
        bf_image, z_axis, x_axis = reconstruct_image(rf_data, fs, pitch, c, 
                                                     z_start=z_start, z_end=z_end, width=0.04,
                                                     dx=high_res_dx) # <--- Using finer grid
        
        # Get Envelope
        envelope = np.abs(hilbert(bf_image, axis=0))
        
        # Find Peak
        max_idx = np.unravel_index(np.argmax(envelope), envelope.shape)
        z_peak_idx, x_peak_idx = max_idx
        
        # Extract Profiles
        axial_profile = envelope[:, x_peak_idx]
        lateral_profile = envelope[z_peak_idx, :]
        
        # Measure FWHM (in mm)
        fwhm_axial = calculate_fwhm(axial_profile, z_axis * 1000)
        fwhm_lateral = calculate_fwhm(lateral_profile, x_axis * 1000)
        
        print(f"  -> Found Point at Z={z_axis[z_peak_idx]*1000:.2f}mm")
        print(f"  -> Axial FWHM: {fwhm_axial:.3f} mm")
        print(f"  -> Lateral FWHM: {fwhm_lateral:.3f} mm")
        
        # Save results
        output_filename = f"processed_{filename}"
        np.savez(output_filename, 
                 envelope=envelope, 
                 z_axis=z_axis, 
                 x_axis=x_axis,
                 axial_profile=axial_profile,
                 lateral_profile=lateral_profile,
                 fwhm_axial=fwhm_axial,
                 fwhm_lateral=fwhm_lateral)
        
        results.append({
            "Frequency": fc_label,
            "Depth (mm)": int(depth_str),
            "Axial FWHM (mm)": round(fwhm_axial, 3),
            "Lateral FWHM (mm)": round(fwhm_lateral, 3)
        })

    # Summary
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Frequency", "Depth (mm)"])
    print("\n--- FINAL RESOLUTION TABLE ---")
    print(df)
    df.to_csv("resolution_results.csv", index=False)

if __name__ == "__main__":
    # Ensure your 'rf_data' folder exists and has the files
    process_all_files()