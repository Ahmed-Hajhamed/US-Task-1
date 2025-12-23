import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import matplotlib.ticker as ticker
import gdown

# --- CONFIGURATION ---
# 1. SETUP FOR DOWNLOADING IMAGES
DRIVE_FOLDER_ID = "1SkHe3OIe_8JN5v0JgcUTJXWn0uYheEoC"
DOWNLOAD_DIR = "downloaded_data"  # Safer to use a simple name for the download target

# 2. SETUP FOR CSV DATA (Local)
# Matches your specific path: processing\results\resolution_results.csv
CSV_PATH = os.path.join("processing", "results", "resolution_results.csv")

# 3. SETUP FOR OUTPUT
OUTPUT_FOLDER = os.path.join("processing", "results_figures")

# Use non-interactive backend to prevent freezing during batch processing
plt.switch_backend('Agg')

# Ensure output directory exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Standardize plotting style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True,
    'lines.linewidth': 2
})

def db_scale(image, dynamic_range=60):
    """Normalize image to 0-1 and convert to dB with fixed dynamic range."""
    img_log = 20 * np.log10(image / np.max(image) + 1e-12)
    img_log = np.clip(img_log, -dynamic_range, 0)
    return img_log

def plot_bmode(data, filename_base):
    """Generates a labeled B-Mode image."""
    envelope = data['envelope']
    z_axis = data['z_axis'] * 1000  # mm
    x_axis = data['x_axis'] * 1000  # mm
    
    try:
        parts = filename_base.replace('.npz', '').split('_')
        freq_lbl = next(p for p in parts if 'MHz' in p)
        depth_lbl = next(p for p in parts if 'mm' in p)
    except StopIteration:
        freq_lbl = "Unknown Freq"
        depth_lbl = "Unknown Depth"

    plt.figure(figsize=(5, 6))
    bmode_db = db_scale(envelope)
    extent = [x_axis[0], x_axis[-1], z_axis[-1], z_axis[0]]
    
    plt.imshow(bmode_db, extent=extent, cmap='gray', aspect='equal', vmin=-60, vmax=0)
    plt.colorbar(label='Amplitude (dB)')
    
    plt.title(f"B-Mode: {freq_lbl}, {depth_lbl} Target")
    plt.xlabel("Lateral Distance (mm)")
    plt.ylabel("Axial Depth (mm)")
    
    save_path = os.path.join(OUTPUT_FOLDER, f"BMode_{freq_lbl}_{depth_lbl}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved B-Mode: {save_path}")

def plot_psf_profiles(data, filename_base):
    """Generates Axial and Lateral PSF profiles side-by-side."""
    axial_prof = data['axial_profile']
    lat_prof = data['lateral_profile']
    
    z_axis = data['z_axis'] * 1000
    x_axis = data['x_axis'] * 1000
    
    z_center = z_axis[np.argmax(axial_prof)]
    x_center = x_axis[np.argmax(lat_prof)]
    
    ax_db = db_scale(axial_prof, dynamic_range=40)
    lat_db = db_scale(lat_prof, dynamic_range=40)

    try:
        parts = filename_base.replace('.npz', '').split('_')
        freq_lbl = next(p for p in parts if 'MHz' in p)
        depth_lbl = next(p for p in parts if 'mm' in p)
    except:
        freq_lbl = "Unknown"
        depth_lbl = "Unknown"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Axial Plot
    ax1.plot(z_axis, ax_db, color='#007acc', label='Axial')
    ax1.axvline(x=z_center, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlim([z_center - 2, z_center + 2])
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Amplitude (dB)')
    ax1.set_title('Axial Profile')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Lateral Plot
    ax2.plot(x_axis, lat_db, color='#d62728', label='Lateral')
    ax2.axvline(x=x_center, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlim([x_center - 2, x_center + 2])
    ax2.set_xlabel('Lateral Position (mm)')
    ax2.set_title('Lateral Profile')
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    
    plt.suptitle(f"Point Spread Function: {freq_lbl} @ {depth_lbl}", y=1.02, fontweight='bold')
    
    save_path = os.path.join(OUTPUT_FOLDER, f"PSF_{freq_lbl}_{depth_lbl}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PSF Plot: {save_path}")

def plot_comparative_metrics():
    """Reads the local CSV file and plots FWHM trends."""
    
    if not os.path.exists(CSV_PATH):
        print(f"WARNING: CSV file not found at: {CSV_PATH}")
        print("Please ensure you have run the processing step (Person 2) locally.")
        return

    print(f"Loading summary data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if 'Frequency' not in df.columns:
        print("Error: CSV missing 'Frequency' column.")
        return
        
    freqs = df['Frequency'].unique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot Axial FWHM vs Depth
    for i, freq in enumerate(freqs):
        subset = df[df['Frequency'] == freq].sort_values(by="Depth (mm)")
        ax1.plot(subset['Depth (mm)'], subset['Axial FWHM (mm)'], 
                 marker=markers[i%3], color=colors[i%3], label=freq, linewidth=2)
    
    ax1.set_title("Axial Resolution vs Depth")
    ax1.set_xlabel("Depth (mm)")
    ax1.set_ylabel("FWHM (mm)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Lateral FWHM vs Depth
    for i, freq in enumerate(freqs):
        subset = df[df['Frequency'] == freq].sort_values(by="Depth (mm)")
        ax2.plot(subset['Depth (mm)'], subset['Lateral FWHM (mm)'], 
                 marker=markers[i%3], color=colors[i%3], label=freq, linewidth=2)
        
    ax2.set_title("Lateral Resolution vs Depth")
    ax2.set_xlabel("Depth (mm)")
    ax2.set_ylabel("FWHM (mm)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_FOLDER, "Comparison_Resolution_vs_Depth.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Comparison Plot: {save_path}")
    
    # Render Table
    fig_tbl, ax_tbl = plt.subplots(figsize=(8, 3))
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    table_data = [df.columns.values.tolist()] + df.values.tolist()
    table = ax_tbl.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    save_path_tbl = os.path.join(OUTPUT_FOLDER, "Resolution_Table.png")
    plt.savefig(save_path_tbl, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Table Image: {save_path_tbl}")

def download_data_if_needed():
    """Downloads folder from Google Drive if missing."""
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"'{DOWNLOAD_DIR}' not found. Downloading from Google Drive...")
        url = f'https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}'
        # Download recursively
        gdown.download_folder(url, output=DOWNLOAD_DIR, quiet=False, use_cookies=False)
        print("Download complete.")
    else:
        print(f"Found '{DOWNLOAD_DIR}'. Skipping download.")

def main():
    print("--- STARTING VISUALIZATION PIPELINE (Person 3) ---")
    
    # 1. DOWNLOAD IMAGES
    try:
        download_data_if_needed()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # 2. LOCATE .npz FILES (Recursive Search)
    # This ensures we find files even if gdown creates extra nested folders
    files = []
    for root, dirs, filenames in os.walk(DOWNLOAD_DIR):
        for filename in filenames:
            if filename.endswith(".npz"):
                files.append(os.path.join(root, filename))

    print(f"Found {len(files)} .npz files in {DOWNLOAD_DIR}.")

    if len(files) == 0:
        print("No .npz files found. Please check the downloaded folder.")
        # Proceeding to plot comparison anyway if CSV exists

    # 3. GENERATE B-MODE & PSF PLOTS
    for filepath in files:
        try:
            filename_base = os.path.basename(filepath)
            data = np.load(filepath)
            
            print(f"Processing {filename_base}...")
            plot_bmode(data, filename_base)
            plot_psf_profiles(data, filename_base)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    # 4. GENERATE AGGREGATE COMPARISONS (Reads from local CSV)
    print("Generating aggregate comparisons...")
    plot_comparative_metrics()
    
    print("\n--- VISUALIZATION COMPLETE ---")
    print(f"Results saved to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()