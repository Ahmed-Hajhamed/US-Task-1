import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QSplitter, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import hilbert
import pandas as pd


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'processing'))

from ultrasound_simulation import UltrasoundSimulator
from analysis_pipeline import reconstruct_image, calculate_fwhm


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt6"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class PSFSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrasound PSF Simulator - Task 1")
        self.setGeometry(100, 100, 1400, 900)
        
        # Simulation parameters
        self.frequency = 5.0e6  # Hz
        self.depths = [0.020, 0.040, 0.060]  # meters
        self.num_elements = 128
        self.element_pitch = 0.0003  # meters
        self.sampling_frequency = 100e6  # Hz
        self.c = 1540  # m/s
        
        # Storage for results
        self.results_data = []
        
        # Storage for depth-specific results (all 3 depths)
        self.depth_results = [None, None, None]  # Store (envelope, z_axis, x_axis) for each depth
        self.current_depth_index = 0  # Which depth is currently being viewed (0, 1, or 2)
        
        # Current visualization data
        self.current_rf_data = None
        self.current_envelope = None
        self.current_z_axis = None
        self.current_x_axis = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel: Controls
        control_panel = self.create_control_panel()
        
        # Right panel: Visualization
        viz_panel = self.create_visualization_panel()
        
        # Add to splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(viz_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """Create the left control panel with sliders"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Simulation Parameters")
        title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Frequency control
        freq_group = QGroupBox("Center Frequency")
        freq_layout = QVBoxLayout()
        
        self.freq_label = QLabel("Frequency: 5.0 MHz")
        freq_layout.addWidget(self.freq_label)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(30)  # 3.0 MHz
        self.freq_slider.setMaximum(75)  # 7.5 MHz
        self.freq_slider.setValue(50)    # 5.0 MHz
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.setTickInterval(5)
        self.freq_slider.setToolTip("Adjust the center frequency of the ultrasound transducer (3.0 - 7.5 MHz)")
        self.freq_slider.valueChanged.connect(self.update_frequency)
        freq_layout.addWidget(self.freq_slider)
        
        # Add preset buttons
        preset_layout = QHBoxLayout()
        btn_3mhz = QPushButton("3 MHz")
        btn_3mhz.setToolTip("Set frequency to 3 MHz (lower resolution, deeper penetration)")
        btn_3mhz.clicked.connect(lambda: self.freq_slider.setValue(30))
        btn_75mhz = QPushButton("7.5 MHz")
        btn_75mhz.setToolTip("Set frequency to 7.5 MHz (higher resolution, limited penetration)")
        btn_75mhz.clicked.connect(lambda: self.freq_slider.setValue(75))
        preset_layout.addWidget(btn_3mhz)
        preset_layout.addWidget(btn_75mhz)
        freq_layout.addLayout(preset_layout)
        
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        # Depth controls for 3 scatterers
        depth_group = QGroupBox("Point Scatterer Depths")
        depth_layout = QVBoxLayout()
        
        self.depth_sliders = []
        self.depth_labels = []
        
        for i in range(3):
            depth_label = QLabel(f"Depth {i+1}: {self.depths[i]*1000:.1f} mm")
            self.depth_labels.append(depth_label)
            depth_layout.addWidget(depth_label)
            
            depth_slider = QSlider(Qt.Orientation.Horizontal)
            depth_slider.setMinimum(10)   # 10 mm
            depth_slider.setMaximum(80)   # 80 mm
            depth_slider.setValue(int(self.depths[i] * 1000))
            depth_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            depth_slider.setTickInterval(10)
            depth_slider.setToolTip(f"Adjust the depth of point scatterer {i+1} (10-80 mm)")
            depth_slider.valueChanged.connect(lambda value, idx=i: self.update_depth(idx, value))
            self.depth_sliders.append(depth_slider)
            depth_layout.addWidget(depth_slider)
        
        depth_group.setLayout(depth_layout)
        layout.addWidget(depth_group)
        
        # Array configuration
        array_group = QGroupBox("Array Configuration")
        array_layout = QGridLayout()
        
        array_layout.addWidget(QLabel("Number of Elements:"), 0, 0)
        self.elements_label = QLabel(str(self.num_elements))
        array_layout.addWidget(self.elements_label, 0, 1)
        
        self.elements_slider = QSlider(Qt.Orientation.Horizontal)
        self.elements_slider.setMinimum(64)
        self.elements_slider.setMaximum(256)
        self.elements_slider.setValue(128)
        self.elements_slider.setTickInterval(32)
        self.elements_slider.setToolTip("Number of transducer elements (more = better lateral resolution but slower)")
        self.elements_slider.valueChanged.connect(self.update_elements)
        array_layout.addWidget(self.elements_slider, 1, 0, 1, 2)
        
        array_layout.addWidget(QLabel("Element Pitch (mm):"), 2, 0)
        self.pitch_label = QLabel(f"{self.element_pitch*1000:.2f}")
        array_layout.addWidget(self.pitch_label, 2, 1)
        
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setMinimum(15)   # 0.15 mm
        self.pitch_slider.setMaximum(50)   # 0.50 mm
        self.pitch_slider.setValue(30)     # 0.30 mm
        self.pitch_slider.setTickInterval(5)
        self.pitch_slider.setToolTip("Distance between adjacent transducer elements (0.15-0.50 mm)")
        self.pitch_slider.valueChanged.connect(self.update_pitch)
        array_layout.addWidget(self.pitch_slider, 3, 0, 1, 2)
        
        array_group.setLayout(array_layout)
        layout.addWidget(array_group)
        
        # Simulation button
        self.sim_button = QPushButton("Run Simulation")
        self.sim_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")
        self.sim_button.setToolTip("Run ultrasound simulation with current parameters (takes 5-15 seconds)")
        self.sim_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.sim_button)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create the right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: B-Mode Image
        bmode_tab = QWidget()
        bmode_layout = QVBoxLayout()
        
        # Add depth selection buttons
        depth_select_group = QGroupBox("Select Depth to View")
        depth_select_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        depth_select_layout = QHBoxLayout()
        
        self.depth_buttons = []
        for i in range(3):
            btn = QPushButton(f"Depth {i+1}")
            btn.setCheckable(True)
            btn.setToolTip(f"View results for depth {i+1}")
            btn.clicked.connect(lambda checked, idx=i: self.switch_depth_view(idx))
            self.depth_buttons.append(btn)
            depth_select_layout.addWidget(btn)
        
        # Set first button as checked by default
        self.depth_buttons[0].setChecked(True)
        
        depth_select_group.setLayout(depth_select_layout)
        bmode_layout.addWidget(depth_select_group)
        
        self.bmode_canvas = MplCanvas(self, width=8, height=8, dpi=100)
        bmode_layout.addWidget(self.bmode_canvas)
        bmode_tab.setLayout(bmode_layout)
        tabs.addTab(bmode_tab, "B-Mode Image")
        
        # Tab 2: PSF Profiles
        psf_tab = QWidget()
        psf_layout = QVBoxLayout()
        
        # Add depth selection buttons at top (matching B-Mode tab)
        psf_depth_select = QGroupBox("Select Depth to View")
        psf_depth_select.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        psf_depth_layout = QHBoxLayout()
        
        # Create duplicate buttons that sync with B-Mode tab buttons
        self.psf_depth_buttons = []
        for i in range(3):
            btn = QPushButton(f"Depth {i+1}")
            btn.setCheckable(True)
            btn.setToolTip(f"View results for depth {i+1}")
            btn.clicked.connect(lambda checked, idx=i: self.switch_depth_view(idx))
            self.psf_depth_buttons.append(btn)
            psf_depth_layout.addWidget(btn)
        
        # Sync with first button checked
        self.psf_depth_buttons[0].setChecked(True)
        
        psf_depth_select.setLayout(psf_depth_layout)
        psf_layout.addWidget(psf_depth_select)
        
        # Add PSF canvas below buttons - larger size for better visibility
        self.psf_canvas = MplCanvas(self, width=12, height=8, dpi=100)
        psf_layout.addWidget(self.psf_canvas)
        psf_tab.setLayout(psf_layout)
        tabs.addTab(psf_tab, "PSF Profiles")
        
        # Tab 3: Resolution Table & Comparison
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        
        # Resolution table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Frequency (MHz)", "Depth (mm)", "Axial FWHM (mm)", "Lateral FWHM (mm)"
        ])
        results_layout.addWidget(self.results_table)
        
        results_tab.setLayout(results_layout)
        tabs.addTab(results_tab, "Results & Analysis")
        
        layout.addWidget(tabs)
        
        return panel
    
    def update_frequency(self, value):
        """Update frequency from slider"""
        self.frequency = value / 10.0 * 1e6  # Convert to Hz
        self.freq_label.setText(f"Frequency: {value/10.0:.1f} MHz")
    
    def update_depth(self, index, value):
        """Update depth for a specific scatterer"""
        self.depths[index] = value / 1000.0  # Convert mm to m
        self.depth_labels[index].setText(f"Depth {index+1}: {value:.1f} mm")
    
    def update_elements(self, value):
        """Update number of elements"""
        self.num_elements = value
        self.elements_label.setText(str(value))
    
    def update_pitch(self, value):
        """Update element pitch"""
        self.element_pitch = value / 100.0 / 1000.0  # Convert to m
        self.pitch_label.setText(f"{value/100.0:.2f}")
    
    def run_simulation(self):
        """Main simulation execution - processes all 3 depths"""
        self.status_label.setText("Running simulation...")
        self.sim_button.setEnabled(False)
        QApplication.processEvents()  # Update UI
        
        try:
            # Create simulator with current parameters
            simulator = UltrasoundSimulator(
                center_frequency=self.frequency,
                sampling_frequency=self.sampling_frequency,
                num_elements=self.num_elements,
                element_width=self.element_pitch * 0.8,
                element_pitch=self.element_pitch
            )
            
            # Process ALL three depths - prioritize processed data
            for depth_idx, depth in enumerate(self.depths):
                freq_mhz = self.frequency / 1e6
                depth_mm = depth * 1000
                
                # Priority 1: Check for fully processed data
                processed_filename = f"processed_rf_data_{freq_mhz:.1f}MHz_{depth_mm:.0f}mm.npz"
                
                if os.path.exists(processed_filename):
                    # Load fully processed data - no reconstruction needed
                    print(f"Loading processed data: {processed_filename}")
                    data = np.load(processed_filename)
                    envelope = data['envelope']
                    z_axis = data['z_axis']
                    x_axis = data['x_axis']
                    
                else:
                    # Priority 2: Check for raw RF data (fast)
                    rf_filename = f"rf_data/rf_data_{freq_mhz:.1f}MHz_{depth_mm:.0f}mm.npz"
                    
                    if os.path.exists(rf_filename):
                        # Load pre-generated RF data and reconstruct
                        print(f"Loading RF data: {rf_filename}")
                        rf_loaded = np.load(rf_filename)
                        rf_data = rf_loaded['rf_data']
                        fs = float(rf_loaded['fs'])
                        pitch = float(rf_loaded['pitch'])
                        c = float(rf_loaded['c'])
                    else:
                        # Priority 3: Simulate from scratch (slow)
                        print(f"Simulating: {freq_mhz:.1f} MHz @ {depth_mm:.0f} mm")
                        scatterer_positions = [(0.0, depth)]
                        rf_data, time_axis = simulator.simulate_rf_data(
                            scatterer_positions,
                            max_depth=0.10
                        )
                        fs = simulator.fs
                        pitch = simulator.pitch
                        c = simulator.c
                    
                    # Reconstruct image
                    z_start = max(0, depth - 0.015)
                    z_end = depth + 0.015
                    
                    bf_image, z_axis, x_axis = reconstruct_image(
                        rf_data, 
                        fs, 
                        pitch, 
                        c,
                        z_start=z_start,
                        z_end=z_end,
                        width=0.04,
                        dx=pitch / 10
                    )
                    
                    # Get envelope
                    envelope = np.abs(hilbert(bf_image, axis=0))
                
                # Store results for this depth
                self.depth_results[depth_idx] = {
                    'envelope': envelope,
                    'z_axis': z_axis,
                    'x_axis': x_axis,
                    'depth': depth
                }
            
            # Set current view to first depth
            self.current_depth_index = 0
            self.load_depth_data(0)
            
            # Update visualizations for first depth
            self.plot_bmode()
            self.plot_psf_profiles()
            self.analyze_all_depths()
            
            self.status_label.setText("Simulation complete! Use buttons to view different depths.")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.sim_button.setEnabled(True)
    
    def switch_depth_view(self, depth_index):
        """Switch to viewing a different depth"""
        if self.depth_results[depth_index] is None:
            self.status_label.setText("Please run simulation first!")
            return
        
        # Update which button is checked (both B-Mode and PSF tabs)
        for i, btn in enumerate(self.depth_buttons):
            btn.setChecked(i == depth_index)
        for i, btn in enumerate(self.psf_depth_buttons):
            btn.setChecked(i == depth_index)
        
        # Update current depth index
        self.current_depth_index = depth_index
        
        # Load data for this depth
        self.load_depth_data(depth_index)
        
        # Update visualizations
        self.plot_bmode()
        self.plot_psf_profiles()
        
        self.status_label.setText(f"Viewing Depth {depth_index + 1}: {self.depths[depth_index]*1000:.1f} mm")
    
    def load_depth_data(self, depth_index):
        """Load envelope and axis data for a specific depth"""
        data = self.depth_results[depth_index]
        self.current_envelope = data['envelope']
        self.current_z_axis = data['z_axis']
        self.current_x_axis = data['x_axis']
    
    def plot_bmode(self):
        """Plot B-mode image - matches original visualization style"""
        if self.current_envelope is None:
            return
        
        # Clear figure completely to avoid duplicate colorbars
        self.bmode_canvas.fig.clear()
        self.bmode_canvas.axes = self.bmode_canvas.fig.add_subplot(111)
        
        envelope = self.current_envelope
        z_axis = self.current_z_axis * 1000  # mm
        x_axis = self.current_x_axis * 1000  # mm
        
        # Convert to dB 
        bmode_db = 20 * np.log10(envelope / np.max(envelope) + 1e-12)
        bmode_db = np.clip(bmode_db, -40, 0)
        
        extent = [x_axis[0], x_axis[-1], z_axis[-1], z_axis[0]]
        
        im = self.bmode_canvas.axes.imshow(
            bmode_db,
            extent=extent,
            cmap='gray',
            aspect='equal',
            vmin=-40,
            vmax=0
        )
        
        self.bmode_canvas.axes.set_title(
            f"B-Mode: {self.frequency/1e6:.1f} MHz, {self.depths[self.current_depth_index]*1000:.0f}mm Target",
            fontsize=12,
            fontweight='bold'
        )
        self.bmode_canvas.axes.set_xlabel("Lateral Distance (mm)")
        self.bmode_canvas.axes.set_ylabel("Axial Depth (mm)")
        
        # Add colorbar
        self.bmode_canvas.fig.colorbar(im, ax=self.bmode_canvas.axes, label='Amplitude (dB)')
        
        self.bmode_canvas.fig.tight_layout()
        self.bmode_canvas.draw()
    
    def plot_psf_profiles(self):
        """Plot PSF profiles - matches original visualization style"""
        if self.current_envelope is None:
            return
        
        envelope = self.current_envelope
        z_axis = self.current_z_axis * 1000  # mm
        x_axis = self.current_x_axis * 1000  # mm
        
        # Find peak
        max_idx = np.unravel_index(np.argmax(envelope), envelope.shape)
        z_peak_idx, x_peak_idx = max_idx
        
        # Extract profiles
        axial_profile = envelope[:, x_peak_idx]
        lateral_profile = envelope[z_peak_idx, :]
        
        # Find centers
        z_center = z_axis[z_peak_idx]
        x_center = x_axis[x_peak_idx]
        
        # Convert to dB 
        axial_db = 20 * np.log10(axial_profile / np.max(axial_profile) + 1e-12)
        axial_db = np.clip(axial_db, -40, 0)
        lateral_db = 20 * np.log10(lateral_profile / np.max(lateral_profile) + 1e-12)
        lateral_db = np.clip(lateral_db, -40, 0)
        
        # Calculate FWHM
        fwhm_axial = calculate_fwhm(axial_profile, z_axis)
        fwhm_lateral = calculate_fwhm(lateral_profile, x_axis)
        
        # Normalize profiles for validation view
        axial_norm = axial_profile / np.max(axial_profile)
        lateral_norm = lateral_profile / np.max(lateral_profile)
        
        # Clear and create subplots (2x2 grid for validation)
        self.psf_canvas.fig.clear()
        ax1 = self.psf_canvas.fig.add_subplot(221)  # Top-left
        ax2 = self.psf_canvas.fig.add_subplot(222)  # Top-right  
        ax3 = self.psf_canvas.fig.add_subplot(223)  # Bottom-left
        ax4 = self.psf_canvas.fig.add_subplot(224)  # Bottom-right
        
        # Top row: dB profiles
        ax1.plot(z_axis, axial_db, color='#007acc', linewidth=2)
        ax1.axvline(x=z_center, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlim([z_center - 2, z_center + 2])
        ax1.set_xlabel('Depth (mm)')
        ax1.set_ylabel('Amplitude (dB)')
        ax1.set_title('Axial Profile (dB)')
        ax1.grid(True, which='both', linestyle='--', alpha=0.6)
        
        ax2.plot(x_axis, lateral_db, color='#d62728', linewidth=2)
        ax2.axvline(x=x_center, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlim([x_center - 2, x_center + 2])
        ax2.set_xlabel('Lateral Position (mm)')
        ax2.set_ylabel('Amplitude (dB)')
        ax2.set_title('Lateral Profile (dB)')
        ax2.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Bottom row: Normalized with FWHM markers
        ax3.plot(z_axis, axial_norm, color='#007acc', linewidth=2, label='Axial')
        ax3.axhline(0.5, color='r', linestyle='--', label='Half-Max', linewidth=1.5)
        ax3.axvline(z_center - fwhm_axial/2, color='g', linestyle=':', label='FWHM', linewidth=1.5)
        ax3.axvline(z_center + fwhm_axial/2, color='g', linestyle=':', linewidth=1.5)
        ax3.set_xlim([z_center - 2, z_center + 2])
        ax3.set_xlabel('Depth (mm)')
        ax3.set_ylabel('Normalized Amplitude')
        ax3.set_title(f'Axial FWHM = {fwhm_axial:.3f} mm')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.set_ylim([0, 1.1])
        
        ax4.plot(x_axis, lateral_norm, color='#d62728', linewidth=2, label='Lateral')
        ax4.axhline(0.5, color='r', linestyle='--', label='Half-Max', linewidth=1.5)
        ax4.axvline(x_center - fwhm_lateral/2, color='g', linestyle=':', label='FWHM', linewidth=1.5)
        ax4.axvline(x_center + fwhm_lateral/2, color='g', linestyle=':', linewidth=1.5)
        ax4.set_xlim([x_center - 2, x_center + 2])
        ax4.set_xlabel('Lateral Position (mm)')
        ax4.set_ylabel('Normalized Amplitude')
        ax4.set_title(f'Lateral FWHM = {fwhm_lateral:.3f} mm')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc='upper right')
        ax4.set_ylim([0, 1.1])
        
        self.psf_canvas.fig.suptitle(
            f'PSF & FWHM Validation: {self.frequency/1e6:.1f} MHz @ {self.depths[self.current_depth_index]*1000:.0f}mm',
            y=0.98,
            fontweight='bold'
        )
        
        self.psf_canvas.fig.tight_layout()
        self.psf_canvas.draw()
    
    def analyze_all_depths(self):
        """Analyze PSF at all depths using precomputed results"""
        current_results = []
        
        # Use already computed depth_results instead of re-simulating
        for depth_idx, depth_data in enumerate(self.depth_results):
            if depth_data is None:
                continue
                
            envelope = depth_data['envelope']
            z_axis = depth_data['z_axis']
            x_axis = depth_data['x_axis']
            depth = depth_data['depth']
            
            # Find peak and measure FWHM
            max_idx = np.unravel_index(np.argmax(envelope), envelope.shape)
            z_peak_idx, x_peak_idx = max_idx
            
            axial_profile = envelope[:, x_peak_idx]
            lateral_profile = envelope[z_peak_idx, :]
            
            fwhm_axial = calculate_fwhm(axial_profile, z_axis * 1000)
            fwhm_lateral = calculate_fwhm(lateral_profile, x_axis * 1000)
            
            current_results.append({
                'Frequency': self.frequency / 1e6,
                'Depth': depth * 1000,
                'Axial FWHM': fwhm_axial,
                'Lateral FWHM': fwhm_lateral
            })
        
        # Add to cumulative results
        self.results_data.extend(current_results)
        
        
        # Update table
        self.update_results_table()
    
    def update_results_table(self):
        """Update the results table widget"""
        self.results_table.setRowCount(len(self.results_data))
        
        for i, result in enumerate(self.results_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{result['Frequency']:.1f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result['Depth']:.1f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result['Axial FWHM']:.3f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result['Lateral FWHM']:.3f}"))
        
        self.results_table.resizeColumnsToContents()
    

    



def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = PSFSimulatorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
