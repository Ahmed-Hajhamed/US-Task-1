import numpy as np
from scipy.signal import hilbert

class UltrasoundSimulator:
    def __init__(self, center_frequency=5e6, sampling_frequency=100e6, 
                 num_elements=128, element_width=0.0002, element_pitch=0.0003):
        
        self.fc = center_frequency
        self.fs = sampling_frequency
        self.num_elements = num_elements
        self.element_width = element_width
        self.pitch = element_pitch
        self.c = 1540  # Speed of sound in tissue (m/s)
        
        # transducer geometry
        self.element_positions = np.arange(num_elements) * self.pitch
        self.element_positions -= np.mean(self.element_positions)  # Center the array
        
    def generate_transmit_pulse(self, duration=3):
        pulse_duration = duration / self.fc
        time_pulse = np.arange(0, pulse_duration, 1/self.fs)
        
        # Gaussian envelope
        t_center = pulse_duration / 2
        sigma = pulse_duration / 6
        envelope = np.exp(-((time_pulse - t_center) ** 2) / (2 * sigma ** 2))
        
        # Modulated sinusoid
        pulse = envelope * np.sin(2 * np.pi * self.fc * time_pulse)
        
        return pulse, time_pulse
    
    def simulate_rf_data(self, scatterer_positions, max_depth=0.08):
        # Generate transmit pulse
        pulse, _ = self.generate_transmit_pulse()
        
        # Calculate time samples needed
        max_time = 2 * max_depth / self.c  # Round trip time
        num_samples = int(max_time * self.fs)
        time_axis = np.arange(num_samples) / self.fs
        
        rf_data = np.zeros((num_samples, self.num_elements))
        
        # Simulate each scatterer
        for idx, (x_scatter, z_scatter) in enumerate(scatterer_positions):
            for elem_idx, x_elem in enumerate(self.element_positions):
                # Calculate distance from element to scatterer and back
                distance = 2 * np.sqrt((x_scatter - x_elem)**2 + z_scatter**2)
                
                # Calculate time delay
                time_delay = distance / self.c
                
                # Calculate sample index
                sample_idx = int(time_delay * self.fs)
                
                if sample_idx + len(pulse) < num_samples:
                    rf_data[sample_idx:sample_idx+len(pulse), elem_idx] += pulse
    
        return rf_data, time_axis
    
    def save_rf_data(self, rf_data, time_axis, scatterer_positions, filename='rf_data.npz'):
        np.savez(filename,
                 rf_data=rf_data,
                 time_axis=time_axis,
                 scatterer_positions=scatterer_positions,
                 fc=self.fc,
                 fs=self.fs,
                 num_elements=self.num_elements,
                 pitch=self.pitch,
                 element_positions=self.element_positions,
                 c=self.c)
        

def main():
    frequencies = [3e6, 7.5e6]  # 3 MHz and 7.5 MHz
    depths = [0.020, 0.040, 0.060]  # 20 mm, 40 mm, 60 mm
    
    # Transducer parameters
    num_elements = 128
    element_pitch = 0.0003  # 0.3 mm
    sampling_frequency = 100e6  # 100 MHz
    
    for freq in frequencies:
        simulator = UltrasoundSimulator(
            center_frequency=freq,
            sampling_frequency=sampling_frequency,
            num_elements=num_elements,
            element_pitch=element_pitch
        )
        
        # Simulate for each depth
        for depth in depths:
            # Define point scatterer at center of array, at specified depth
            scatterer_positions = [(0.0, depth)]
            
            # Generate RF data
            rf_data, time_axis = simulator.simulate_rf_data(
                scatterer_positions,
                max_depth=0.08  # 80 mm max depth
            )
            
            # Save RF data
            filename = f"rf_data/rf_data_{freq/1e6:.1f}MHz_{depth*1000:.0f}mm.npz"
            simulator.save_rf_data(rf_data, time_axis, scatterer_positions, filename)

if __name__ == "__main__":
    main()
