import numpy as np
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import time
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Verify versions and set Matplotlib backend
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
import os
os.environ['MPLBACKEND'] = 'Agg'

class TimeSourceSelector:
    def __init__(self):
        """Initialize with data buffer and model."""
        self.data_buffer = []
        self.model = None

    def fetch_real_data(self):
        """Fetch or simulate real-time data from Falcon switch and servers."""
        try:
            # Attempt to use ptp4l (Falcon switch)
            try:
                ptp_output = subprocess.run(['ptp4l', '-m'], capture_output=True, text=True, timeout=2).stdout
                latency = float(ptp_output.split('delay')[-1].split()[0]) if 'delay' in ptp_output else np.random.uniform(0, 50)
                jitter = float(ptp_output.split('jitter')[-1].split()[0]) if 'jitter' in ptp_output else np.random.uniform(0.04, 0.2)
            except (subprocess.SubprocessError, FileNotFoundError):
                latency = np.random.uniform(0, 50)  # ns
                jitter = np.random.uniform(0.04, 0.2)  # ns

            # Attempt to use ntpstat or ntpq (servers)
            try:
                ntp_output = subprocess.run(['ntpstat'], capture_output=True, text=True, timeout=2).stdout
                ntp_latency = float(ntp_output.split('time offset')[-1].split()[0]) if 'time offset' in ntp_output else np.random.uniform(0, 15)
            except (subprocess.SubprocessError, FileNotFoundError):
                try:
                    ntpq_output = subprocess.run(['ntpq', '-p'], capture_output=True, text=True, timeout=2).stdout
                    ntp_latency = float(ntpq_output.split('\n')[2].split()[7]) if len(ntpq_output.split('\n')) > 2 else np.random.uniform(0, 15)
                except (subprocess.SubprocessError, FileNotFoundError):
                    ntp_latency = np.random.uniform(0, 15)  # ms
                    print("Warning: NTP tools (ntpstat/ntpq) not found. Install 'ntp' package for real data.")

            # Simulated availability (placeholder for SNR)
            availability = np.random.uniform(25, 46)  # dB-Hz

            print(f"Fetched data: Latency={latency:.2f} ns, Jitter={jitter:.2f} ns, Availability={availability:.2f} dB-Hz, NTP Latency={ntp_latency:.2f} ms")
            return [latency, jitter, availability, ntp_latency]
        except Exception as e:
            print(f"Error fetching data: {e}. Using simulated values.")
            return [np.random.uniform(0, 50), np.random.uniform(0.04, 0.2), np.random.uniform(25, 46), np.random.uniform(0, 15)]

    def add_data(self):
        """Add current data to buffer."""
        data_point = self.fetch_real_data()
        self.data_buffer.append(data_point)

    def train_model(self):
        """Train Decision Tree on current data buffer."""
        if len(self.data_buffer) < 3:
            print("Insufficient data to train model.")
            return
        X = np.array(self.data_buffer)[:, :3]  # Features: latency, jitter, availability
        y = np.random.choice([0, 1, 2], len(self.data_buffer))  # Simulated labels (0=GNSS, 1=OCXO, 2=Atomic)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.2f}")

        new_data = np.array([[10, 0.05, 45], [30, 0.15, 30], [15, 0.08, 40]])
        new_predictions = self.model.predict(new_data)
        print(f"New predictions: {new_predictions} (0=GNSS, 1=OCXO, 2=Atomic)")

        importance = self.model.feature_importances_
        print(f"Feature importance: Latency: {importance[0]:.2f}, Jitter: {importance[1]:.2f}, Availability: {importance[2]:.2f}")

    def generate_diagram(self):
        """Generate diagram of data trends."""
        if len(self.data_buffer) == 0:
            print("No data to generate diagram.")
            return
        data = np.array(self.data_buffer)
        plt.figure(figsize=(12, 6))
        plt.plot(data[:, 0], label='Latency (ns)', color='#1E90FF')  # DodgerBlue
        plt.plot(data[:, 1], label='Jitter (ns)', color='#32CD32')  # LimeGreen
        plt.plot(data[:, 2], label='Availability (dB-Hz)', color='#FF4500')  # OrangeRed
        plt.title('Adaptive Time Source Selection Data Trends')
        plt.xlabel('Data Points')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('time_source_diagram.png')
        plt.close()
        print("Data presented in diagram form as time_source_diagram.png")

    def run(self):
        """Run the selector with continuous monitoring."""
        while True:
            self.add_data()
            if len(self.data_buffer) >= 10:  # Minimum for model
                self.train_model()
                self.generate_diagram()
            print("Continue? (y/n): ", end='', flush=True)
            sys.stdout.flush()
            user_input = input().strip().lower()
            if user_input != 'y':
                print("Exiting. Generating diagram with available data...")
                if len(self.data_buffer) > 0:
                    self.generate_diagram()
                break
            try:
                time.sleep(5)  # Reduced to 5 seconds for testing
            except KeyboardInterrupt:
                print("Interrupted by user. Generating diagram with available data...")
                if len(self.data_buffer) > 0:
                    self.generate_diagram()
                break

if __name__ == "__main__":
    selector = TimeSourceSelector()
    selector.run()
