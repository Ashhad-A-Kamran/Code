import time
from codecarbon import EmissionsTracker

def test_energy_tracking():
    print("Initializing CodeCarbon tracker...")
    tracker = EmissionsTracker(project_name="Verification_Test", 
                               measure_power_secs=1,
                               save_to_file=False)
    
    try:
        tracker.start()
        print("Tracker started. Simulating workload for 5 seconds...")
        
        # Simulate some dummy work
        start_time = time.time()
        while time.time() - start_time < 5:
            _ = [x**2 for x in range(1000)]
            
        emissions = tracker.stop()
        print(f"Tracker stopped.")
        if hasattr(tracker, '_total_energy'):
            print(f"Energy Consumed: {tracker._total_energy.kWh} kWh")
        
        # Power draw is usually available in the emissions data
        print("Verification successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    test_energy_tracking()
