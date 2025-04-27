import os
import subprocess
import sys
import time

def run_script(script_path, script_name):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print(f"\n✅ {script_name} completed successfully.\n")
            return True
        else:
            print(f"\n❌ {script_name} failed with return code {result.returncode}.\n")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} failed with error: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}\n")
        return False

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Define scripts to run in order
    scripts = [
        ("models/visuals.py", "Visualization Script"),
        ("models/linear_regresion_analysis.py", "Linear Regression Analysis"),
        ("models/freq.py", "Frequency Analysis"),
        ("models/genre_share.py", "Genre Share Analysis"),
        ("models/KNN_movies_500.py", "KNN Analysis"),
        ("models/random_forest.py", "Random Forest Analysis")
    ]
    
    # Track overall success
    all_successful = True
    start_time = time.time()
    
    # Run each script in sequence
    for script_path, script_name in scripts:
        if not os.path.exists(script_path):
            print(f"\n⚠️ Warning: {script_path} does not exist. Skipping.")
            continue
            
        success = run_script(script_path, script_name)
        all_successful = all_successful and success
        
        # Add a small delay between scripts
        time.sleep(1)
    
    # Report overall status
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    if all_successful:
        print(f"✅ All scripts completed successfully in {elapsed_time:.2f} seconds!")
    else:
        print(f"⚠️ Some scripts encountered errors. Please check the output above.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    print("\n🚀 Starting movie data analysis pipeline...\n")
    main()