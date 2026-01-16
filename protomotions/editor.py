"""
editor.py
Launches an empty Isaac Lab window.
"""
import argparse

# --- STEP 1: LOAD SIMULATOR ---
# Use your repo's specific loader to get the paths right
from protomotions.utils.simulator_imports import import_simulator_before_torch

# Initialize the launcher
AppLauncher = import_simulator_before_torch("isaaclab")

# --- STEP 2: CONFIGURE THE WINDOW ---
parser = argparse.ArgumentParser(description="Isaac Lab Interactive Editor")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FORCE HEADLESS OFF so you see the GUI
args_cli.headless = False

# Launch the App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- STEP 3: MAIN LOOP ---
# We removed 'omni.isaac.core' imports because they were crashing.
# The GUI already has physics running, so we just keep the window open.

def main():
    print("\n" + "="*50)
    print("Interactive Mode Started!")
    print("Use the GUI to Import URDFs and create Materials.")
    print("="*50 + "\n")

    # Keep the window open
    while simulation_app.is_running():
        # This updates the graphics and physics every frame
        simulation_app.update()

    simulation_app.close()

if __name__ == "__main__":
    main()