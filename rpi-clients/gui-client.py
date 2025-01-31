import websockets
import json
import asyncio
import random
from time import sleep
import tkinter as tk
from tkinter import messagebox
import threading

# Global flag to control whether the system is running
running = False

# Function to generate synthetic data from the engine health dataset
def generate_engine_health_data():
    engine_rpm = random.uniform(400, 900)  # Random RPM between 400 and 900
    lub_oil_pressure = random.uniform(2, 6)  # Random lubrication oil pressure
    fuel_pressure = random.uniform(6, 20)  # Random fuel pressure
    coolant_pressure = random.uniform(1, 5)  # Random coolant pressure
    lub_oil_temp = random.uniform(70, 90)  # Random lubrication oil temperature
    coolant_temp = random.uniform(70, 90)  # Random coolant temperature

    return engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp

# Function to generate synthetic data from the predictive maintenance dataset
def generate_predictive_maintenance_data():
    air_temp = random.uniform(298, 300)  # Air temperature in Kelvin
    process_temp = random.uniform(308, 310)  # Process temperature in Kelvin
    rotational_speed = random.uniform(1400, 1600)  # Rotational speed in RPM
    torque = random.uniform(30, 60)  # Torque in Nm
    tool_wear = random.randint(0, 15)  # Tool wear in minutes
    energy = torque * rotational_speed
    tdiff = process_temp - air_temp

    return air_temp, process_temp, rotational_speed, torque, tool_wear, tdiff, energy

# Function to create the test instance and send it over WebSocket
async def send_data(websocket, test_instance, label_status, label_failure, label_condition):
    try:
        # Send the data as a JSON string to the WebSocket server
        await websocket.send(json.dumps(test_instance))

        # Wait for and receive the response from the server
        response = await websocket.recv()

        # Update the label with the server response
        response_data = json.loads(response)
        predicted_failure = response_data.get("Predicted Failure Type", "N/A")
        predicted_condition = response_data.get("Predicted Engine Condition", "N/A")

        # Update the GUI with the results
        label_status.config(text="Data sent successfully!")
        label_failure.config(text=f"Predicted Failure: {predicted_failure}")
        label_condition.config(text=f"Predicted Condition: {predicted_condition}")

    except Exception as e:
        label_status.config(text=f"Error: {e}")

# Function to run WebSocket client in the event loop
async def run_client(test_instance, label_status, label_failure, label_condition):
    uri = "ws://13.51.56.144:8765"  # WebSocket server address
    try:
        async with websockets.connect(uri) as websocket:
            await send_data(websocket, test_instance, label_status, label_failure, label_condition)
    except Exception as e:
        label_status.config(text=f"Connection error: {e}")

# Function to handle continuous sending of data until stopped
def run_continuous_data_sending(label_status, label_failure, label_condition):
    global running
    while running:
        # Generate synthetic data for both models
        predictive_model_input = generate_predictive_maintenance_data()
        engine_condition_input = generate_engine_health_data()

        # Create the payload with the generated data
        test_instance = {
            "predictive_model_input": predictive_model_input,
            "engine_condition_input": engine_condition_input,
        }

        # Call the asyncio event loop to send the data
        asyncio.run(run_client(test_instance, label_status, label_failure, label_condition))

        # Wait for 1 second before generating new data
        sleep(1)

# Function to start or stop the system based on the button's state
def toggle_system_state(button, label_status, label_failure, label_condition):
    global running

    if running:
        running = False
        button.config(text="Start")  # Change button text to "Start"
        label_status.config(text="System stopped.")
        label_failure.config(text="Predicted Failure: N/A")
        label_condition.config(text="Predicted Condition: N/A")
    else:
        running = True
        button.config(text="Stop")  # Change button text to "Stop"
        label_status.config(text="System started...\nGenerating data...")

        # Start the continuous data sending in a separate thread
        threading.Thread(target=run_continuous_data_sending, args=(label_status, label_failure, label_condition), daemon=True).start()

# Create the main tkinter window
def create_window():
    global running

    # Initialize the Tkinter window
    root = tk.Tk()
    root.title("WebSocket Client Dashboard")

    # Set window size and position
    root.geometry("600x400")
    root.config(bg="#f5f5f5")

    # Create a frame to hold the content in the center
    frame = tk.Frame(root, bg="#f5f5f5")
    frame.pack(expand=True)

    # Create a label to show system status
    label_status = tk.Label(frame, text="System stopped.", font=("Helvetica", 16), fg="#333", bg="#f5f5f5", wraplength=500)
    label_status.pack(pady=20)

    # Create a label to show predicted failure type
    label_failure = tk.Label(frame, text="Predicted Failure: N/A", font=("Helvetica", 14), fg="#FF6347", bg="#f5f5f5")
    label_failure.pack(pady=10)

    # Create a label to show predicted engine condition
    label_condition = tk.Label(frame, text="Predicted Condition: N/A", font=("Helvetica", 14), fg="#4682B4", bg="#f5f5f5")
    label_condition.pack(pady=10)

    # Create a button to start/stop the system
    start_stop_button = tk.Button(frame, text="Start", font=("Helvetica", 14), bg="#4CAF50", fg="white", command=lambda: toggle_system_state(start_stop_button, label_status, label_failure, label_condition), width=15, height=2)
    start_stop_button.pack(pady=20)

    # Run the Tkinter event loop
    root.mainloop()

# Run the tkinter window
if __name__ == "__main__":
    create_window()
