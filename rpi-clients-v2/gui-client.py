import asyncio
import websockets
import json
import random
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

# Function to generate the test instance with vehicle_id
async def generate_test_instance(vehicle_id):
    predictive_model_input = generate_predictive_maintenance_data()
    engine_condition_input = generate_engine_health_data()

    test_instance = {
        "vehicle_id": vehicle_id,  # Include the vehicle_id
        "predictive_model_input": predictive_model_input,
        "engine_condition_input": engine_condition_input,
    }

    return json.dumps(test_instance)

# Function to send the data over WebSocket and update the GUI
async def send_data(ws, vehicle_id, label_status, label_failure, label_condition):
    while running:
        test_instance = await generate_test_instance(vehicle_id)
        try:
            # Send the data as a JSON string to the WebSocket server
            await ws.send(test_instance)
            print(f"Data sent successfully for vehicle {vehicle_id}")

            # Wait for and receive the response from the server
            response = await ws.recv()

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

        await asyncio.sleep(1)  # Wait for 1 second before sending the next data

# Function to run WebSocket client in the event loop
async def run_client(vehicle_id, label_status, label_failure, label_condition):
    uri = "ws://16.170.232.142:8765/vehicle"  # WebSocket server address
    try:
        async with websockets.connect(uri) as websocket:
            await send_data(websocket, vehicle_id, label_status, label_failure, label_condition)
    except Exception as e:
        label_status.config(text=f"Connection error: {e}")

# Function to handle continuous sending of data until stopped
def run_continuous_data_sending(vehicle_id, label_status, label_failure, label_condition):
    global running
    while running:
        # Start the event loop in a new thread
        asyncio.run(run_client(vehicle_id, label_status, label_failure, label_condition))

# Function to start or stop the system based on the button's state
def toggle_system_state(button, label_status, label_failure, label_condition, vehicle_id_entry, label_vehicle_id):
    global running

    # Disable the vehicle ID entry and hide the label after it's entered
    vehicle_id = vehicle_id_entry.get()
    if not vehicle_id:
        label_status.config(text="Please enter a valid Vehicle ID.")
        return

    vehicle_id_entry.config(state=tk.DISABLED)
    label_vehicle_id.pack_forget()  # Hide the vehicle ID label after submission

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
        threading.Thread(target=run_continuous_data_sending, args=(vehicle_id, label_status, label_failure, label_condition), daemon=True).start()

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

    # Create a label to enter the vehicle ID
    label_vehicle_id = tk.Label(frame, text="Enter Vehicle ID:", font=("Helvetica", 12), fg="#333", bg="#f5f5f5")
    label_vehicle_id.pack(pady=10)

    # Entry field for vehicle ID
    vehicle_id_entry = tk.Entry(frame, font=("Helvetica", 12), width=20)
    vehicle_id_entry.pack(pady=10)

    # Create a button to start/stop the system
    start_stop_button = tk.Button(frame, text="Start", font=("Helvetica", 14), bg="#4CAF50", fg="white", command=lambda: toggle_system_state(start_stop_button, label_status, label_failure, label_condition, vehicle_id_entry, label_vehicle_id), width=15, height=2)
    start_stop_button.pack(pady=20)

    # Run the Tkinter event loop
    root.mainloop()

# Run the tkinter window
if __name__ == "__main__":
    create_window()
