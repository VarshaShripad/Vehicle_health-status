import websockets
import json
import asyncio
import random
from time import sleep
import tkinter as tk
from tkinter import scrolledtext
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

    return [engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]

# Function to generate synthetic data from the predictive maintenance dataset
def generate_predictive_maintenance_data():
    air_temp = random.uniform(298, 300)  # Air temperature in Kelvin
    process_temp = random.uniform(308, 310)  # Process temperature in Kelvin
    rotational_speed = random.uniform(1400, 1600)  # Rotational speed in RPM
    torque = random.uniform(30, 60)  # Torque in Nm
    tool_wear = random.randint(0, 15)  # Tool wear in minutes
    energy = torque * rotational_speed
    tdiff = process_temp - air_temp

    return [air_temp, process_temp, rotational_speed, torque, tool_wear, tdiff, energy]

# Function to create the test instance and send it over WebSocket
async def send_data(websocket, test_instance, text_widget):
    try:
        # Send the data as a JSON string to the WebSocket server
        await websocket.send(json.dumps(test_instance))

        # Wait for and receive the response from the server
        response = await websocket.recv()

        # Update the text widget with the server response
        text_widget.insert(tk.END, f"Received response: {response}\n")
        text_widget.yview(tk.END)  # Scroll to the bottom

    except Exception as e:
        text_widget.insert(tk.END, f"An error occurred: {e}\n")
        text_widget.yview(tk.END)  # Scroll to the bottom

# Function to run WebSocket client in the event loop
async def run_client(test_instance, text_widget):
    uri = "ws://16.170.232.142:8765/vehicle"  # WebSocket server address
    try:
        async with websockets.connect(uri) as websocket:
            await send_data(websocket, test_instance, text_widget)
    except Exception as e:
        text_widget.insert(tk.END, f"Connection error: {e}\n")
        text_widget.yview(tk.END)  # Scroll to the bottom

# Function to handle continuous sending of data until stopped
def run_continuous_data_sending(text_widget):
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
        asyncio.run(run_client(test_instance, text_widget))

        # Wait for 1 second before generating new data
        sleep(1)

# Function to start or stop the system based on the button's state
def toggle_system_state(button, text_widget):
    global running

    if running:
        running = False
        button.config(text="Start")  # Change button text to "Start"
        text_widget.insert(tk.END, "System stopped.\n")
        text_widget.yview(tk.END)  # Scroll to the bottom
    else:
        running = True
        button.config(text="Stop")  # Change button text to "Stop"
        text_widget.insert(tk.END, "System started...\n")
        text_widget.yview(tk.END)  # Scroll to the bottom
        
        # Start the continuous data sending in a separate thread
        threading.Thread(target=run_continuous_data_sending, args=(text_widget,), daemon=True).start()

# Create the main tkinter window
def create_window():
    root = tk.Tk()
    root.title("WebSocket Client for Predictive Maintenance")

    # Create a button to start/stop the system
    start_stop_button = tk.Button(root, text="Start", command=lambda: toggle_system_state(start_stop_button, text_widget))
    start_stop_button.pack(pady=10)

    # Create a scrolled text widget to display responses from the server
    text_widget = scrolledtext.ScrolledText(root, width=80, height=20)
    text_widget.pack(padx=10, pady=10)

    # Run the Tkinter event loop
    root.mainloop()

# Run the tkinter window
if __name__ == "__main__":
    create_window()
