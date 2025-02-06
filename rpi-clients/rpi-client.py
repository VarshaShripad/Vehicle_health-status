import websockets
import json
import asyncio
import numpy as np
import random
from time  import sleep

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
    energy=torque*rotational_speed
    tdiff=process_temp-air_temp

    # Target and failure type are random but must match the available labels
    target = random.randint(0, 1)  # Random target

    return [air_temp, process_temp, rotational_speed, torque, tool_wear,tdiff,energy]

# Function to create the test instance and send it over WebSocket
async def send_data():
    uri = "ws://13.51.56.144:8765"  # WebSocket server address
    
    try:
        async with websockets.connect(uri) as websocket:
            # Generate synthetic data for both models
            predictive_model_input = generate_predictive_maintenance_data()
            engine_condition_input = generate_engine_health_data()

            # Create the payload with the generated data
            test_instance = {
                "predictive_model_input": predictive_model_input,
                "engine_condition_input": engine_condition_input,
            }

            # Send the data as a JSON string to the WebSocket server
            await websocket.send(json.dumps(test_instance))

            # Wait for and receive the response from the server
            response = await websocket.recv()
            print(f"Received response: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the client
while True:
    asyncio.run(send_data())
    sleep(1)