import asyncio
import websockets
import json
import random
import numpy as np
import time

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

# Function to send the data over WebSocket
async def send_data(ws, vehicle_id):
    while True:
        test_instance = await generate_test_instance(vehicle_id)

        try:
            await ws.send(test_instance)
            print(f"Data sent successfully for vehicle {vehicle_id}")
        except Exception as e:
            print(f"An error occurred while sending data: {e}")
            break  # Break the loop on error (if needed, you can try to reconnect here)

        await asyncio.sleep(1)  # Wait for 1 second before sending the next data

# Explicitly handle ping/pong frames
async def pong_handler(ws, ping_data):
    # Respond with a pong when a ping is received
    await ws.pong(ping_data)

# Main WebSocket connection handler
async def connect_to_server():
    # Ask the user for the vehicle ID
    vehicle_id = input("Please enter the vehicle ID: ")

    uri = "ws://16.170.232.142:8765/vehicle"  # WebSocket server address

    backoff_delay = 5  # Initial reconnection delay

    while True:  # Loop indefinitely to handle reconnect attempts
        try:
            async with websockets.connect(uri, ping_interval=10, ping_timeout=30) as ws:  # Send pings every 10 seconds, wait for pong response for 30 seconds
                print(f"Connected to {uri}")

                # Listen for incoming messages and handle pongs explicitly
                async def receive_messages():
                    while True:
                        message = await ws.recv()
                        print(f"Received message: {message}")
                        # Handle pong (optional if server pings)
                        if message.startswith("ping"):
                            ping_data = message
                            await pong_handler(ws, ping_data)
                
                # Start sending data and receiving messages
                await asyncio.gather(send_data(ws, vehicle_id), receive_messages())
                break  # If connection is successful and data is sent, break the loop
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"Invalid WebSocket Status Code: {e}")
        except Exception as e:
            print(f"Error: {e}")

        print(f"Reconnecting in {backoff_delay} seconds...")
        await asyncio.sleep(backoff_delay)
        backoff_delay = min(backoff_delay * 2, 30)  # Exponential backoff, max delay 30 seconds

# Run the client
if __name__ == "__main__":
    asyncio.run(connect_to_server())  # Run the WebSocket client
