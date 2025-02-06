import asyncio
import websockets
import json

# Function to handle receiving and displaying vehicle data
async def monitor_vehicles():
    uri = "ws://localhost:8765/monitor"  # WebSocket server URI for monitoring client
    async with websockets.connect(uri) as websocket:
        print("Connected to the monitoring server.")

        # Continuously listen for vehicle data updates from the server
        try:
            while True:
                # Receive the message (vehicle data)
                message = await websocket.recv()
                print(message)
                vehicle_data = json.loads(message)

                # Display the vehicle data (you can customize this display logic)
                print(f"Vehicle ID: {vehicle_data['vehicle_id']}")
                print(f"Predicted Failure Type: {vehicle_data['Predicted Failure Type']}")
                print(f"Predicted Engine Condition: {vehicle_data['Predicted Engine Condition']}")
                print("-" * 50)

        except websockets.exceptions.ConnectionClosed:
            print("Disconnected from the server.")
        except Exception as e:
            print(f"Error: {e}")

# Run the monitoring client
if __name__ == "__main__":
    asyncio.run(monitor_vehicles())
