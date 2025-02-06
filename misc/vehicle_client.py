import asyncio
import websockets
import json
import numpy as np

async def send_vehicle_data():
    uri = "ws://127.0.0.1:8765/"
    async with websockets.connect(uri) as websocket:
        try:
            # Example telemetry payload
            payload = {
                "vehicle_id": "vehicle_123",
                "predictive_model_input": [298, 308, 1500, 45, 10, 10, 67500],
                "engine_condition_input": [600, 3.5, 12, 3, 80, 80],
            }
            
            # Send the payload
            await websocket.send(json.dumps(payload))
            print(f"Sent payload: {payload}")
            
            # Wait for and print the response
            response = await websocket.recv()
            print("Response from server:", json.loads(response))
            
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed unexpectedly")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(send_vehicle_data())