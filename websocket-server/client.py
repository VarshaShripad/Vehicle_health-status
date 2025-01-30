import websockets
import json
import asyncio

async def send_data():
    uri = "ws://localhost:8765/"
    try:
        async with websockets.connect(uri) as websocket:
            test_instance = {
                "predictive_model_input": [300, 350, 1500, 50, 200, 50, 75_000],
                "engine_condition_input": [700, 2.49359182, 11.79092738, 3.178980794, 84.14416293, 81.6321865],
            }

            await websocket.send(json.dumps(test_instance))  # Send data as JSON
            response = await websocket.recv()
            print(f"Received response: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

asyncio.run(send_data())
