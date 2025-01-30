from models import PredictiveMaintenanceModel, EngineConditionPredictor
import asyncio
import websockets
import json
import numpy as np

# Load models once when the server starts
predictive_model = PredictiveMaintenanceModel()
engine_condition_predictor = EngineConditionPredictor()

# WebSocket server function
async def handle_connection(websocket, path="/"):
    # When a message is received
    data = await websocket.recv()
    print(f"Received data: {data}")

    try:
        # Parse the received data as JSON
        instance_data = json.loads(data)  # Deserialize JSON to a dictionary

        # Extract input data for the models
        predictive_input = instance_data.get("predictive_model_input")
        engine_condition_input = instance_data.get("engine_condition_input")

        # Ensure that both required inputs are available
        if predictive_input is None or engine_condition_input is None:
            raise ValueError("Missing required input data for one of the models")

        # Convert inputs to numpy arrays (as both models expect numpy arrays)
        # Reshaping the input to 2D (single sample)
        predictive_input = np.array(predictive_input).reshape(1, -1)  # Shape it into (1, n_features)
        engine_condition_input = np.array(engine_condition_input).reshape(1, -1)  # Shape it into (1, n_features)

        # Process the data with the predictive maintenance model
        predicted_failure = predictive_model.predict_failure(predictive_input)
        print(f"Predicted Failure Type: {predicted_failure}")

        # Process the data with the engine condition predictor model
        predicted_condition = engine_condition_predictor.predict_condition(engine_condition_input)
        print(f"Predicted Engine Condition: {predicted_condition}")

        # Convert predictions to Python native types (int, str, float, etc.)
        predicted_failure = predicted_failure.item() if isinstance(predicted_failure, np.generic) else predicted_failure
        predicted_condition = predicted_condition.item() if isinstance(predicted_condition, np.generic) else predicted_condition

        # Send back the predictions to the client
        response = {
            "Predicted Failure Type": predicted_failure,
            "Predicted Engine Condition": predicted_condition
        }
        await websocket.send(json.dumps(response))  # Send the response as JSON

    except Exception as e:
        error_message = {"error": str(e)}
        await websocket.send(json.dumps(error_message))  # Send error message to client

# Start the WebSocket server
async def main():
    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    await server.wait_closed()

# Run the WebSocket server
if __name__ == "__main__":
    asyncio.run(main())
