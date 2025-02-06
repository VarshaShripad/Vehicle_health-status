import json
import numpy as np
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from models import PredictiveMaintenanceModel, EngineConditionPredictor
import warnings
warnings.filterwarnings("ignore")

# Initialize models
predictive_model = PredictiveMaintenanceModel()
engine_condition_predictor = EngineConditionPredictor()

# Maintain connected vehicles and monitoring clients
connected_vehicles = {}
monitoring_clients = set()

# Create FastAPI instance
app = FastAPI()

# Configure logging for better debugging and error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for all origins (if needed for the client-side app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle vehicle WebSocket connections
@app.websocket("/vehicle")
async def handle_vehicle(websocket: WebSocket):
    await websocket.accept()
    try:
        async for message in websocket.iter_text():
            instance_data = json.loads(message)
            vehicle_id = instance_data.get("vehicle_id", str(websocket.client))

            # Prepare input for models
            predictive_input = np.array(instance_data["predictive_model_input"]).reshape(1, -1)
            engine_condition_input = np.array(instance_data["engine_condition_input"]).reshape(1, -1)

            # Predict failure type and engine condition
            predicted_failure = predictive_model.predict_failure(predictive_input)
            predicted_condition = engine_condition_predictor.predict_condition(engine_condition_input)

            response = {
                "vehicle_id": vehicle_id,
                "Predicted Failure Type": str(predicted_failure),
                "Predicted Engine Condition": str(predicted_condition),
            }

            # Store the vehicle data in the connected vehicles dictionary
            connected_vehicles[vehicle_id] = response

            # Send the prediction response to the vehicle
            await websocket.send_text(json.dumps(response))

            # Broadcast the new prediction to monitoring clients
            if monitoring_clients:
                broadcast_message = json.dumps(response)
                await asyncio.gather(*[client.send_text(broadcast_message) for client in monitoring_clients])

    except WebSocketDisconnect:
        logger.info(f"Vehicle client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"Error in vehicle handler: {e}")

# Handle monitoring client connections
@app.websocket("/monitor")
async def handle_monitor(websocket: WebSocket):
    # Add monitoring client to the set
    await websocket.accept()
    monitoring_clients.add(websocket)
    try:
        # Send current vehicle data to the monitoring client
        for vehicle_data in connected_vehicles.values():
            await websocket.send_text(json.dumps(vehicle_data))

        # Keep the connection open and listen for any messages (currently no actions needed for monitor)
        await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info(f"Monitoring client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"Error in monitor handler: {e}")
    finally:
        # Remove the monitoring client from the set when disconnected
        monitoring_clients.remove(websocket)

# Start the FastAPI server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
