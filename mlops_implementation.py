import mlflow
from mlflow.models import infer_signature
import os

def log_model(model, model_name, input_example):
    mlflow.set_tracking_uri("http://localhost:5000")  # Set MLflow tracking server
    mlflow.set_experiment("content_generation")
    
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params({
            "model_name": model_name,
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
        })
        
        # Log the model
        signature = infer_signature(input_example, model(input_example).logits)
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        
        print(f"Model logged in run: {mlflow.active_run().info.run_id}")

# Usage
model = FlexibleContentGenerator("gpt2")
input_example = model.tokenizer("Example input text", return_tensors="pt")
log_model(model.model, "gpt2", input_example)

# To load the model later
loaded_model = mlflow.transformers.load_model(f"runs:/{mlflow.active_run().info.run_id}/model")