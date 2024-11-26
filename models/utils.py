import importlib
import sys

def load_model(model_name):
  model_path = f"models.{model_name.lower()}"
  try:
    module = importlib.import_module(model_path)
    return module.Model
  except ModuleNotFoundError:
    print(f"Error: Model '{model_name}' not found in 'models' folder.")
    sys.exit(1)
