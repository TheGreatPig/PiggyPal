from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.exporters.onnx import main_export
import os

model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Automatically convert to ONNX
main_export(
    model_name_or_path=model_name,
    output=os.path.join("models", "mDeBERTa-v3-base-mnli-xnli"),
    task="text-classification",
    opset=14
)
