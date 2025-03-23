# download_model.py
from transformers import AutoProcessor, AutoModelForVision2Seq
model_id = "ds4sd/SmolDocling-256M-preview"
target_dir = "models/smol_docling"

# Download processor
AutoProcessor.from_pretrained(model_id).save_pretrained(target_dir)

# Download model
AutoModelForVision2Seq.from_pretrained(model_id).save_pretrained(target_dir)