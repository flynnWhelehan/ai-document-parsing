from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
import torch
import time
import json

# region Path assignments
local_model_path = "models/smol_docling"    
file_path = "examples/test_files"
file_name = "x.pdf"
full_file_path = f"{file_path}/{file_name}"
output_path = Path(f"{file_path}/output")
output_path.mkdir(exist_ok=True)
# endregion

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

# region Load model and processor
processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda" if torch.cuda.is_available() else "cpu")
# endregion

def process_image(image, page_number):
    start_time = time.time()
    print(f"Processing page {page_number}...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt", truncation=True).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name=f"Document_Page_{page_number}")
    doc.load_from_doctags(doctags_doc)

    # Save as JSON
    json_path = output_path / f"page_{page_number}.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(doc.export_to_dict(), json_file, indent=2)

    # Save as Markdown
    '''
    md_path = output_dir / f"page_{page_number}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(doc.export_to_markdown())

    # Save as HTML
    html_path = output_dir / f"page_{page_number}.html"
    doc.save_as_html(html_path)'
    '''

    end_time = time.time()
    print(f"Page {page_number} processed in {end_time - start_time:.2f} seconds.")

def process_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=150)
    for i, image in enumerate(images, start=1):
        process_image(image, i)

if __name__ == "__main__":
    start_time = time.time()
    process_pdf(full_file_path)
    end_time = time.time()
    print(f"\nFinished in {end_time - start_time:.2f} seconds.")