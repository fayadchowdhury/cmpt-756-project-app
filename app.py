from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import numpy as np
import io
from PIL import Image
from mangum import Mangum

from model.model import UNet, transform

import torch

app=FastAPI()
handler=Mangum(app)

model = UNet()
model.load_state_dict(torch.load('model/checkpoint.pt', map_location=torch.device('cpu')))

print("Configured model")

@app.get('/health')
def health_check():
    return JSONResponse({"health":"Okay"})

async def sample_pred(image_bytes):
    print("Received request")
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    print("Opened image")
    image_new = transform(image)  # Assuming transform is defined somewhere
    image_new = image_new.unsqueeze(0)
    print("Transformed and unsqueezed image")
    print("Beginning inference")
    pred = model(image_new)  # Assuming model is defined globally
    print("Inference done")
    output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()
    
    # Convert the output mask to a bitmap image
    output_image = Image.fromarray(output_mask.astype('uint8') * 255, mode='L')
    with io.BytesIO() as output:
        output_image.save(output, format="BMP")
        processed_image_bytes = output.getvalue()
    
    print("Image processed")
    return processed_image_bytes

@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    # Read the uploaded image as bytes
    image_bytes = await image.read()

    # # Process the image (example: convert to grayscale)
    # processed_image_bytes = process_image_in_memory(image_bytes)
    
    # Run inference on the image
    processed_image_bytes = await sample_pred(image_bytes)
    
    # Return the processed image as a stream
    return StreamingResponse(io.BytesIO(processed_image_bytes), media_type="image/bmp")

def process_image_in_memory(image_bytes):
    # Example: Convert the image to grayscale using PIL
    image = Image.open(io.BytesIO(image_bytes))
    
    # Ensure the image is in "RGB" mode for compatibility
    image = image.convert("RGB")
    
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Save the processed image as bytes
    with io.BytesIO() as output:
        gray_image.save(output, format="BMP")
        processed_image_bytes = output.getvalue()
    return processed_image_bytes


@app.post("/process-image-sync/")
def process_image(image: UploadFile = File(...)):
    # Read the uploaded image as bytes
    image_bytes = image.file.read()

    # Run inference on the image
    output_mask = sample_pred_sync(image_bytes)

    # Convert the output mask to a BMP image
    output_image = Image.fromarray(output_mask.astype('uint8') * 255, mode='L')
    with io.BytesIO() as output:
        output_image.save(output, format="BMP")
        processed_image_bytes = output.getvalue()

    # Return the processed image as a stream
    return StreamingResponse(io.BytesIO(processed_image_bytes), media_type="image/bmp")

def sample_pred_sync(image_bytes):
    print("Received request")
    # Load the BMP image
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    print("Opened image")

    # Apply transformations
    image_new = transform(image)
    image_new = image_new.unsqueeze(0)
    print("Transformed and unsqueezed image")

    # Perform inference
    pred = model(image_new)
    print("Inference done")

    # Get the output mask
    output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

    # Save the output mask as BMP (Optional)
    # plt.imsave("output.bmp", output_mask)
    # print("Saved image")

    return output_mask



if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0", port=8000)

