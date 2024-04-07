from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import io
from PIL import Image
from mangum import Mangum

from datetime import datetime
import json

from model import model

app=FastAPI()
handler=Mangum(app)

@app.get('/health')
def health_check():
    return JSONResponse({"health":"Okay"})

@app.post("/predict-user")
async def pred(id: str, file: UploadFile = File(...)):
    req_time_server = datetime.now()
    if file.content_type != "image/png":
        return {"error": "Only PNG images are supported"}

    # Read the image file
    image_bytes = await file.read()

    # Open the image using PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    pred = model.predict(img)
    
    res_time_server = datetime.now()
    
    return JSONResponse(
        {
            "req_id": id,
            "req_time": req_time_server.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "prediction": pred,
            "res_time": res_time_server.strftime("%Y-%m-%d %H:%M:%S.%f"),
        }
    )

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0", port=8000)