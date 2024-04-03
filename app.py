import pickle
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum

app=FastAPI()
handler=Mangum(app)

@app.get('/health')
def health_check():
  return JSONResponse({"health":"Okay"})

if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0", port=8000)

