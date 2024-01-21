from typing import Union, Annotated
import multipart
from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
import base64
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3000/scan/upload",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    image: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect")
async def detect(image: Annotated[str, Form()]):
    # decode the image from base64
    print(image)
    decoded_bytes = base64.b64decode(image)

    # Create an in-memory file-like object
    image_stream = BytesIO(decoded_bytes)

    # Open the image using PIL
    pil_image = Image.open(image_stream)

    # DEBUG
    pil_image.save("test.jpg")

    # insert segmentation and identification here

    # encode the image in base64
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format="JPEG")
    base64_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    # send it back as a response
    return {"image": base64_string}
