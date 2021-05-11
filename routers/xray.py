from models.xray import Xray
from fastapi import APIRouter, HTTPException, UploadFile, File
#import psycopg2
from typing import List, Optional
#from passlib.context import CryptContext
from nn.inception_v3 import predict_image, model_ft


router = APIRouter(    
    prefix="/v1/image",
    tags=["Images"],
    #dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},)

@router.post("/")
async def post_image(file: UploadFile =  File(...)):
    data = predict_image(file.file, model_ft)
    print('la data ',data)
    return {"response": data}    