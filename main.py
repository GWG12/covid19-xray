from typing import Optional
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordBearer
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routers import xray


app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app.include_router(xray.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)    