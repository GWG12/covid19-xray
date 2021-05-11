from pydantic import BaseModel, Field


class Xray(BaseModel):
    image: bytes = Field(..., title="X-ray image")
