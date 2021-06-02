from fastapi import APIRouter, UploadFile, File, HTTPException
from nn.inception_v3 import predict_image, model_ft
from image_similarity.ImageSimilarity import ImageSimilarity


router = APIRouter(
    prefix="/v1/image",
    tags=["Images"],
    responses={404: {"description": "Not found"}},)


@router.post("/")
async def post_image(file: UploadFile = File(...)):
    similarity = ImageSimilarity()
    similarity_veredict = similarity.compare_image(file.file)
    print(f'similarity ranking: {similarity_veredict}')
    if similarity_veredict:
        print("inside similarity approved")
        data = predict_image(file.file, model_ft)
        print('la data ', data)
        return {"response": data}
    raise HTTPException(
        status_code=422, detail="Invalid input, please verify you are sending a clean chest x-ray image")
