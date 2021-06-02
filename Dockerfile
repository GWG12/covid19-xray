FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app

RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt