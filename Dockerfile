FROM python:3.10
RUN pip install --upgrade pip
WORKDIR /intent-classifier
COPY requirements.txt .
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install pip install xformers
RUN python3 -m pip install -r requirements.txt
COPY main.py .
COPY models /intent-classifier/models
CMD [ "python", "main.py"]
