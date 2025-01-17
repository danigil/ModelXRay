FROM tensorflow/tensorflow:2.14.0
WORKDIR /usr/local/app

RUN pip install fastapi numpy uvicorn scikit-learn pandas tqdm

COPY . ./ModelXRay

WORKDIR /usr/local/app/ModelXRay/
RUN pip install -e .

ARG MODELXRAY_MODELS_DIR
# COPY ${MODELXRAY_MODELS_DIR}/*.keras /usr/local/app/models
# COPY /mnt/exdisk1/model_xray/models_subset/*.keras /usr/local/app/models
# RUN mkdir -p /usr/local/app/models
# COPY ./models/*.keras /usr/local/app/models

# ARG MODELS_DIR
# ENV MODELS_DIR /usr/local/app/models

EXPOSE 8001

WORKDIR /usr/local/app/ModelXRay/server
CMD ["uvicorn",  "api:app", "--host", "0.0.0.0", "--port", "8001"]