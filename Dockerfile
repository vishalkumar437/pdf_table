# Docker file to Dockerize django application
# Version: 1.0
FROM python:3.10
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code


RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get install -y poppler-utils

# Install Tesseract
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    apt-get install -y tesseract-ocr-ara

RUN rm -rf /var/lib/apt/lists/*
# Copy project
COPY . /code/

# install requirements
RUN pip install -r requirements.txt
RUN npm install
RUN npx run dev

# Tessaract_Prefix 
ENV TESSDATA_PREFIX /usr/share/tesseract-ocr/5/tessdata
# Expose port
EXPOSE 3000

# Run the application
CMD ["sh", "-c", "npm run dev && \ 
                    python manage.py runserver 0.0.0.0:8000"]
