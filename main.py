import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import pandas as pd
import re

app = FastAPI()

# Allow requests from http://localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define Google Cloud Vision credentials as a dictionary
credentials = {
  "type": "service_account",
  "project_id": "ocr-demo-azure-ai-vision",
  "private_key_id": "164894fad2d5d6c393fb335ab9991b589e9ade3a",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCHVeKaSyliQmMp\ntpqT3btrhxTjZ1b6NUVBgZid/GA68hCSkR30bMESS7hxuTy1csYc+cNmZeXDxas2\n6qBI6AY9+3BPii6ozZ7xNzX9NmvDJ6kbS3ENmhhAmansZpggO716BGxFPBCzKGc8\nQ9brwY8CN9r8qMR63ZVX92ps9yEwE1swTXGUoi0QSd1AteOvzQMArqq5Yi86r5//\nmk6hGfAJNb9HeiR4UsZNAxHUUyPshTjXBj8sBxJQNpuUuAv0fDwpyOPqrf6Ii8cm\nRmkyqkw1aETUAPr3wHjYyniesQ6tjHfeyzdU/nwk2o0FgKEYzIl20nv+XDsjuuDK\nRKLB3WxLAgMBAAECggEANRxAVFrAv93buof9u5+dqIIXcXKhkImNRzrLoNxlC1zp\nogsvHTf8wOhUVtTh/TcwhjiCNVIxzBVrrp5/Dn13zaU6GwzYaMhg4rrPBjLwMWME\nG0EM+dCKSffD1pEgjC3FmvfXYYhD6XsmoGDjHBwWukEfQF4e1TTWIfJnoqGNNacg\n3OZXHjT4/jrD5RhssutNDOWV5PdPDZ1v/CYsvOwjQN8i55jhEim1jOvibE6N/wWH\nOZuXBCFcsFYVu7Q1lLLYzoKgCxVuEBxsOqOTzxTRTsyUXaVqFCN+nhf7zGutvlnO\nlPbM2+xDuYXrZJztSVG08bd+OIhdwYfnt0y9tEzC8QKBgQC8PfpumQUt9fRXC0q8\nLhsJv4M1pUoA3jHmIp+AJI87LCGbq3tUq8UCMHJzehqBPhToDJ06rYVzPXCWdlz+\n78pKjXbr6dTBxp1kUBKEbeAD3T+z0gVzvQdyRQzjxQKNXruHFlHSAdZ/6lzaoTex\nY43+aP7+sE0tfF0HfqzcZ0TgMwKBgQC4DLMa05kuplsJkcarxLQnYND4VTWxl4/G\nsA6TaP5yXNS3Vdf3MgtQuL6cf8bhl9vDfwCqmEhhGC4+rFOIb4EbEwYqZFnwB1+g\nVIweeVCTD5ZBQZbMmfRQAoeQSZi+n2BOCCydTq+Mc0AvAdRRx5bPbAYdYWxEY+72\nOAJjoePLiQKBgEXAQXeRw9WP+YX3bS3ld6dZC2lpYc6Ihrzbv3ZgFaK7a4ifNgfd\nzhZNlVsst32EX4LMicYgXf6hmYJnQXZFrBOL77Di6C8VRWTSNspTXFqSNPSQsex8\n8rFo3KnZamSv4ZTgtFi4zZ6AXP+2FUjptse6aCI/eZmNJ3uLeMoCigb5AoGBAI+J\nX/lRAsst9Bvfc6isTK/VQsQZeDmbcQbMcWGnZaF9Imwk57wibE609fsJb+qqSzsJ\nBlUFVJVcjVxVewQRqgeaa5mOD9IxffOFXI27oQpAAre71kaU3sOzZVQzAYvQsgPJ\nPjokjqYjj+/ZJmPtG4GCxrYNL6maolel0L8xF325AoGAQXDjOWcMAHz568rWYIuB\nhkKCHW9ihiFJXGkeKm0EU1DMF+m0VT3/N9RUmY5FoV3qXIbMG9v0lXGd+Pn8/9s1\nnV1wiSUsvA5H8lcvsPa6jAyDS+12sRXC5HoRSlDEHH6vB8SPM3VfXLcv/ayu/x2x\n0GbObFBONBeTfMK/Yn/89vY=\n-----END PRIVATE KEY-----\n",
  "client_email": "ocr-demo-google-vision-service@ocr-demo-azure-ai-vision.iam.gserviceaccount.com",
  "client_id": "116263851015712687067",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/ocr-demo-google-vision-service%40ocr-demo-azure-ai-vision.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# Initialize Google Cloud Vision client with the credentials
client = vision.ImageAnnotatorClient.from_service_account_info(credentials)

@app.get("/")
async def welcome_screen():
    return {"message": "Welcome to OCR Python"}

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    # Read the content of the uploaded file
    content = await file.read()
    
    # Perform OCR on the image content
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Extract OCR results
    df = pd.DataFrame(columns=['locale', 'description'])
    for text in texts:
        df = pd.concat([df, pd.DataFrame({'locale': [text.locale], 'description': [text.description]})], ignore_index=True)

    ocr_output = df['description'][0]
    
    # Extract last name
    last_name_index = ocr_output.find("Last Name")
    if last_name_index != -1:
        last_name_section = ocr_output[last_name_index + len("Last Name"):]
        last_name = last_name_section.split()[0]
    else:
        last_name = None

    # Extract first name
    first_name_index = ocr_output.find("Given Names")
    if first_name_index != -1:
        first_name_section = ocr_output[first_name_index + len("Given Names"):]
        first_name = first_name_section.split()[0]
    else:
        first_name = None

    # Extract middle name
    middle_name_index = ocr_output.find("Middle Name")
    if middle_name_index != -1:
        middle_name_section = ocr_output[middle_name_index + len("Middle Name"):]
        middle_name = middle_name_section.split()[0]
    else:
        middle_name = None

    # Extract date of birth
    dob_index = ocr_output.find("Petsa ng Kapanganakan/Date of Birth")
    if dob_index != -1:
        dob_section = ocr_output[dob_index + len("Petsa ng Kapanganakan/Date of Birth"):]
        # Using regular expression to extract the date in the format "MONTH DAY, YEAR"
        dob_match = re.search(r'([A-Z]+\s\d+,\s\d+)', dob_section)
        if dob_match:
            dob = dob_match.group(0)
        else:
            dob = None
    else:
        dob = None

    return {"first_name": first_name, "middle_name": middle_name, "last_name": last_name, "date_of_birth": dob}

