import os
from typing import List
import numpy as np
import cv2
import dlib
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pymongo import MongoClient, errors
import gridfs
from bson import ObjectId
from pydantic import BaseModel

app = FastAPI()

# MongoDB Atlas connection
connection_string = "mongodb+srv://davidihenatuoha:VCWFM86dntXkYI1P@cluster0.3iifs6b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(connection_string)
db = client["paylensdb"]
users_collection = db["Users"]
data_collection = db["data"]
fs = gridfs.GridFS(db)

# Ensure unique index on email field
users_collection.create_index([("email", 1)], unique=True)

# Remove unique index on account_name field if it exists
#try:
 #   users_collection.drop_index("account_name_1")
#except Exception as e:
#    print(f"No unique index on account_name to drop: {e}")

# Paths to the model files
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Initialize dlib's face detector and face landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Function to compute the 128D face encoding for an image
def get_face_encodings(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    shape = predictor(gray, face)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

@app.post("/upload_user_details/")
async def upload_user_details(
    email: str = Form(...),
    phone: str = Form(...),
    #password: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    country: str = Form(...),
    username: str = Form(...),
    gender: str = Form(...),
    #isEmailVerified: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        # Check if user with this email already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        # Insert user details into Users collection
        user_id = ObjectId()
        user_data = {
            "_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "country": country,
            "gender": gender,
            "username": username,
        }

        users_collection.insert_one(user_data)
        print(f"Inserted user with ID: {user_id}")

        # Process and store images
        image_ids = []
        image_encodings = []
        for file in files:
            # Read the file content
            file_content = await file.read()
            if not file_content:
                print(f"File {file.filename} is empty")
                continue

            # Save file content to GridFS
            filename = os.path.basename(file.filename)
            image_id = fs.put(file_content, filename=filename, user_id=user_id)
            image_ids.append(image_id)
            print(f"Stored image {filename} with ID: {image_id}")

            # Convert file content to numpy array and decode
            np_arr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Failed to decode image {filename}")
                continue

            # Compute face encodings
            encoding = get_face_encodings(image)
            if encoding is not None:
                image_encodings.append(encoding.tolist())
                print(f"Computed encoding for image {filename}")

        # Update user document with image IDs and encodings
        if image_ids:
            users_collection.update_one(
                {"_id": user_id},
                {"$set": {"image_ids": image_ids, "image_encodings": image_encodings}}
            )
            print(f"Updated user document with image IDs and encodings for user ID: {user_id}")

        return {"message": "User data and images uploaded successfully", "user_id": str(user_id)}

    except HTTPException as he:
        print(f"HTTP error occurred: {he.detail}")
        raise
    except errors.DuplicateKeyError as dke:
        print(f"Duplicate key error occurred: {dke.details}")
        raise HTTPException(status_code=400, detail="User with this email already exists")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_bank_details/")
async def upload_bank_details(
    email: str = Form(...),
    account_name: str = Form(...),
    account_number: str = Form(...),
    bank_name: str = Form(...)
):
    try:

        # Check if the user exists
        user = users_collection.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        

        user_id = user["_id"]

        # Insert bank details into data collection with the same _id as user_id
        data_collection.insert_one({
            "_id": user_id,
            "account_name": account_name,
            "account_number": account_number,
            "bank_name": bank_name,
        })
        print(f"Inserted bank details for user ID: {user_id}")

        return {"message": "Bank details uploaded successfully", "user_id": str(user_id)}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    try:
        # Read the file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")

        # Convert file content to numpy array and decode
        np_arr = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Compute face encoding for the uploaded image
        encoding = get_face_encodings(image)
        if encoding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Search for a matching face in the database
        users = users_collection.find({})
        for user in users:
            for stored_encoding in user.get("image_encodings", []):
                stored_encoding = np.array(stored_encoding)
                distance = np.linalg.norm(stored_encoding - encoding)
                if distance < 0.35:  # A threshold for face match
                    return {"message": "Face recognized", "user_id": str(user["_id"])}

        return {"message": "Face not recognized"}

    except HTTPException as he:
        print(f"HTTP error occurred: {he.detail}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))



class AccountInfoRequest(BaseModel):
    account_number: str

@app.post("/account_info/")
async def get_account_info(account_info: AccountInfoRequest):
    try:
        # Fetch user details from data collection
        user_data = data_collection.find_one({"account_number": account_info.account_number})
        
        if user_data:
            return {
                "account_number": user_data["account_number"],
                "bank_name": user_data["bank_name"],
                "bank_code": user_data["bank_code"],
                "account_name": user_data["account_name"]
            }
        else:
            raise HTTPException(status_code=404, detail="No matching account found")

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
