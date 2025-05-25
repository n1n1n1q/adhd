from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import subprocess
import json
from datetime import datetime
from pathlib import Path
import uuid
import random

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]  # Next.js dev server
UPLOAD_DIR = Path("./back/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

with open("back/.env_info", "r", encoding="utf-8") as f:
    uri = f.read().strip()


client = MongoClient(uri, server_api=ServerApi("1"))
db = client["etdh"]
videos_collection = db["videos"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")


class TimestampsModel(BaseModel):
    event: str = None
    start: str = None
    end: str = None


class VideoCreate(BaseModel):
    video_name: str
    file_path: str
    timestamps: list[TimestampsModel] = None


@app.post("/api/videos/", status_code=201)
async def create_video(video_name: str, video_file: UploadFile):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(video_file.filename)[1]
    safe_filename = (
        f"{video_name.replace(' ', '_')}_{timestamp}_{unique_id}{file_extension}"
    )

    file_path = UPLOAD_DIR / safe_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    print(f"Extracting metadata from {file_path}")
    metadata = extract_video_metadata(str(file_path))

    timestamps = []
    result = insert_video(
        video_name=video_name,
        file_path=str(file_path),
        timestamps_json=timestamps,
        metadata=metadata,
    )

    return {
        "id": str(result),
        "video_name": video_name,
        "file_path": str(file_path),
        "file_url": f"/static/videos/{safe_filename}",
        "metadata": metadata,
        "message": "Video successfully uploaded and added to database",
    }


def insert_video(video_name, file_path, timestamps_json, metadata=None):
    created_at = metadata.get("format", {}).get("tags", {}).get("creation_time", None)
    video_document = {
        "video_name": video_name,
        "file_path": file_path,
        "timestamps": timestamps_json,
        "metadata": metadata or {},
        "created_at": created_at if metadata else "",
    }

    result = videos_collection.insert_one(video_document)
    return result.inserted_id


@app.get("/api/videos/")
async def get_videos():
    videos = list(videos_collection.find({}))

    for video in videos:
        video["_id"] = str(video["_id"])

        filename = os.path.basename(video["file_path"])
        video["url"] = f"/static/videos/{filename}"

        if "created_at" in video and isinstance(video["created_at"], datetime):
            video["created_at"] = video["created_at"].isoformat()

    return videos


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    from bson.objectid import ObjectId

    video = videos_collection.find_one({"_id": ObjectId(video_id)})

    if not video:
        return {"error": "Video not found"}, 404

    video["_id"] = str(video["_id"])

    if "created_at" in video and isinstance(video["created_at"], datetime):
        video["created_at"] = video["created_at"].isoformat()

    filename = os.path.basename(video["file_path"])
    video["url"] = f"/static/videos/{filename}"

    if "metadata" not in video or not video["metadata"]:
        if os.path.exists(video["file_path"]):
            print(f"Extracting missing metadata for video {video_id}")
            metadata = extract_video_metadata(video["file_path"])
            video["metadata"] = metadata
            videos_collection.update_one(
                {"_id": ObjectId(video_id)}, {"$set": {"metadata": metadata}}
            )
        else:
            video["metadata"] = {"error": "File not found"}

    return video


def generate_random_timestamps():
    random_timestamps = []
    event_types = ["стілець", "кнур", "goko_4orta", "tank", "pidoor", "acceleration"]

    num_events = random.randint(5, 10)

    for i in range(num_events):
        event_name = random.choice(event_types)
        start_time = random.randint(0, 10)
        random_timestamps.append(
            {"event": f"{event_name} {i+1}", "start": start_time}
        )

    return random_timestamps


@app.get("/api/compare-videos/")
async def compare_videos(video_id1: str, video_id2: str):
    print("doing smth")
    from bson.objectid import ObjectId

    video1 = videos_collection.find_one({"_id": ObjectId(video_id1)})
    video2 = videos_collection.find_one({"_id": ObjectId(video_id2)})

    if not video1 or not video2:
        return {"error": "One or both videos not found"}, 404

    random_timestamps = generate_random_timestamps()

    video1["timestamps"] = random_timestamps
    video2["timestamps"] = random_timestamps

    videos_collection.update_one(
        {"_id": ObjectId(video_id1)}, {"$set": {"timestamps": random_timestamps}}
    )

    videos_collection.update_one(
        {"_id": ObjectId(video_id2)}, {"$set": {"timestamps": random_timestamps}}
    )

    filename1 = os.path.basename(video1["file_path"])
    filename2 = os.path.basename(video2["file_path"])

    video1["url"] = f"/static/videos/{filename1}"
    video2["url"] = f"/static/videos/{filename2}"
    video1["_id"] = str(video1["_id"])
    video2["_id"] = str(video2["_id"])
    video1["metadata"] = None
    video2["metadata"] = None
    return {
        "video1": video1,
        "video2": video2,
        "message": "Random timestamps generated and saved to database",
    }


@app.post("/api/videos/{video_id}/extract-metadata")
async def extract_video_metadata_endpoint(video_id: str):
    from bson.objectid import ObjectId

    video = videos_collection.find_one({"_id": ObjectId(video_id)})

    if not video:
        return {"error": "Video not found"}, 404

    if not os.path.exists(video["file_path"]):
        return {"error": f"Video file not found at {video['file_path']}"}, 404

    metadata = extract_video_metadata(video["file_path"])

    videos_collection.update_one(
        {"_id": ObjectId(video_id)}, {"$set": {"metadata": metadata}}
    )

    return {
        "id": str(video["_id"]),
        "video_name": video["video_name"],
        "metadata": metadata,
        "message": "Metadata successfully extracted and updated",
    }


def extract_video_metadata(file_path):
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running ffprobe: {result.stderr}")
            return {"error": "Failed to extract metadata"}

        metadata = json.loads(result.stdout)

        return metadata

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {"error": str(e)}
