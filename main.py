import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import motor.motor_asyncio
from bson import ObjectId
from bson.errors import InvalidId
from fastapi.middleware.cors import CORSMiddleware
from app.rag_pipeline import RagPipeline

# --- Utility ---
def safe_user_id(user_id: str):
    """Convert to ObjectId if possible, else return as string (e.g., 'guest-user')."""
    try:
        return ObjectId(user_id)
    except (InvalidId, TypeError):
        return user_id


# --- MongoDB Setup ---
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://toivi:nKBJyrQxbWtawZqE@cluster0.5hriaqs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.legiwise  # database name

# --- RAG Model ---
# --- THIS IS THE MODIFIED LINE ---
# The API will now use your custom-built Ollama model.
rag_agent = RagPipeline(model="indian-legal-assistant")

app = FastAPI(title="LegiWise Chat API")

# --- Middleware (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class UserRequest(BaseModel):
    username: str
    email: str
    image: str


class ChatRequest(BaseModel):
    chatID: str
    userID: str
    userPrompt: str
    imgName: Optional[str] = None
    doc_state: Optional[str] = None


class UpdateRequest(BaseModel):
    messageID: str
    updatedResponse: str


# --- Helpers ---
def serialize_doc(doc):
    return {
        "id": str(doc["_id"]),
        "chatID": doc["chatID"],
        "userPrompt": doc["message"]["userPrompt"],
        "llmResponse": doc["message"]["llmResponse"],
        "imgName": doc["message"].get("imgName"),
    }


# --- User Routes ---
@app.post("/user/register")
async def register_user(user: UserRequest):
    """Register new user or return existing"""
    existing = await db.users.find_one({"email": user.email})
    if existing:
        return {"userID": str(existing["_id"]), "message": "User already exists"}
    result = await db.users.insert_one(user.dict())
    return {"userID": str(result.inserted_id), "message": "User registered"}


# --- Chat Routes ---
@app.post("/chat/send")
async def send_message(req: ChatRequest):
    """Save user prompt and generate response"""
    user_filter = {"_id": safe_user_id(req.userID)}
    user = await db.users.find_one(user_filter)

    if not user and req.userID != "guest-user":
        raise HTTPException(status_code=404, detail="User not found")

    # --- THIS LOGIC IS NOW CORRECTED ---
    # It now correctly handles both document-based and general queries.
    if req.doc_state and req.doc_state.strip():
        # Document Mode
        rag_agent.create_knowledge_base(req.doc_state)
        answer = rag_agent.ask(req.userPrompt)
    else:
        # General Mode
        answer = rag_agent.general_ask(req.userPrompt)

    chat_doc = {
        "participant": safe_user_id(req.userID),
        "chatInfo": {"icon": None, "title": None},
        "chatID": req.chatID,
        "message": {
            "userPrompt": req.userPrompt,
            "llmResponse": answer,
            "imgName": req.imgName,
        },
    }

    await db.chats.insert_one(chat_doc)
    return {"success": True, "chat": serialize_doc(chat_doc)}


@app.get("/chat/{chatID}")
async def get_history(chatID: str, userID: str):
    """Fetch chat history for a given chat"""
    user_filter = {"chatID": chatID, "participant": safe_user_id(userID)}
    cursor = db.chats.find(user_filter)
    history = []
    async for doc in cursor:
        history.append(serialize_doc(doc))
    return {"success": True, "messages": history}


@app.post("/chat/update")
async def update_response(req: UpdateRequest):
    """Update an existing response"""
    result = await db.chats.update_one(
        {"_id": ObjectId(req.messageID)},
        {"$set": {"message.llmResponse": req.updatedResponse}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Message not found or unchanged")
    updated_doc = await db.chats.find_one({"_id": ObjectId(req.messageID)})
    return {"chat": serialize_doc(updated_doc), "success": True}


@app.get("/chat/sidebar")
async def get_sidebar(userID: str):
    """Fetch list of chats for sidebar"""
    user_filter = {"participant": safe_user_id(userID)}
    cursor = db.chats.find(user_filter, {"chatID": 1, "chatInfo": 1})
    chats = await cursor.to_list(length=100)
    return {"success": True, "chats": chats}
