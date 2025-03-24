import os
import json
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1000"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.7"))

# Initialize FastAPI app
app = FastAPI(
    title="CropConnect Direct API",
    description="Streamlined API for the CropConnect chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"
    user_id: Optional[str] = None

# Response model
class ChatResponse(BaseModel):
    message: str
    navigations: List[str] = []
    tags: Dict[str, Any] = {}
    language: str
    source_language: Optional[str] = None

# Load navigation data
try:
    with open("app/data/app_context.json", "r") as f:
        APP_CONTEXT = json.load(f)
except Exception as e:
    print(f"Warning: Could not load app_context.json: {e}")
    APP_CONTEXT = {
        "navigation_contexts": [],
        "meta": {}
    }

# Simple in-memory chat history
chat_history = {}

# Helper functions
def detect_language(text: str) -> str:
    """Simple language detection based on character sets"""
    # Check for Hindi characters
    hindi_range = range(0x0900, 0x097F)
    if any(ord(char) in hindi_range for char in text):
        return "hi"
    
    # Default to English
    return "en"

def suggest_navigations(message: str, tags: Dict[str, Any]) -> List[str]:
    """Suggest relevant app navigations based on tags and context"""
    navigations = []
    
    # Always include chatbot
    navigations.append("/chatbot")
    
    # Add podcasts for learning
    if "crops" in tags and tags["crops"]:
        navigations.append("/podcasts")
    
    # Add community for discussions
    navigations.append("/community")
    
    # Add resource pool if equipment related
    if "topics" in tags and any(topic in ["equipment", "machinery", "tools"] for topic in tags.get("topics", [])):
        navigations.append("/resource-pool")
    
    # Limit to 3 suggestions
    return navigations[:3]

@app.get("/")
async def root():
    return {"message": "Welcome to CropConnect Direct API"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response"""
    try:
        # Detect source language
        source_language = detect_language(request.message)
        
        # Create system prompt
        system_prompt = f"""You are the agricultural assistant for the CropConnect app, a platform for farmers.

Your task is to:
1. Provide helpful farming advice in {"Hindi" if request.language == "hi" else "English"} language
2. Extract relevant crop names, locations, and topics from the query

CRITICAL INSTRUCTIONS - YOUR RESPONSE FORMAT:
Return a VALID JSON object with exactly this structure:
{{
    "message": "Your farming advice here in {"Hindi" if request.language == "hi" else "English"}",
    "tags": {{
        "crops": ["crop1", "crop2"],  // List of crops mentioned IN ENGLISH ONLY
        "city": "location_name",      // Location mentioned IN ENGLISH ONLY
        "topics": ["topic1", "topic2"], // Agricultural topics IN ENGLISH ONLY
        "issues": ["issue1", "issue2"], // Farming problems IN ENGLISH ONLY
        "seasons": ["season1", "season2"] // Seasons IN ENGLISH ONLY
    }}
}}

ONLY return the JSON object, nothing else. Make sure it's valid JSON.
"""
        
        # Prepare request payload
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "temperature": CLAUDE_TEMPERATURE,
            "system": system_prompt,
            "messages": [{"role": "user", "content": request.message}]
        }
        
        # Make direct API call to Claude
        async with httpx.AsyncClient() as client:
            claude_response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json=payload,
                timeout=30.0
            )
            
            # Check if the request was successful
            if claude_response.status_code != 200:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API Error: {claude_response.status_code}"}
                )
            
            # Parse the response
            response_data = claude_response.json()
            
            # Extract the text content
            if 'content' in response_data and len(response_data['content']) > 0:
                response_text = response_data['content'][0]['text']
            else:
                response_text = "Sorry, I couldn't generate a response."
            
            # Try to parse as JSON
            try:
                # Find the JSON part of the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_response = json.loads(json_str)
                    
                    # Extract message and tags
                    message = parsed_response.get("message", response_text)
                    tags = parsed_response.get("tags", {})
                    
                    # Ensure tags are properly formatted
                    for key in ["crops", "topics", "issues", "seasons"]:
                        if key not in tags or tags[key] is None:
                            tags[key] = []
                else:
                    # If no JSON found, use the whole text as message
                    message = response_text
                    tags = {
                        "crops": [],
                        "city": None,
                        "topics": [],
                        "issues": [],
                        "seasons": []
                    }
            except json.JSONDecodeError:
                # If parsing fails, use the whole text as message
                message = response_text
                tags = {
                    "crops": [],
                    "city": None,
                    "topics": [],
                    "issues": [],
                    "seasons": []
                }
            
            # Generate navigation suggestions
            navigations = suggest_navigations(request.message, tags)
            
            # Create the response
            response = {
                "message": message,
                "navigations": navigations,
                "tags": tags,
                "language": request.language,
                "source_language": source_language
            }
            
            # Store in chat history if user_id provided
            if request.user_id:
                if request.user_id not in chat_history:
                    chat_history[request.user_id] = []
                
                chat_history[request.user_id].append({
                    "role": "user",
                    "content": request.message
                })
                
                chat_history[request.user_id].append({
                    "role": "assistant",
                    "content": message
                })
                
                # Limit history to 10 messages
                if len(chat_history[request.user_id]) > 10:
                    chat_history[request.user_id] = chat_history[request.user_id][-10:]
            
            return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = os.getenv("APP_RELOAD", "true").lower() == "true"
    
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)