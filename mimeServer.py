import whisper
from dotenv import load_dotenv
import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
import traceback

load_dotenv()
EXPRESS_SERVER_URL=os.getenv("EXPRESS_BASE_URL")
origins = [EXPRESS_SERVER_URL]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s -%(lineno)d'
)

logger=logging.getLogger(__name__)

print("FASTAPI Server Started")

app=FastAPI(
    title="FastAPI Transcription Server",
    description="OpenAI Whisper speech to text server for Mime voice command"
)

origins=[
    "https://abundant-merissa-aakash-star320-05713f07.koyeb.app",  # Express server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model=None

class TranscriptedResponse(BaseModel):
    success:bool
    transcription:str
    processing_time_ms: Optional[int]=None
    language: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming HTTP Request: {request.method} {request.url.path}")
    print(f"📨 [FastAPI] Headers: {dict(request.headers)}")
    print(f"📨 [FastAPI] Client: {request.client.host}:{request.client.port}")
    response=await call_next(request)
    print(f"Response is {response.status_code}")
    return response

@app.on_event("startup")
async def initialize_whisper_mode():
    global whisper_model
    try:
        print("Loading whisper 'base' model")
        start_time=time.time()
        whisper_model=whisper.load_model("base")
        load_time=(time.time()-start_time)*1000
        print(f"✅ FastAPI whisper model loaded in {load_time} ms")
    except Exception as e:
        print(f"Failed to load whisper model {e}")
        raise

@app.get("/")
def root():
    return {
        "message": "Whisper Transcription Server is running",
        "model_loaded": whisper_model is not None,
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": whisper_model is not None,
        "model_name": "base",
        "timestamp": time.time()
    }

@app.post("/transcribe",response_model=TranscriptedResponse)
async def transcribe_audio_to_text(audio:UploadFile=File(...)):
    temp_file_path=None
    print("\n🎤 FastAPI transcription request received")
    try:
        if whisper_model is None:
            print("Whisper model not loaded!!")
            raise HTTPException(
                status_code=503, 
                detail="Whisper model not loaded. Please restart the server."
            )
        print(f"Filename: {audio.filename}")
        print(f"Content type: {audio.content_type}")

        if not(audio.filename):
          print(f"❌ [FastAPI] [{request_id}] No filename provided")
          raise HTTPException(status_code=400, detail="No filename provided")
    
        audio_data=await audio.read()
        print(f"The audio data size is {len(audio_data)} bytes")

        if len(audio_data)<1000: #less than 1Kb
          print(f"File too small!!")
          return TranscriptedResponse(
                success=False,
                transcription="",
                message=f"Audio file too small: {len(audio_data)} bytes",
          )
    
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as temp_file:
           temp_file.write(audio_data)
           temp_file_path=temp_file.name
           print(f"Temporary audio data storing file size {os.path.getsize(temp_file_path)} bytes")
           transcription_start=time.time()
           final_result=whisper_model.transcribe(
            temp_file_path,
            language="en",  # Force English for better accuracy
            task="transcribe",  # transcribe (not translate)
            fp16=False,  # Use FP32 for better compatibility
            verbose=False  # Reduce output
        )

        transcription_time = int((time.time() - transcription_start) * 1000)
        print(f"Transcription completed in {transcription_time} ms")

        transcribed_text = final_result["text"].strip()
        transcription_language = final_result.get("language", "unknown")

        print(f"Detected text is '{transcribed_text}'")
        
        if not(transcribed_text):
            return TranscriptedResponse(
                success=False,
                transcriptionss="",
                message="No speech detected in audio",
                processing_time_ms=transcription_time,
                language=transcription_language
            )
        
        print(f"✅Transcription succesfully completed")
        
        return TranscriptedResponse(
                success=True,
                transcription=transcribed_text,
                message="Transcription successful",
                processing_time_ms=transcription_time,
                language=transcription_language
        )
    except HTTPException:
      raise
    except Exception as e:
        print(f"Transcription failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
              "error_message":f"Transcription failed: {str(e)}"
            }
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"🗑️ Cleaned up temporary audio_data file")
            except Exception as cleanup_error:
                print(f"❌ [FastAPI]  Failed to cleanup temp audio_data file: {cleanup_error}")
        print(f"🏁 ===== TRANSCRIPTION REQUEST END =====\n")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        log_level="info",
        access_log=True
    )
        
    

