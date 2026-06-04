from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"
    check_faithfulness: bool = False

class TopicRequest(BaseModel):
    topic: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    cache_size: int
    model: str
    embedding_model: str

class AnswerResponse(BaseModel):
    answer: str
    confidence: str
    confidence_score: float
    sources: list
    session_id: str
    rewritten_query: str
    cached: bool
    faithfulness_score: Optional[float] = None
    faithfulness_reason: Optional[str] = None

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str