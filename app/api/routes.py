from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.services.chatbot_service import ChatbotService

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask_question(request: Request, ask_request: AskRequest):
    service = ChatbotService(request.app)
    result = service.answer(ask_request.question)
    return result
