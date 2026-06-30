"""
FastAPI - Sales Agent SK API

Endpoints:
  GET  /                    → health check
  GET  /ventas/resumen      → resumen general sin agente
  POST /chat                → conversación con memoria in-session
  POST /chat/persistent     → conversación con memoria persistente
  GET  /memory/{session_id} → ver historial
  DELETE /memory/{session_id} → limpiar historial
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent_sk import run_agent
from app.tools import resumen_general
from app.memory import in_session_memory, persistent_memory

app = FastAPI(
    title="Sales Agent SK API",
    description="Agente de análisis de ventas con Semantic Kernel + Groq + FastAPI",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    session_id: str
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user-123",
                "question": "¿Quién vendió más este trimestre?"
            }
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "sales-agent-sk", "framework": "Semantic Kernel"}


@app.get("/ventas/resumen", tags=["Ventas"])
def get_resumen():
    return resumen_general()


@app.post("/chat", response_model=ChatResponse, tags=["Agente - Memoria In-Session"])
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    history = in_session_memory.get_history(request.session_id)

    try:
        answer = run_agent(question=request.question, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    in_session_memory.add_message(request.session_id, "user", request.question)
    in_session_memory.add_message(request.session_id, "assistant", answer)

    return ChatResponse(session_id=request.session_id, question=request.question, answer=answer)


@app.post("/chat/persistent", response_model=ChatResponse, tags=["Agente - Memoria Persistente"])
def chat_persistent(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    history = persistent_memory.get_history(request.session_id)

    try:
        answer = run_agent(question=request.question, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    persistent_memory.add_message(request.session_id, "user", request.question)
    persistent_memory.add_message(request.session_id, "assistant", answer)

    return ChatResponse(session_id=request.session_id, question=request.question, answer=answer)


@app.get("/memory/{session_id}", tags=["Memoria"])
def get_memory(session_id: str, persistent: bool = False):
    if persistent:
        history = persistent_memory.get_history_with_timestamps(session_id)
    else:
        history = in_session_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history, "total": len(history)}


@app.delete("/memory/{session_id}", tags=["Memoria"])
def clear_memory(session_id: str, persistent: bool = False):
    if persistent:
        persistent_memory.clear(session_id)
    else:
        in_session_memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
