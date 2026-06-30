"""
Agente de análisis de ventas - Semantic Kernel + Azure OpenAI

Diferencias clave con LangGraph:
- No hay grafo manual — SK maneja el loop automáticamente
- Las tools se agrupan en un Plugin (clase)
- FunctionChoiceBehavior.Auto() reemplaza el edge condicional
- ChatHistory reemplaza el AgentState

Flujo interno de SK:
  [START] → LLM decide → ejecuta función → LLM decide → ... → respuesta final
  (igual que LangGraph pero SK lo maneja internamente)

Nota de migración: este archivo originalmente usaba Groq via el conector
OpenAIChatCompletion (API compatible con OpenAI). La migración a Azure OpenAI
solo cambió la sección de build_kernel() — el resto del agente (plugin, tools,
manejo de historial, function calling) es exactamente el mismo código.
"""

import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory

from app.sales_plugin import SalesPlugin

load_dotenv()

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sos un asistente experto en análisis de ventas.
Tenés acceso a funciones con datos reales de ventas.
REGLA CRITICA: NUNCA digas que no tenes informacion si existe una función que puede obtenerla.
Siempre usa las funciones para responder preguntas sobre datos.
Si la pregunta usa pronombres como el, ella, ese producto, revisa el historial
e identifica a qué persona o producto se refiere, luego usa la función con ese nombre.
Responde en español con insights accionables para el negocio."""

SERVICE_ID = "azure-openai"


# ── Construcción del Kernel ───────────────────────────────────────────────────

def build_kernel() -> Kernel:
    """
    Crea y configura el Kernel de Semantic Kernel conectado a Azure OpenAI.

    Variables de entorno necesarias (.env):
        AZURE_OPENAI_ENDPOINT     -> https://<tu-recurso>.openai.azure.com/
        AZURE_OPENAI_KEY          -> Key 1 o Key 2 del recurso
        AZURE_OPENAI_DEPLOYMENT   -> nombre del deployment (ej: gpt-4o-mini)
    """
    kernel = Kernel()

    kernel.add_service(
        AzureChatCompletion(
            service_id=SERVICE_ID,
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    )

    # Registrar el plugin con todas las tools (sin cambios respecto a Groq)
    kernel.add_plugin(SalesPlugin(), plugin_name="sales")

    return kernel


# ── Función pública ───────────────────────────────────────────────────────────

async def run_agent_async(question: str, history: list[dict] | None = None) -> str:
    """
    Ejecuta el agente con una pregunta y retorna la respuesta.

    Args:
        question: Pregunta del usuario en lenguaje natural.
        history: Historial previo [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        Respuesta del agente en lenguaje natural.
    """
    kernel = build_kernel()

    # Configurar tool calling automático
    # FunctionChoiceBehavior.Auto() = el LLM decide cuándo llamar tools
    # Equivalente al edge condicional should_continue en LangGraph
    settings = OpenAIChatPromptExecutionSettings(
        service_id=SERVICE_ID,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
        temperature=0,
    )

    # Construir historial de conversación
    chat_history = ChatHistory()
    chat_history.add_system_message(SYSTEM_PROMPT)

    if history:
        for msg in history:
            if msg["role"] == "user":
                chat_history.add_user_message(msg["content"])
            else:
                chat_history.add_assistant_message(msg["content"])

    chat_history.add_user_message(question)

    # Invocar el agente
    chat_service = kernel.get_service(SERVICE_ID)
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=settings,
        kernel=kernel,
    )

    return str(response)


def run_agent(question: str, history: list[dict] | None = None) -> str:
    """
    Wrapper síncrono para usar en FastAPI.
    FastAPI puede manejar async directamente pero este wrapper
    facilita el testing y la compatibilidad.
    """
    return asyncio.run(run_agent_async(question, history))