from fastapi import APIRouter
from .health import router as health_router
from .models import router as models_router
from .chat import router as chat_router
from .messages import router as messages_router
from .management import router as management_router
from .debug import router as debug_router

main_router = APIRouter()
main_router.include_router(health_router)
main_router.include_router(models_router)
main_router.include_router(chat_router)
main_router.include_router(messages_router)
main_router.include_router(management_router)
main_router.include_router(debug_router)
