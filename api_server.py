import asyncio
import time
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database import Database
from api_routes import router as api_router
from dependencies import get_db, get_start_time, get_request_count, get_keep_alive_enabled, get_anti_detection
from api_utils import GeminiAntiDetectionInjector, keep_alive_ping
from api_services import record_hourly_health_check, auto_cleanup_failed_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables
request_count = 0
start_time = time.time()
scheduler = None
keep_alive_enabled = os.getenv('ENABLE_KEEP_ALIVE', 'false').lower() == 'true'

# Initialize database and anti-detection injector
db = Database()
anti_detection = GeminiAntiDetectionInjector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    global scheduler, keep_alive_enabled
    logger.info("🚀 Service starting up...")

    # Initialize and start the scheduler on startup if keep-alive is enabled
    if keep_alive_enabled:
        scheduler = AsyncIOScheduler()
        
        keep_alive_interval = int(os.getenv('KEEP_ALIVE_INTERVAL', '10'))
        scheduler.add_job(
            keep_alive_ping, 'interval', minutes=keep_alive_interval,
            id='keep_alive', max_instances=1, coalesce=True, misfire_grace_time=30
        )
        
        scheduler.add_job(
            record_hourly_health_check, 'interval', hours=1,
            id='hourly_health_check', max_instances=1, coalesce=True
        )
        
        scheduler.add_job(
            auto_cleanup_failed_keys, 'cron', hour=2, minute=0,
            id='daily_cleanup', max_instances=1, coalesce=True
        )
        
        scheduler.start()
        logger.info(f"🟢 Keep-alive enabled (interval: {keep_alive_interval} minutes)")
        
        # Perform an initial ping
        asyncio.create_task(keep_alive_ping())
    else:
        logger.info("🔴 Keep-alive disabled")

    yield

    # Shutdown the scheduler on application exit
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down.")
    logger.info("👋 Service shutting down...")

# Create FastAPI app instance
app = FastAPI(
    title="Gemini API Proxy",
    version="1.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Dependency overrides
def _get_db():
    return db

def _get_start_time():
    return start_time

def _get_request_count():
    return request_count

def _get_keep_alive_enabled():
    return keep_alive_enabled

def _get_anti_detection():
    return anti_detection

app.dependency_overrides[get_db] = _get_db
app.dependency_overrides[get_start_time] = _get_start_time
app.dependency_overrides[get_request_count] = _get_request_count
app.dependency_overrides[get_keep_alive_enabled] = _get_keep_alive_enabled
app.dependency_overrides[get_anti_detection] = _get_anti_detection

logger.info("✅ FastAPI app initialized with routes and dependencies.")
