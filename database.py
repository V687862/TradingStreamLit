# database.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Uses asyncpg without SSL
ALEMBIC_DATABASE_URL = os.getenv(
    "ALEMBIC_DATABASE_URL"
)  # For Alembic migrations without SSL

# Create asynchronous engine without SSL
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for verbose SQL logs
    future=True,
    # Removed connect_args since SSL is not used
)

# Create a configured "Session" class
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
