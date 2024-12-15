# database_handler.py

from sqlalchemy.exc import SQLAlchemyError
from models import USDPairData
from database import AsyncSessionLocal
import logging


async def save_live_data_batch(data_batch):
    """
    Save a batch of data points to the database.
    """
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                usd_pairs = [
                    USDPairData(
                        product_id=data["product_id"],
                        time=data["time"],
                        price=data["price"],
                        volume=data["volume"],
                    )
                    for data in data_batch
                ]
                session.add_all(usd_pairs)
        logging.info(f"Inserted batch of {len(data_batch)} records into the database.")
    except SQLAlchemyError as e:
        logging.error(f"Batch database insertion error: {e}")
        # Optionally, implement retry logic or other error handling
