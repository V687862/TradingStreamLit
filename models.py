# models.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func

Base = declarative_base()

class USDPairData(Base):
    __tablename__ = 'USDPairData'

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(20), nullable=False)
    time = Column(DateTime(timezone=True), nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    createdAt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
