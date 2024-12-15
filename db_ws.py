import asyncio
import websockets
import json
import asyncpg
from datetime import datetime

# List of USD trading pairs
usd_pairs = [
    "BTC-USD", "ETH-USD", "LTC-USD"  # Add all desired USD pairs here
]

# Supabase PostgreSQL connection details
SUPABASE_DB_URL = "postgresql://postgres.mtmtioudqxvhhbpdmwcf:t7GOwVVQWqosxlpk@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

async def setup_database():
    """
    Ensure the database table exists.
    """
    conn = await asyncpg.connect(SUPABASE_DB_URL)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS USDPairData (
            id SERIAL PRIMARY KEY,
            product_id TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            price FLOAT NOT NULL,
            volume FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    await conn.close()


async def insert_data(product_id, time, price, volume):
    """
    Insert a record into the Supabase database.
    """
    conn = await asyncpg.connect(SUPABASE_DB_URL)
    try:
        # Convert timezone-aware datetime to naive datetime
        time_naive = time.replace(tzinfo=None)

        await conn.execute("""
            INSERT INTO USDPairData (product_id, time, price, volume)
            VALUES ($1, $2, $3, $4);
        """, product_id, time_naive, price, volume)
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        await conn.close()


async def subscribe():
    """
    Subscribe to Coinbase's WebSocket feed and process data.
    """
    url = "wss://ws-feed.exchange.coinbase.com"
    async with websockets.connect(url) as websocket:
        # Subscribe message
        subscribe_message = {
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": usd_pairs}]
        }
        await websocket.send(json.dumps(subscribe_message))
        print(f"Subscribed to channels for USD pairs: {usd_pairs}")

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Process ticker messages
                if data['type'] == 'ticker' and data['product_id'] in usd_pairs:
                    timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                    product_id = data['product_id']
                    price = float(data['price'])
                    volume = float(data['last_size'])

                    # Insert into Supabase database
                    await insert_data(product_id, timestamp, price, volume)

                    print(f"Stored data for {product_id} at {timestamp}: Price={price}, Volume={volume}")

            except Exception as e:
                print(f"Error: {e}")
                break

async def main():
    """
    Main entry point to setup the database and start subscribing.
    """
    await setup_database()
    while True:
        await subscribe()
        print("Reconnecting in 5 seconds...")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
