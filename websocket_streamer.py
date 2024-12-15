# stream.py

import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime
from database_handler import save_live_data_batch
from historical_handler import resample_data
import logging
import os
from dotenv import load_dotenv
import signal

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to capture all logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define constants
PRODUCT_IDS = [
    "XRP-USD",
    "BTC-USD",
    "ETH-USD",
    "HBAR-USD",
    "USDT-USD",
    "SOL-USD",
    "DOGE-USD",
    "XLM-USD",
    "SUI-USD",
    "ADA-USD",
    "LINK-USD",
    "SHIB-USD",
    "XYO-USD",
    "JASMY-USD",
    "LTC-USD",
    "AVAX-USD",
    "ONDO-USD",
    "AMP-USD",
    "VET-USD",
    "ALGO-USD",
    "BONK-USD",
    "PEPE-USD",
    "CRV-USD",
    "SEI-USD",
    "BCH-USD",
    "DOT-USD",
    "FET-USD",
    "XTZ-USD",
    "AAVE-USD",
    "ICP-USD",
    "UNI-USD",
    "TIA-USD",
    "EOS-USD",
    "FIL-USD",
    "ETC-USD",
    "NEAR-USD",
    "STX-USD",
    "APE-USD",
    "INJ-USD",
    "ARB-USD",
    "APT-USD",
    "SAND-USD",
    "SHPING-USD",
    "GRT-USD",
    "ENS-USD",
    "SWFTC-USD",
    "RNDR-USD",
    "MATIC-USD",
    "QNT-USD",
    "AERO-USD",
    "LDO-USD",
    "POL-USD",
    "WIF-USD",
    "MOBILE-USD",
    "BIGTIME-USD",
    "ATOM-USD",
    "OP-USD",
    "SUPER-USD",
    "MKR-USD",
    "RARI-USD",
    "LRC-USD",
    "RENDER-USD",
    "AIOZ-USD",
    "GST-USD",
    "CRO-USD",
    "CVX-USD",
    "COMP-USD",
    "ROSE-USD",
    "MANA-USD",
    "FLOKI-USD",
    "IMX-USD",
    "XCN-USD",
    "ACH-USD",
    "SPA-USD",
    "TRU-USD",
    "IOTX-USD",
    "JTO-USD",
    "ZRX-USD",
    "WELL-USD",
    "ANKR-USD",
    "SUKU-USD",
    "DASH-USD",
    "LCX-USD",
    "FLR-USD",
    "ASM-USD",
    "VTHO-USD",
    "DEGEN-USD",
    "SPELL-USD",
    "TRB-USD",
    "EGLD-USD",
    "YFI-USD",
    "SUSHI-USD",
    "COTI-USD",
    "HNT-USD",
    "SKL-USD",
    "PRIME-USD",
    "AXS-USD",
    "ZETA-USD",
    "MPL-USD",
    "SYN-USD",
    "ZEC-USD",
    "OXT-USD",
    "BLUR-USD",
    "CHZ-USD",
    "AKT-USD",
    "EIGEN-USD",
    "GFI-USD",
    "AXL-USD",
    "PRO-USD",
    "SWELL-USD",
    "VELO-USD",
    "FOX-USD",
    "KARRAT-USD",
    "DAI-USD",
    "CGLD-USD",
    "QI-USD",
    "SNX-USD",
    "KSM-USD",
    "METIS-USD",
    "A8-USD",
    "BAT-USD",
    "DRIFT-USD",
    "FLOW-USD",
    "ARKM-USD",
    "ALEO-USD",
    "STORJ-USD",
    "MINA-USD",
    "DNT-USD",
    "GMT-USD",
    "AURORA-USD",
    "BOBA-USD",
    "LPT-USD",
    "GLM-USD",
    "SEAM-USD",
    "RARE-USD",
    "1INCH-USD",
    "CORECHAIN-USD",
    "PNG-USD",
    "ORCA-USD",
    "VARA-USD",
    "IO-USD",
    "CBETH-USD",
    "ABT-USD",
    "HONEY-USD",
    "CTSI-USD",
    "NCT-USD",
    "STRK-USD",
    "HIGH-USD",
    "CLV-USD",
    "API3-USD",
    "AUCTION-USD",
    "KRL-USD",
    "HOPR-USD",
    "ZRO-USD",
    "KNC-USD",
    "BLZ-USD",
    "ILV-USD",
    "BLAST-USD",
    "GODS-USD",
    "KAVA-USD",
    "PRQ-USD",
    "UMA-USD",
    "ARPA-USD",
    "FIDA-USD",
    "LRDS-USD",
    "TRAC-USD",
    "PERP-USD",
    "OCEAN-USD",
    "AVT-USD",
    "HFT-USD",
    "TNSR-USD",
    "BICO-USD",
    "MATH-USD",
    "BTRST-USD",
    "BIT-USD",
    "NKN-USD",
    "FARM-USD",
    "CVC-USD",
    "ORN-USD",
    "DIA-USD",
    "POND-USD",
    "00-USD",
    "OGN-USD",
    "BAND-USD",
    "ALCX-USD",
    "RPL-USD",
    "ALEPH-USD",
    "RONIN-USD",
    "NMR-USD",
    "SAFE-USD",
    "ACS-USD",
    "MUSE-USD",
    "POLS-USD",
    "AUDIO-USD",
    "AGLD-USD",
    "PIRATE-USD",
    "WCFG-USD",
    "MAGIC-USD",
    "MASK-USD",
    "BAL-USD",
    "SD-USD",
    "RLC-USD",
    "PYR-USD",
    "AERGO-USD",
    "PLU-USD",
    "POWR-USD",
    "FORT-USD",
    "ZEN-USD",
    "OMNI-USD",
    "INDEX-USD",
    "ERN-USD",
    "MDT-USD",
    "OSMO-USD",
    "CELR-USD",
    "ZK-USD",
    "REQ-USD",
    "DYP-USD",
    "NEON-USD",
    "LQTY-USD",
    "CTX-USD",
    "MLN-USD",
    "AST-USD",
    "MSOL-USD",
    "TVK-USD",
    "WBTC-USD",
    "T-USD",
    "ALICE-USD",
    "BADGER-USD",
    "FX-USD",
    "LOKA-USD",
    "SHDW-USD",
    "RAD-USD",
    "DIMO-USD",
    "ELA-USD",
    "IDEX-USD",
    "VOXEL-USD",
    "RBN-USD",
    "FORTH-USD",
    "DAR-USD",
    "BNT-USD",
    "MNDE-USD",
    "GTC-USD",
    "STG-USD",
    "GNO-USD",
    "COW-USD",
    "GHST-USD",
    "PYUSD-USD",
    "C98-USD",
    "ZETACHAIN-USD",
    "PUNDIX-USD",
    "GAL-USD",
    "ACX-USD",
    "G-USD",
    "TIME-USD",
    "LIT-USD",
    "MEDIA-USD",
    "INV-USD",
    "FIS-USD",
    "DEXT-USD",
    "WAXL-USD",
    "GUSD-USD",
    "GYEN-USD",
    "LSETH-USD",
    "PAX-USD",
    "MOODENG-USD"
]
# Adjust as needed
BATCH_SIZE = 100  # Number of records to accumulate before inserting
RESAMPLING_PERIODS = ["5min", "1h", "1D"]
OUTPUT_DIR = "data/historical_data"

# Flag to control running state
running = True

def shutdown_handler(signum, frame):
    global running
    logging.info("Shutdown signal received. Terminating gracefully...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

def validate_message(message):
    """
    Validate incoming ticker message.
    Returns:
        dict or None: Parsed data if valid, else None.
    """
    required_fields = ["product_id", "time", "price", "last_size"]
    if not all(field in message for field in required_fields):
        logging.warning(f"Missing required fields in message: {message}")
        return None

    try:
        parsed_data = {
            "product_id": message["product_id"],
            "time": message["time"],  # Keep as string for resampling function to handle
            "price": float(message["price"]),
            "volume": float(message["last_size"]),
        }
    except (ValueError, TypeError) as e:
        logging.warning(f"Error parsing message: {e}. Message: {message}")
        return None

    if not parsed_data["time"]:
        logging.warning(f"Invalid 'time' in message: {message}")
        return None

    return parsed_data

async def stream_live_data():
    url = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1  # Start with 1 second

    data_buffer = []  # Buffer to accumulate data for batch inserts

    while running:
        try:
            async with websockets.connect(url) as websocket:
                subscribe_message = {
                    "type": "subscribe",
                    "channels": [{
                        "name": "ticker",
                        "product_ids": PRODUCT_IDS
                    }]
                }
                await websocket.send(json.dumps(subscribe_message))
                logging.info(f"Subscribed to channels for {len(PRODUCT_IDS)} products.")

                while running:
                    try:
                        response = await websocket.recv()
                        message = json.loads(response)

                        if message.get("type") == "ticker":
                            parsed_data = validate_message(message)
                            if parsed_data:
                                data_buffer.append(parsed_data)
                                logging.debug(f"Appended data: {parsed_data}")

                                # Check if buffer has reached BATCH_SIZE
                                if len(data_buffer) >= BATCH_SIZE:
                                    logging.info(f"Buffer reached {BATCH_SIZE} records. Processing batch.")

                                    # Save batch to database
                                    await save_live_data_batch(data_buffer)

                                    # Convert buffer to DataFrame for resampling
                                    df_buffer = pd.DataFrame(data_buffer)

                                    # Log DataFrame structure and content
                                    logging.debug(f"DataFrame columns: {df_buffer.columns.tolist()}")
                                    logging.debug(f"DataFrame head:\n{df_buffer.head()}")

                                    # Resample data concurrently for all periods
                                    resample_tasks = [
                                        asyncio.to_thread(resample_data, df_buffer, period, OUTPUT_DIR)
                                        for period in RESAMPLING_PERIODS
                                    ]
                                    await asyncio.gather(*resample_tasks)

                                    # Clear the buffer
                                    data_buffer.clear()

                            else:
                                logging.debug("Skipped invalid or incomplete message.")

                    except websockets.ConnectionClosed:
                        logging.warning("WebSocket connection closed. Attempting to reconnect...")
                        break  # Exit inner loop to reconnect
                    except Exception as e:
                        logging.error(f"Error receiving data: {e}")
                        continue

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

        if running:
            logging.info(f"Reconnecting in {backoff} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)  # Exponential backoff up to 60 seconds

    # After exiting the loop, process any remaining data
    if data_buffer:
        try:
            logging.info(f"Processing remaining {len(data_buffer)} records before shutdown.")

            await save_live_data_batch(data_buffer)
            df_buffer = pd.DataFrame(data_buffer)

            # Log DataFrame structure and content
            logging.debug(f"DataFrame columns: {df_buffer.columns.tolist()}")
            logging.debug(f"DataFrame head:\n{df_buffer.head()}")

            # Resample data concurrently for all periods
            resample_tasks = [
                asyncio.to_thread(resample_data, df_buffer, period, OUTPUT_DIR)
                for period in RESAMPLING_PERIODS
            ]
            await asyncio.gather(*resample_tasks)

            logging.info("Processed remaining data before shutdown.")
        except Exception as e:
            logging.error(f"Error processing remaining data: {e}")

    logging.info("Shutdown complete.")

# Run the async function
if __name__ == "__main__":
    try:
        asyncio.run(stream_live_data())
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        logging.info("Application has exited.")