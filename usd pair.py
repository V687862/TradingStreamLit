import os
import traceback
from coinbase.rest import RESTClient

def create_client():
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    if not api_key or not api_secret:
        raise EnvironmentError("Missing API credentials. Please set COINBASE_API_KEY and COINBASE_API_SECRET.")
    return RESTClient(api_key=api_key, api_secret=api_secret)

def get_tradeable_usd_pairs(client):
    try:
        # Fetch all products
        products_response = client.get_products()
        products = products_response.products  # Access products directly

        # Debugging: Print the structure of one product
        if products:
            print("Example Product:", vars(products[0]))

        # Filter for tradeable pairs where the quote currency is USD
        tradeable_usd_pairs = [
            product.product_id
            for product in products
            if product.quote_currency_id == 'USD' and not product.trading_disabled
        ]
        return tradeable_usd_pairs
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return []

def format_tradeable_pairs(tradeable_pairs):
    # Comma-separated format
    comma_separated = ', '.join(f'"{pair}"' for pair in tradeable_pairs)
    print("Comma-separated format:")
    print(comma_separated)

    # Line-separated format
    line_separated = '\n'.join(tradeable_pairs)
    print("\nLine-separated format:")
    print(line_separated)

def save_tradeable_pairs_to_file(tradeable_pairs, filename="tradeable_usd_pairs.txt"):
    with open(filename, 'w') as file:
        file.write('\n'.join(tradeable_pairs))
    print(f"Tradeable pairs saved to {filename}")

if __name__ == '__main__':
    # Initialize the REST client
    client = create_client()

    # Get tradeable pairs
    tradeable_pairs = get_tradeable_usd_pairs(client)

    # Format and print them
    format_tradeable_pairs(tradeable_pairs)

    # Save to a file
    save_tradeable_pairs_to_file(tradeable_pairs)
