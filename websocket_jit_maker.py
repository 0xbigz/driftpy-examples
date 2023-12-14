# import ccxt.async_support as ccxt
import asyncio
import websockets
import json
import httpx 

import os
import json
import copy

import sys

from anchorpy import Wallet
from anchorpy import Provider
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient

from driftpy.constants.config import configs
from driftpy.types import *
#MarketType, OrderType, OrderParams, PositionDirection, OrderTriggerCondition

from driftpy.drift_client import DriftClient
from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION
from borsh_construct.enum import _rust_enum
import os

# @dataclass
# class MakerInfo:
#     maker: Pubkey
#     order: Order

MAX_TS_BUFFER = 15 # five seconds

async def check_position(drift_acct: DriftClient):
    user = await drift_acct.get_user()
    bb = user.perp_positions[0].base_asset_amount / 1e9
    return bb


def calculate_mark_premium(perp_market: PerpMarket):
    prem1 = perp_market.amm.last_mark_price_twap5min/1e6 - perp_market.amm.historical_oracle_data.last_oracle_price_twap5min/1e6
    prem2 = perp_market.amm.last_mark_price_twap/1e6 - perp_market.amm.historical_oracle_data.last_oracle_price_twap/1e6
    return min(prem1, prem2)


def calculate_reservation_offset(inventory, max_inventory, ref_price, mark_premium):
    return (-inventory/max(abs(inventory), max_inventory)) * .0001 * ref_price + mark_premium
    # try:
    #     # rr = httpx.get('https://mainnet-beta.api.drift.trade/dlob/l3?marketIndex=0&marketType=perp&depth=1').json()
    #     rr = httpx.get('https://mainnet-beta.api.drift.trade/dlob/l2?marketIndex=0&marketType=perp&depth=1&includeVamm=True').json()
    #     bb = int(rr['bids'][0]['price'])/1e6
    #     bo = int(rr['asks'][0]['price'])/1e6
    #     bid = min(bb + .001, bid)
    #     ask = max(bo - .001, ask)
    # except Exception as e:
    #     print(e)
    #     print('ERROR: with l3 drift data')

async def do_trade(drift_acct: DriftClient, market_index, bid, ask):
    current_position = await check_position(drift_acct)
    # last_slot = drift_acct.account_subscriber.cache['state'].slot
    # last_slot = None
    # if last_slot is None:
    #     last_slot = (await drift_acct.program.provider.connection.get_slot()).value
    # print(last_slot)
    perp_market = await drift_acct.get_perp_market(market_index)
    rev_offset = calculate_reservation_offset(current_position, 
                                              1, 
                                              (bid+ask)/2, 
                                              calculate_mark_premium(perp_market)
                                              )
    bid += rev_offset
    ask += rev_offset
    print(f'Quoting Price: {bid:.3f}/{ask:.3f}')


    max_ts = None # (await drift_acct.program.provider.connection.get_block_time(last_slot)).value + MAX_TS_BUFFER
    default_order_params = OrderParams(
            order_type=OrderType.LIMIT(),
            market_type=MarketType.PERP(),
            direction=PositionDirection.LONG(),
            user_order_id=0,
            base_asset_amount=int(.1 * BASE_PRECISION),
            price=0,
            market_index=market_index,
            reduce_only=False,
            post_only=PostOnlyParams.NONE(),
            immediate_or_cancel=False,
            trigger_price=0,
            trigger_condition=OrderTriggerCondition.ABOVE(),
            oracle_price_offset=0,
            auction_duration=None,
            max_ts=max_ts,
            auction_start_price=None,
            auction_end_price=None,
        )
    bid_order_params = copy.deepcopy(default_order_params)
    bid_order_params.direction = PositionDirection.LONG()
    bid_order_params.price = int(bid * PRICE_PRECISION - 1)
             
    ask_order_params = copy.deepcopy(default_order_params)
    ask_order_params.direction = PositionDirection.SHORT()
    ask_order_params.price = int(ask * PRICE_PRECISION + 1)
    # maker_info = MakerInfo(maker=Pubkey('TODO'), order=None)
    orders = []
    # print('POSTIION =', current_position)
    if current_position > -1:
        orders.append(ask_order_params)
    if current_position < 1:
        orders.append(bid_order_params)
    # print(orders)
    instr = await drift_acct.get_place_perp_orders_ix(orders)

    await drift_acct.send_ixs(instr)

def calculate_weighted_average_price(bids, asks):
    total_bid_price = 0
    total_bid_quantity = 0
    total_ask_price = 0
    total_ask_quantity = 0

    for bid in bids:
        price, quantity = map(float, bid)
        total_bid_price += price * quantity
        total_bid_quantity += quantity

    for ask in asks:
        price, quantity = map(float, ask)
        total_ask_price += price * quantity
        total_ask_quantity += quantity

    if total_bid_quantity == 0 or total_ask_quantity == 0:
        return None, None  # Avoid division by zero

    weighted_average_bid_price = total_bid_price / total_bid_quantity
    weighted_average_ask_price = total_ask_price / total_ask_quantity

    return weighted_average_bid_price, weighted_average_ask_price

async def handle_binance_depth_feed(keypath, url, symbol, levels, update_speed):
    # Initialize the Binance exchange object
    # exchange = ccxt.binance()
    market_index = 0
    subaccount_id = 0
    env = 'mainnet'
    with open(os.path.expanduser(keypath), 'r') as f: secret = json.load(f) 
    kp: Pubkey = Keypair.from_bytes(bytes(secret))
    print('using address:', kp.pubkey(), 'subaccount =', subaccount_id)
    config = configs[env]
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)
    drift_acct = DriftClient.from_config(config, provider)
    await drift_acct.account_subscriber.cache_if_needed()

    # Subscribe to the bid/ask depth updates
    depth_channel = f'{symbol.lower()}@depth{levels}@{update_speed}'
    url = f'wss://stream.binance.com:9443/ws/{depth_channel}'
    print('Starting...')
    last_bid, last_ask = None, None

    
    while True:
        try:
            async with websockets.connect(url) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    bids = data['bids'][:3]  # Top bids
                    asks = data['asks'][:3]  # Top asks

                    # Calculate weighted average prices
                    weighted_average_bid_price, weighted_average_ask_price = calculate_weighted_average_price(bids, asks)
                    if last_bid is None:
                        last_bid = weighted_average_bid_price
                    if last_ask is None:
                        last_ask = weighted_average_ask_price
                    if weighted_average_bid_price is not None and weighted_average_ask_price is not None:
                        
                        if abs(last_ask - weighted_average_ask_price)/last_ask > .0001 or abs(last_bid - weighted_average_bid_price)/last_bid > .0001:
                            print(f'Wgt Price: {weighted_average_bid_price:.3f}/{weighted_average_ask_price:.3f}')
                            await do_trade(drift_acct, market_index, weighted_average_bid_price, weighted_average_ask_price)
                            last_bid = weighted_average_bid_price
                            last_ask = weighted_average_ask_price
                        else:
                            print('...')

        except Exception as e:
            print(f'An error occurred: {str(e)}')
            await asyncio.sleep(5)  # Wait for 5 seconds before retrying

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False, default=os.environ.get('ANCHOR_WALLET'))
    parser.add_argument('--env', type=str, default='devnet')
    parser.add_argument('--url', type=str,)
    parser.add_argument('--market', type=str, required=True, default='SOLUSDT') # todo
    parser.add_argument('--min-position', type=float, required=False, default=None) # todo
    parser.add_argument('--max-position', type=float, required=False, default=None) # todo
    parser.add_argument('--subaccount', type=int, required=False, default=0)
    parser.add_argument('--authority', type=str, required=False, default=None) # todo
    args = parser.parse_args()

    # assert(args.spread > 0, 'spread must be > $0')
    # assert(args.spread+args.offset < 2000, 'Invalid offset + spread (> $2000)')

    if args.keypath is None:
        if os.environ['ANCHOR_WALLET'] is None:
            raise NotImplementedError("need to provide keypath or set ANCHOR_WALLET")
        else:
            args.keypath = os.environ['ANCHOR_WALLET']

    if args.env == 'devnet':
        url = 'https://api.devnet.solana.com'
    elif args.env == 'mainnet':
        url = 'https://api.mainnet-beta.solana.com'
    else:
        raise NotImplementedError('only devnet/mainnet env supported')

    if args.url:
        url = args.url

    print(args.env, url)

    keypath = args.keypath
    symbol = args.market  # Replace with the desired trading pair symbol
    levels = 5  # You can change this to 5, 10, or 20
    update_speed = '100ms'  # You can change this to '1000ms' or '100ms'
    asyncio.get_event_loop().run_until_complete(handle_binance_depth_feed(keypath, url, symbol, levels, update_speed))