import os
import json
import copy

from anchorpy import Wallet
from anchorpy import Provider
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient

from driftpy.constants.config import configs
from driftpy.types import *
from driftpy.math.market import calculate_bid_ask_price
#MarketType, OrderType, OrderParams, PositionDirection, OrderTriggerCondition

from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION
from driftpy.constants.markets import mainnet_markets
from borsh_construct.enum import _rust_enum

@_rust_enum
class PostOnlyParams:
    NONE = constructor()
    TRY_POST_ONLY = constructor()
    MUST_POST_ONLY = constructor()

def order_print(orders: list[OrderParams], market_str=None):
    for order in orders:
        if order.price == 0:
            pricestr = '$ORACLE'
            if order.oracle_price_offset > 0:
                pricestr += ' + '+str(order.oracle_price_offset/1e6)
            else:
                pricestr += ' - '+str(abs(order.oracle_price_offset)/1e6)
        else:
            pricestr = '$' + str(order.price/1e6)

        if market_str == None:
            market_str = configs['mainnet'].markets[order.market_index].symbol

        print(str(order.direction).split('.')[-1].replace('()',''), market_str, '@', pricestr)


async def main(
    keypath, 
    env, 
    url, 
    market_name,
    base_asset_amount,
    subaccount_id,
    spread = .01,
    offset = 0,
):
    with open(os.path.expanduser(keypath), 'r') as f: secret = json.load(f) 
    kp = Keypair.from_secret_key(bytes(secret))
    print('using public key:', kp.public_key, 'subaccount=', subaccount_id)
    config = configs[env]
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)
    drift_acct = ClearingHouse.from_config(config, provider)

    is_perp  = 'PERP' in market_name.upper()
    market_type = MarketType.PERP() if is_perp else MarketType.SPOT()

    market_index = -1
    for perp_market_config in config.markets:
        if perp_market_config.symbol == market_name:
            market_index = perp_market_config.market_index
    for spot_market_config in config.banks:
        if spot_market_config.symbol == market_name:
            market_index = spot_market_config.bank_index

    default_order_params = OrderParams(
                order_type=OrderType.LIMIT(),
                market_type=market_type,
                direction=PositionDirection.LONG(),
                user_order_id=0,
                base_asset_amount=int(base_asset_amount * BASE_PRECISION),
                price=0,
                market_index=market_index,
                reduce_only=False,
                post_only=PostOnlyParams.TRY_POST_ONLY(),
                immediate_or_cancel=False,
                trigger_price=0,
                trigger_condition=OrderTriggerCondition.ABOVE(),
                oracle_price_offset=0,
                auction_duration=None,
                max_ts=None,
                auction_start_price=None,
                auction_end_price=None,
            )

    bid_order_params = copy.deepcopy(default_order_params)
    bid_order_params.direction = PositionDirection.LONG()
    bid_order_params.oracle_price_offset = int((offset - spread/2) * PRICE_PRECISION)
             
    ask_order_params = copy.deepcopy(default_order_params)
    ask_order_params.direction = PositionDirection.SHORT()
    ask_order_params.oracle_price_offset = int((offset + spread/2) * PRICE_PRECISION)

    ###### Get bid/ask ??? 
    drift_acct 



    ###

    print("order_print is below")
    order_print([bid_order_params, ask_order_params], market_name)

    perp_orders_ix = []
    spot_orders_ix = []
    print("About to place perp order")
    print("bid_order_params = ", bid_order_params)
    print("ask_order_params = ", ask_order_params)
    print("subaccount_id=", subaccount_id)
    print("*"*100)
    if is_perp:
        perp_orders_ix = [
            await drift_acct.get_place_perp_order_ix(bid_order_params, subaccount_id),
            await drift_acct.get_place_perp_order_ix(ask_order_params, subaccount_id)
            ]
    else:
        spot_orders_ix =  [
            await drift_acct.get_place_spot_order_ix(bid_order_params, subaccount_id),
            await drift_acct.get_place_spot_order_ix(ask_order_params, subaccount_id)
        ]
    print("abvout to send ixs")

    await drift_acct.send_ixs(
        [
        await drift_acct.get_cancel_orders_ix(subaccount_id),
        ] + perp_orders_ix + spot_orders_ix
    )
    print(" sent ixs")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False, default=os.environ.get('ANCHOR_WALLET'))
    parser.add_argument('--env', type=str, default='devnet')
    parser.add_argument('--amount', type=float, required=True)
    parser.add_argument('--market', type=str, required=True)
    parser.add_argument('--subaccount', type=int, required=False, default=0)
    parser.add_argument('--spread', type=float, required=False, default=.01) # $0.01
    parser.add_argument('--offset', type=float, required=False, default=0) # $0.00

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

    import asyncio
    asyncio.run(main(
        args.keypath, 
        args.env, 
        url,
        args.market, 
        args.amount,
        args.subaccount,
        args.spread,
        args.offset,
    ))



