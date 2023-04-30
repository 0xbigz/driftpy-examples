import os
import json
import copy

from anchorpy import Wallet
from anchorpy import Provider
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient

from driftpy.constants.config import configs
from driftpy.types import *
#MarketType, OrderType, OrderParams, PositionDirection, OrderTriggerCondition

from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION
from borsh_construct.enum import _rust_enum

import requests, json
import pandas as pd


URL_DEVNET_OB = "https://master.dlob.drift.trade/orders/json"
MARKET_INDEX_SOL = 0
userAccount = "ECbtn9Y34m6L89EBDMRYxhpyrFgSHEcbELgozpqAvcDY"
PKEY = "7f6d3DWGkNrKTERtW9McbBQCVbZibuHUf7CsdUjJyK1t"


@_rust_enum
class PostOnlyParams:
    NONE = constructor()
    TRY_POST_ONLY = constructor()
    MUST_POST_ONLY = constructor()




def orderbook_all(perp_market="SOL-PERP"):
    """ Queries DEVNET orderbok for specified perp market, organizes data into bid/asks sorted

    """
    try:
        r = requests.get(URL_DEVNET_OB).json()
    except:
        print(r)
        raise Exception("Unable to fetch orderbook")

    slot    = r['slot']
    oracles = r['oracles']
    orders  = r['orders']

    ## market index -> price
    market_to_oracle_map = pd.DataFrame(r['oracles']).set_index('marketIndex').to_dict()['price']

    ## Get list of outstanding orders    
    orders_all = list(map(lambda d: {"user":d['user'][:10], **d.pop('order')}, orders))
    df = pd.DataFrame(orders_all)
    df_raw = pd.DataFrame(orders_all)
    
    ## Only interested in (a) SOL-PERP (b) limit orders
    # SOL is marketINdex=0
    df = df[(df.marketIndex == MARKET_INDEX_SOL) & (df.marketType=='perp')]
    
    df["oraclePrice"] = df["marketIndex"].apply(lambda x: market_to_oracle_map.get(x, None))

    ## Convert from lamports:
    for col in ['price', 'oraclePrice', 'oraclePriceOffset']:
        df[col] = df[col].astype(int)
        df[col] /= 1e6

    for col in ['quoteAssetAmountFilled']:
        df[col] = df[col].astype(int)
        df[col] /= 1e6 

    for col in ['baseAssetAmount', 'baseAssetAmountFilled']:
        df[col] = df[col].astype(int)
        df[col] /= 1e9
        
    
    bid_df = df[(df.direction == 'long') & (df.orderType == 'limit')]
    ask_df = df[(df.direction == 'short')   &( df.orderType == 'limit')]
    
    ## Fixed prices are ok
    ## Floating prices need to be marked to a fixed price according to oracle price    
    bid_float = bid_df.loc[bid_df.price == 0] # inv: 0 price <-> floating order .w.r.t oracle 
    ask_float = ask_df.loc[ask_df.price == 0] #
    
    bid_df.loc[bid_df.price == 0, "price"] = bid_float["oraclePrice"] + bid_float["oraclePriceOffset"]
    ask_df.loc[ask_df.price == 0, "price"] = ask_float["oraclePrice"] + ask_float["oraclePriceOffset"]
    
    
    bid_df = bid_df.sort_values(['price'], ascending=False)
    ask_df = ask_df.sort_values(['price'])
    
    bid_df = bid_df.reset_index(drop=True)
    ask_df = ask_df.reset_index(drop=True)
    
    return df_raw, bid_df, ask_df


def orderbook_expanded(perp_market="SOL-PERP"):
    """
        Simplified View of Orderbook, by order
    """
    _, bid, ask = orderbook_all(perp_market)
    
    
    
    bid_simple = bid[["price", "baseAssetAmount", "baseAssetAmountFilled", "postOnly", "oraclePriceOffset", "oraclePrice"]]
    ask_simple = ask[["price", "baseAssetAmount", "baseAssetAmountFilled", "postOnly", "oraclePriceOffset", "oraclePrice"]]

    #### SOME ARE PARTIALLY FILLED
    bid_simple["qty"] = bid_simple["baseAssetAmount"] - bid_simple["baseAssetAmountFilled"]
    ask_simple["qty"] = ask_simple["baseAssetAmount"] - ask_simple["baseAssetAmountFilled"]
    

    return bid_simple, ask_simple

def get_dlob(perp_market="SOL-PERP"):
    """
        L2 orderbook view (only price / quantities)
    """
    bid, ask = orderbook_expanded(perp_market)
    
    bid_ob = bid[["price", "qty"]].groupby("price", sort=False).sum()
    ask_ob = ask[["price", "qty"]].groupby("price", sort=False).sum()
    return bid_ob, ask_ob
    

def dlob_metrics(bid_ob, ask_ob, oraclePrice=None):
    """
        Returns various metrics on L2 orderbook, either in absolute terms or scaled by oraclePrice 
    """
    tob = bid_ob.iloc[:3].reset_index()
    toa = ask_ob.iloc[:3].reset_index()

    if(oraclePrice is None):
        return {"mid"        : (bid_ob.index.max() + ask_ob.index.min() ) / 2.0,
                "bid"        :  bid_ob.index.max(),
                "ask"        :  ask_ob.index.min(),
                "bid_avg"    : (tob["price"] * tob["qty"]).sum() / tob["qty"].sum(),
                "ask_avg"    : (toa["price"] * toa["qty"]).sum() / toa["qty"].sum()

        }
    else:
        return {"mid"       : (bid_ob.index.max() + ask_ob.index.min() ) / 2.0 - oraclePrice,
                "bid"       :  bid_ob.index.max() - oraclePrice,
                "ask"       :  ask_ob.index.min() - oraclePrice,
                "bid_avg"   : (tob["price"] * tob["qty"]).sum() / tob["qty"].sum() - oraclePrice,
                "ask_avg"   : (toa["price"] * toa["qty"]).sum() / toa["qty"].sum() - oraclePrice,
        }
    

async def get_solana_perp_oracle():
    """ return SOL-PERP oracle """
    with open(os.path.expanduser("/home/sjl/.config/solana/key.json"), 'r') as f: secret = json.load(f)
    kp = Keypair.from_secret_key(bytes(secret))
    subaccount = 0
    print('using public key:', kp.public_key, " subaccount = ", subaccount)

    env = 'devnet'
    config = configs[env]
    wallet = Wallet(kp) 
    connection = AsyncClient(config.default_http)
    provider = Provider(connection, wallet)

    ch = ClearingHouse.from_config(config, provider)
    chu = ClearingHouseUser(ch, authority=ch, subaccount_id=subaccount)
    perp_market = await chu.get_perp_market(0)
    oracle = (await chu.get_perp_oracle_data(perp_market)).price / PRICE_PRECISION

    return oracle





bid_offsets = [-10, -12, -15]
bid_qtys    = [1.1, 2.2, 1.1]

ask_offsets = [10, 12, 15]
ask_qtys    = [1.1, 2.2, 1.1]

# await construct_and_place(bid_offsets, bid_qtys, ask_offsets, ask_qtys)






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

    order_print([bid_order_params, ask_order_params], market_name)

    perp_orders_ix = []
    spot_orders_ix = []
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

    await drift_acct.send_ixs(
        [
        await drift_acct.get_cancel_orders_ix(subaccount_id),
        ] + perp_orders_ix + spot_orders_ix
    )

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



