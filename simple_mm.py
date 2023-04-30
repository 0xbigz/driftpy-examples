"""
Simple Market Maker using driftpy.

Functionalities:
    + Fetches list of outstanding limit orders, cleans it up into a simple orderbook
    + Makes market incorporating buy/sell pressure from orderbook and current position
"""

import os, json, copy, requests

from anchorpy import Wallet, Provider
# from anchorpy import Provider
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient

from driftpy.constants.config import configs
from driftpy.types import *

from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION
from borsh_construct.enum import _rust_enum

import pandas as pd
import numpy as np
import asyncio

pd.options.mode.chained_assignment = None


URL_DEVNET_OB = "https://master.dlob.drift.trade/orders/json"
MARKET_INDEX_SOL = 0
userAccount = "ECbtn9Y34m6L89EBDMRYxhpyrFgSHEcbELgozpqAvcDY"
PKEY = "7f6d3DWGkNrKTERtW9McbBQCVbZibuHUf7CsdUjJyK1t"


@_rust_enum
class PostOnlyParams:
    NONE = constructor()
    TRY_POST_ONLY = constructor()
    MUST_POST_ONLY = constructor()

####### Orderbook Related ##########

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
    # Look at weighted average of the top of the book .. can signal short term volatiltiy
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
    

####### Placing Orders  ##########

def mm_order_single(order_offset, order_direction, order_qty):
    
    return OrderParams(
        order_type=OrderType.LIMIT(),
        market_type=MarketType.PERP(),
        direction=order_direction,
        user_order_id=0,
        base_asset_amount = int(order_qty * BASE_PRECISION),
        price=0,
        market_index=0,
        reduce_only=False,
        post_only=PostOnlyParams.TRY_POST_ONLY(),
        immediate_or_cancel=False,
        trigger_price=0,
        trigger_condition=OrderTriggerCondition.ABOVE(),
        oracle_price_offset= int(order_offset * PRICE_PRECISION),
        auction_duration=None,
        max_ts=None,
        auction_start_price=None,
        auction_end_price=None,
    )

# For testing
# async def get_solana_perp_oracle(chu):
#     """ return SOL-PERP oracle """
#     perp_market = await chu.get_perp_market(0)
#     oracle = (await chu.get_perp_oracle_data(perp_market)).price / PRICE_PRECISION
# 
#     return oracle

async def construct_and_place(drift_acct, bid_offsets, bid_qtys, ask_offsets, ask_qtys, cancel_existing=True):
    """
        Bundles offsets / quantities for bids / asks into an array of orders
        and places order
        
        cancels existing orders before placing
    """
    all_orders = []
    if(cancel_existing):
        all_orders.append(await drift_acct.get_cancel_orders_ix(0))
    for bo, bq in zip(bid_offsets, bid_qtys):
        if(bq > 0):
            temp_order = mm_order_single(bo, PositionDirection.LONG(), bq)
            all_orders.append(
                await drift_acct.get_place_perp_order_ix(temp_order, 0)
            )
    for ao, aq in zip(ask_offsets, ask_qtys):
        if(aq > 0):
            all_orders.append(
                await drift_acct.get_place_perp_order_ix(mm_order_single(ao, PositionDirection.SHORT(), aq), 0)
            )
    ret = await drift_acct.send_ixs(all_orders)
    return ret

async def main(
        keypath, 
        env, 
        url, 
        subaccount_id,
        file,
    ):
    """
    Main event loop. 

    Set hyperparmaeters, and then:
    
    LOOP:
        (0) Pull Stats
        (1) Get position
        (2) Get orderbook Data
        (3) Compute "fair price" based on orderbook
        (4) Signals to shrink/expand spread
        (5) Cancel existing orders + place new orders

    """
    with open(os.path.expanduser(keypath), 'r') as f: secret = json.load(f) 
    kp = Keypair.from_secret_key(bytes(secret))
    print('using public key:', kp.public_key, 'subaccount=', subaccount_id)
    config = configs[env]
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)
    drift_acct = ClearingHouse.from_config(config, provider)

    chu        = ClearingHouseUser(drift_acct, use_cache=True)

    ## Margin is as maintained on the exchange (can be adjusted with deposits)
    TARGET_MAX_SIZE = 100
    LEVERAGE_LIMIT  = 2
    AGGRESSION      = 50e-4 # 50 bips
    SLEEP_INTERVAL  = 10 # In seconds

    async def get_stats(chu):
        perp_market = await chu.get_perp_market(0)
        return {
            "unrealized_pnl" : await chu.get_unrealized_pnl() / PRICE_PRECISION,
            "oracle_price"   : (await chu.get_perp_oracle_data(perp_market)).price / PRICE_PRECISION,
            "curr_leverage"  : ( await chu.get_leverage() ) / 10_000,  # h/t bigz,
            "SOL_PERP_pos"   : (await chu.get_user_position(0))
        }
    
    ##############
    ## Main event loop
    ##############
    history = {}
    epoch = 0 # So it doesn't run infinitely while testing 
    while epoch < 5:
        
        print("Epoch: ", epoch)
        epoch += 1
        
        ## ---------- Pull stats  --------------
        await chu.set_cache()
        stats = await get_stats(chu)
        print(f"> upnl = {stats['unrealized_pnl']}")
        print(f"> oracle = {stats['oracle_price']}")
        print(f"leverage = {stats['curr_leverage']}")
        
        # (1) Current SOL-PERP position
        pp = stats['SOL_PERP_pos'] #await chu.get_user_position(0)
        current_pos = pp.base_asset_amount / 1e9  
        print(f"> current pos = {current_pos}")
        
        # (2): Get the current order book (GET Request)
        bid_ob, ask_ob = get_dlob()

        # (3): Calculate the fair price based on the order book
        metrics = dlob_metrics(bid_ob, ask_ob, stats['oracle_price'])
        print(f"> dlob metrics = {metrics}")
        
        ## PART 2: Prepare new bids / ask price points
        
        #### Signals --- very simple ----
        
        ## check if overleveraged -- here we need to reduce position immediately
        SIGNAL_over_leveraged = True if stats['curr_leverage'] > 0.8 * LEVERAGE_LIMIT else False
        
        ## Backoff position if needed
        SIGNAL_engage_sell_pressure  = True if current_pos > TARGET_MAX_SIZE  * 0.8 else False # Hyperparemeter 
        SIGNAL_engage_buy_pressure   = True if current_pos < -TARGET_MAX_SIZE * 0.8 else False # Hyperparemeter 
        
        ## no bid / no ask
        SIGNAL_no_bid = True if SIGNAL_over_leveraged and SIGNAL_engage_sell_pressure else False
        SIGNAL_no_ask = True if SIGNAL_over_leveraged and SIGNAL_engage_buy_pressure else False
        
            
        ## Adjustment on buy / sell spread due to buying and selling pressure in orderbook
        SIGNAL_order_book_adj = False
        if(metrics['bid_avg'] <= stats['oracle_price'] <= metrics['ask_avg']):
            SIGNAL_order_book_adj = True
            ask_to_bid_factor = (metrics['ask_avg'] - stats['oracle_price']) / (stats['oracle_price'] - metrics['bid_avg'])
            if(ask_to_bid_factor > 1):
                ask_adj_factor = 1 + np.ln(ask_to_bid_factor)
                bid_adj_factor = 1
            elif(ask_to_bid_factor < 1):
                bid_adj_factor = 1 + np.ln(ask_to_bid_factor)
                ask_adj_factor = 1
        
        # ---------------- BID SIDE -----------------
        ## Compute total size
        bid_target_pos     = max(0, TARGET_MAX_SIZE - current_pos)
        bid_offset_width   = AGGRESSION 
        
        ## Apply signals to stretch, etc
        if(SIGNAL_order_book_adj):
            bid_offset_width *= bid_adj_factor
        
        if(SIGNAL_engage_buy_pressure):
            bid_offset_width *= 0.5 #Hyperparemeter

        # Compute bid offsets (w.r.t oracle)  
        if(not SIGNAL_no_bid):
            bid_offsets= np.linspace(0, bid_offset_width, 4)[1:]      # e.g. 0.05, 0.1, 0.15
            bid_qts    = bid_target_pos * np.array([0.25, 0.5, 0.25]) # "parabolic" weighting for now
            
            print(f"> bid_target_pos = {bid_target_pos}")
            print(f"> bid_offsets = {bid_offsets}")
            print(f"> bid_qts = {bid_qts}")
        else:
            print("No bid, due to overleveraged position")
        
        # ---------------- ASK SIDE -----------------
        ask_target_pos   = max(0, TARGET_MAX_SIZE + current_pos)
        ask_offset_width = AGGRESSION * 1.01 # Bias due to funding payments
        
        ## Apply signals to stretch, etc
        if(SIGNAL_order_book_adj):
            ask_offset_width *= ask_adj_factor

        if(SIGNAL_engage_sell_pressure):
            ask_offset_width *= 0.5
        
        if(not SIGNAL_no_ask):
            ask_offsets      = np.linspace(0, ask_offset_width, 4)[1:]
            ask_qts          = ask_target_pos * np.array([0.25, 0.5, 0.25])

            print(f"> ask_target_pos = {ask_target_pos}")
            print(f"> ask_prices = {ask_offsets}")
            print(f"> ask_qts = {ask_qts}")
        else:
            print("No ask, due to overleveraged position")
        
        ## ---------- Send order  --------------
        ret = await construct_and_place(drift_acct, bid_offsets, bid_qts, ask_offsets, ask_qts)
        print(ret)

        history[epoch] = {
            "oraclePrice" : stats['oracle_price'],
            "currLeverage": stats["curr_leverage"],
            "ob_mid"         : metrics['mid'],
            "ob_bid"         : metrics['bid'],
            "ob_ask"         : metrics['ask'],
            "bid_offsets"    : bid_offsets,
            "bid_qts"        : bid_qts,
            "ask_offsets"    : ask_offsets,
            "ask_qts"        : ask_qts,
            "txn"            : ret,
        }
        
        ## ---------- Logging --------------    
        # Step 6: Wait for 1 second before repeating the loop
        await asyncio.sleep(SLEEP_INTERVAL)

    ## Save data for analysis / tuning parmaeters
    history_df = pd.DataFrame.from_dict(history)
    history_df.to_pickle(file)

    return history

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False, default=os.environ.get('ANCHOR_WALLET'))
    parser.add_argument('--env', type=str, default='devnet')
    parser.add_argument('--file', type=str, default="history.h5")
    parser.add_argument('--subaccount', type=int, required=False, default=0)

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

    asyncio.run(main(
        args.keypath, 
        args.env, 
        url,
        args.subaccount,
        args.file,
    ))


