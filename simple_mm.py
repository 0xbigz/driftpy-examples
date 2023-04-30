"""
Simple Market Maker using driftpy.
Sameer Lal

WRITEUP BELOW:
    https://docs.google.com/document/d/1-RLxJFIUnLdCsSEhgFzScwMichheS_FlFB17M6HmmYQ/edit?usp=sharing


Setup:
    * Include the following in ~/.config/solana/key.json: <redacted, in google docs> 
        This account has already airdropped tokens on devnet and is registered w/drift with collateral
    * Set: export ANCHOR_WALLET=’~/.config/solana/key.json”
    * Run:  python3 simple_mm.py


Functionalities:
    + Fetches list of outstanding limit orders, cleans it up into a simple orderbook (L2 view)
    + Makes market incorporating buy/sell pressure from orderbook and current position, ability to add custom signals
    + Able to cancel existing orders + submit bulk orders atomically
    + Dumps statistics into data frame that can be post processed to see how market maker compares to top of the book 
"""

import os, json, copy, requests
import pandas as pd
import numpy as np
import datetime as dt
pd.options.mode.chained_assignment = None
import asyncio

from anchorpy import Wallet, Provider
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.constants.config import configs
from driftpy.types import *
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION
from borsh_construct.enum import _rust_enum

URL_DEVNET_OB    = "https://master.dlob.drift.trade/orders/json"
MARKET_INDEX_SOL = 0
userAccount      = "ECbtn9Y34m6L89EBDMRYxhpyrFgSHEcbELgozpqAvcDY"
# Use the following: https://beta.drift.trade/overview/history/tradeHistory?userAccount=7f6d3DWGkNrKTERtW9McbBQCVbZibuHUf7CsdUjJyK1t 
PKEY             = "7f6d3DWGkNrKTERtW9McbBQCVbZibuHUf7CsdUjJyK1t" 

################################################
################################################
# Market Making parameters 
TARGET_MAX_SIZE = 100
LEVERAGE_LIMIT  = 2     
AGGRESSION      = 50e-4 # 50 bips
SLEEP_INTERVAL  = 5     # In seconds
NUM_EPOCHS      = 10    # (number of iterations of loop)
################################################
################################################

@_rust_enum
class PostOnlyParams:
    NONE = constructor()
    TRY_POST_ONLY = constructor()
    MUST_POST_ONLY = constructor()


class OrderBook():
    """ Processes devnet orderbook

    Comments:
        - GET request to endpoint to get orderbook data  (most expensive step)
        - "Parse" by a market (e.g SOL-PERP) 
        - Create a friendly L2 view, by market
        - Metrics to evaluate orderbook 
        - Cache results for quick access
    
    """
    def __init__(self, url_ob=URL_DEVNET_OB):
        print(f"Fetching orderbook from {URL_DEVNET_OB}")
        try:
            r = requests.get(URL_DEVNET_OB).json()
        except:
            raise Exception("Unable to fetch orderbook")
        self.ob_data      = r
        self.ob_parsed    = {} 
        self.ob_l2_view   = {} 
    
    def refresh(self):
        try:
            r = requests.get(URL_DEVNET_OB).json()
        except:
            raise Exception("Unable to fetch orderbook")
        self.ob_data = r


    def parse_orderbook(self, perp_market="SOL-PERP", market_index=MARKET_INDEX_SOL):
        """ Parses orderbook data looking for specified market and places into a dataframe 
        """
        slot    = self.ob_data['slot']
        oracles = self.ob_data['oracles']
        orders  = self.ob_data['orders']

        market_to_oracle_map = pd.DataFrame(self.ob_data['oracles']).set_index('marketIndex').to_dict()['price']

        ## Get list of outstanding orders    
        orders_all = list(map(lambda d: {"user":d['user'], **d.pop('order')}, orders))
        df         = pd.DataFrame(orders_all)

        # Below is returned for strictly testing  / debugging purposes
        df_raw     = pd.DataFrame(orders_all)
    
        ## Orderbook contains many orders, we are only interested in ___-PERP markets
        df = df[(df.marketIndex == market_index) & (df.marketType=='perp')]
        
        df["oraclePrice"] = df["marketIndex"].apply(lambda x: market_to_oracle_map.get(x, None))

        ## Convert from lamports / various precisions:
        for col in ['price', 'oraclePrice', 'oraclePriceOffset']:
            df[col] = df[col].astype(int)
            df[col] /= PRICE_PRECISION 

        for col in ['quoteAssetAmountFilled']:
            df[col] = df[col].astype(int)
            df[col] /= PRICE_PRECISION # Taken from example script, need to verify downstream

        for col in ['baseAssetAmount', 'baseAssetAmountFilled']:
            df[col] = df[col].astype(int)
            df[col] /= BASE_PRECISION 
        
        ## Split up bids/asks and filter for only limit orders 
        bid_df = df[(df.direction == 'long')  & (df.orderType == 'limit')]
        ask_df = df[(df.direction == 'short') & (df.orderType == 'limit')]
        
        ## Fixed prices are ok price wise.
        ## Floating prices need to be marked to a fixed price according to oracle price    
        bid_float = bid_df.loc[bid_df.price == 0] # inv: 0 price <-> floating order .w.r.t oracle 
        ask_float = ask_df.loc[ask_df.price == 0] #
        
        bid_df.loc[bid_df.price == 0, "price"] = bid_float["oraclePrice"] + bid_float["oraclePriceOffset"]
        ask_df.loc[ask_df.price == 0, "price"] = ask_float["oraclePrice"] + ask_float["oraclePriceOffset"]
        
        
        bid_df = bid_df.sort_values(['price'], ascending=False)
        ask_df = ask_df.sort_values(['price'])
        
        bid_df = bid_df.reset_index(drop=True)
        ask_df = ask_df.reset_index(drop=True)

        # Store parsed data for that specified market
        self.ob_parsed[perp_market] = (bid_df, ask_df)

        return df_raw, bid_df, ask_df

    def _orderbook_expanded(self, perp_market="SOL-PERP", market_index=MARKET_INDEX_SOL):
        """ Simplified View of Orderbook, by order
        """
        ## Fetch parsed dataframe
        if(perp_market in self.ob_parsed):
            _, bid, ask = self.ob_parsed[perp_market] 
        else:
            _, bid, ask = self.parse_orderbook(perp_market, market_index)
        
        bid_simple = bid[["price", "baseAssetAmount", "baseAssetAmountFilled", "postOnly", "oraclePriceOffset", "oraclePrice"]]
        ask_simple = ask[["price", "baseAssetAmount", "baseAssetAmountFilled", "postOnly", "oraclePriceOffset", "oraclePrice"]]

        ## Display unfilled order amounts
        bid_simple["qty"] = bid_simple["baseAssetAmount"] - bid_simple["baseAssetAmountFilled"]
        ask_simple["qty"] = ask_simple["baseAssetAmount"] - ask_simple["baseAssetAmountFilled"]

        return bid_simple, ask_simple

    def get_dlob(self, perp_market="SOL-PERP", market_index=MARKET_INDEX_SOL):
        """ L2 orderbook view (only price / quantities) split by bid/ask
        """
        bid, ask = self._orderbook_expanded(perp_market, market_index)
        
        bid_ob = bid[["price", "qty"]].groupby("price", sort=False).sum()
        ask_ob = ask[["price", "qty"]].groupby("price", sort=False).sum()

        self.ob_l2_view[perp_market] = (bid_ob, ask_ob)

        return bid_ob, ask_ob
    

    def dlob_metrics(self, perp_market="SOL-PERP", oraclePrice=0):
        """ Returns various metrics on L2 orderbook, either in absolute terms or scaled by oraclePrice 
        """
        # Look at weighted average of the top of the book .. can signal short term volatiltiy
        if(perp_market not in self.ob_l2_view):
            raise Exception("L2 Orderbook View not populated")

        bid_ob, ask_ob = self.ob_l2_view[perp_market]

        tob = bid_ob.iloc[:3].reset_index()
        toa = ask_ob.iloc[:3].reset_index()

        return {"mid"       : (bid_ob.index.max() + ask_ob.index.min() ) / 2.0 - oraclePrice,
                "bid"       :  bid_ob.index.max() - oraclePrice,
                "ask"       :  ask_ob.index.min() - oraclePrice,
                "bid_avg"   : (tob["price"] * tob["qty"]).sum() / tob["qty"].sum() - oraclePrice,
                "ask_avg"   : (toa["price"] * toa["qty"]).sum() / toa["qty"].sum() - oraclePrice,
        }
    

#### Order Wrappers
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

async def construct_and_place(drift_acct, bid_offsets, bid_qtys, ask_offsets, ask_qtys, cancel_existing=True):
    """
        Bundles offsets / quantities for bids / asks into an array of orders
        and places order
        
        cancels existing orders before placing --> can turn this off

        Only places orders with >0 quantity 
        Only 0th subaccount (For simplcity of bookkeeping)
    """
    all_orders = []
    if(cancel_existing):
        all_orders.append(await drift_acct.get_cancel_orders_ix(0))

    for bo, bq in zip(bid_offsets, bid_qtys):
        if(bq > 0):
            all_orders.append(
                await drift_acct.get_place_perp_order_ix(mm_order_single(bo, PositionDirection.LONG(), bq), 0)
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
        file,
    ):
    """
    Main event loop. 

    Set hyperparmaeters, and then:
    
    LOOP:
        (0) Pull Stats
        (1) Get position
        (2) Get orderbook Data
        (3) Compute metrics on orderobok Data
        (4) Signals to shrink/expand spread
        (5) Cancel existing orders + place new orders

    """
    with open(os.path.expanduser(keypath), 'r') as f: secret = json.load(f) 
    kp = Keypair.from_secret_key(bytes(secret))
    print('Using public key:', kp.public_key)

    ## Set up clients to interact with drift
    config      = configs[env]
    wallet      = Wallet(kp)
    connection  = AsyncClient(url)
    provider    = Provider(connection, wallet)
    drift_acct  = ClearingHouse.from_config(config, provider)
    chu         = ClearingHouseUser(drift_acct, use_cache=True)

    # Delegate margin calculation / pnl / etc to clearing house 
    async def get_chu_stats(chu):
        perp_market = await chu.get_perp_market(0)
        return {
            "unrealized_pnl" : await chu.get_unrealized_pnl() / PRICE_PRECISION,
            "oracle_price"   : (await chu.get_perp_oracle_data(perp_market)).price / PRICE_PRECISION,
            "curr_leverage"  : ( await chu.get_leverage() ) / 10_000,  # h/t bigz
            "SOL_PERP_pos"   : (await chu.get_user_position(0))
        }
    
    ############################
    ## Main event loop
    ############################
    history = {}
    epoch = 0 
    while epoch < NUM_EPOCHS: 
        
        print("*" * 10, "Epoch: ", epoch, "*" * 10)
        epoch += 1
        
        ## ---------- Pull stats  --------------
        await chu.set_cache()
        stats = await get_chu_stats(chu)


        print(f"> upnl = {stats['unrealized_pnl']}")
        print(f"> oracle = {stats['oracle_price']}")
        print(f"leverage = {stats['curr_leverage']}")
        
        pp = stats['SOL_PERP_pos'] 
        current_pos = pp.base_asset_amount / 1e9  
        print(f"> current pos = {current_pos}")
        
        ## ---------- Get Orderbook Data  --------------
        curr_ob = OrderBook()

        # Get L2 View + Compute metrics
        bid_ob, ask_ob = curr_ob.get_dlob()
        metrics = curr_ob.dlob_metrics("SOL-PERP", stats['oracle_price'])
        print(f"> dlob metrics = {metrics}")
        
        ## ---------- Construct signals  --------------

        ## check if approaching overleveraged -- here we need to reduce position IMMEDIATELY
        SIGNAL_over_leveraged = True if stats['curr_leverage'] > 0.9 * LEVERAGE_LIMIT else False
        
        ## check if approaching max size  ... start to back off
        SIGNAL_engage_sell_pressure  = True if current_pos > TARGET_MAX_SIZE  * 0.8 else False # Hyperparemeter 
        SIGNAL_engage_buy_pressure   = True if current_pos < -TARGET_MAX_SIZE * 0.8 else False # Hyperparemeter 
        
        ## hide (or quote REALLY bad) bid / asks if we have a large position and need to dump 
        SIGNAL_no_bid = True if SIGNAL_over_leveraged and SIGNAL_engage_sell_pressure else False
        SIGNAL_no_ask = True if SIGNAL_over_leveraged and SIGNAL_engage_buy_pressure else False
        
        ## signal -- skew interval based on buy/sell pressure 
        SIGNAL_order_book_adj = False
        if(metrics['bid_avg'] <= stats['oracle_price'] <= metrics['ask_avg']):
            SIGNAL_order_book_adj = True
            ask_to_bid_factor = (metrics['ask_avg'] - stats['oracle_price']) / (stats['oracle_price'] - metrics['bid_avg'])
            if(ask_to_bid_factor > 1):
                ask_adj_factor = 1 + np.ln(ask_to_bid_factor)
                bid_adj_factor = 1
            elif(ask_to_bid_factor < 1):
                bid_adj_factor = 1 + np.ln(1/ask_to_bid_factor)
                ask_adj_factor = 1
        
        # ---------------- BID SIDE -----------------
        ## Compute total size
        bid_target_pos     = max(0, TARGET_MAX_SIZE - current_pos)
        bid_offset_width   = AGGRESSION 
        
        ## Apply each signal -- shrink or expand the quoting width
        if(SIGNAL_order_book_adj):
            bid_offset_width *= bid_adj_factor
        
        if(SIGNAL_engage_buy_pressure):
            bid_offset_width *= 0.5 #Hyperparemeter

        # Compute bid offsets (w.r.t oracle)  
        if(not SIGNAL_no_bid):
            bid_offsets= np.linspace(0, bid_offset_width, 4)[1:]      # e.g. 0.05, 0.1, 0.15
            bid_qts    = bid_target_pos * np.array([0.25, 0.5, 0.25]) # "parabolic" weighting that's easier to analyze  

            print(f"\n> bid_target_pos = {bid_target_pos}")
            print(f"> bid_offsets = {bid_offsets}")
            print(f"> bid_qts = {bid_qts}")
        else:
            print("\nNo bid, due to overleveraged position")
        
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

            print(f"\n> ask_target_pos = {ask_target_pos}")
            print(f"> ask_prices = {ask_offsets}")
            print(f"> ask_qts = {ask_qts}")
        else:
            print("\nNo ask, due to overleveraged position")
        
        ## ---------- Cancel existing orders + Send new orders  --------------
        ret = await construct_and_place(drift_acct, bid_offsets, bid_qts, ask_offsets, ask_qts)
        print("Order confirmation: ", ret)

        history[epoch] = {
            "time"        : dt.datetime.now(),
            "oraclePrice" : stats['oracle_price'],
            "currLeverage": stats["curr_leverage"],
            ## Below are orderbook values
            "ob_mid"         : metrics['mid'],
            "ob_bid"         : metrics['bid'],
            "ob_ask"         : metrics['ask'],
            ## Below are market maker (us) values
            "mm_bid_offsets"    : bid_offsets,
            "mm_bid_qts"        : bid_qts,
            "mm_ask_offsets"    : ask_offsets,
            "mm_ask_qts"        : ask_qts,
            "txn"               : ret,
        }

        ## Save data for analysis / tuning parmaeters
        history_df = pd.DataFrame.from_dict(history)
        history_df.to_pickle(file)
        
        ## ---------- Logging --------------    
        # Step 6: Wait before repeating the loop
        await asyncio.sleep(SLEEP_INTERVAL)

    return history_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False, default=os.environ.get('ANCHOR_WALLET'))
    parser.add_argument('--env', type=str, default='devnet')
    parser.add_argument('--file', type=str, default="history.h5")

    args = parser.parse_args()

    if args.keypath is None:
        if os.environ['ANCHOR_WALLET'] is None:
            raise NotImplementedError("need to provide keypath or set ANCHOR_WALLET")
        else:
            args.keypath = os.environ['ANCHOR_WALLET']

    if args.env == 'devnet':
        url = 'https://api.devnet.solana.com'
    else:
        raise NotImplementedError('only devnet env supported at this time')

    asyncio.run(main(
        args.keypath, 
        args.env, 
        url,
        args.file,
    ))