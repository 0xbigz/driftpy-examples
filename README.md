# driftpy examples

examples to interact with drift protocol-v2 using python sdk

dependencies: [driftpy](https://drift-labs.github.io/driftpy/)

## Quick Setup

creates a virtualenv called "venv"

```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Run

can either use `--keypath` flag or run `export ANCHOR_WALLET=path/to/id.json` for key


post two maker orders for SOL-PERP on devnet (default) with main account (default)

```
python floating_maker.py --amount .1 --market SOL-PERP
```

post two maker orders for SOL/USDC on mainnet with subaccount_id = 2

```
python floating_maker.py --amount .69 --market SOL --env mainnet --subaccount 2
```

stake 1 USDC (market_index=0) to drift v2 insurance fund (on devnet, also default)
```
python if_stake.py --operation add --amount 1 --market 0 --env devnet
```

place SOL-USDC limit orders grid (bids and asks) around the oracle price ($100 worth of orders)
```
python limit_order_grid.py --amount 100 --market SOL --grids 10 --env mainnet
```

place SOL-USDC limit orders grid (bids and asks) around the oracle price ($100 worth of orders)
```
python limit_order_grid.py --amount 100 --market SOL --grids 10 --env mainnet
```

place SOL-USDC limit orders grid (bids and asks) around the oracle price ($100 worth of orders) and enforce a min/max base position
```
python limit_order_grid.py --amount 100 --market SOL --grids 10 --env mainnet --max-position 10 --min-position 0
```

place SOL-USDC limit orders grid (asks only) above oracle but up to $21.50 (enforce a min/max base position)
```
python limit_order_grid.py --amount 100 --market SOL --grids 10 --env mainnet --max-position 10 --min-position 0 --upper-price 21.5
```

place SOL-USDC limit orders grid (bids and asks) between $21.40 and $21.50 (bids below oracle, asks above oracle, and still enforce a min/max base position)
```
python limit_order_grid.py --amount 100 --market SOL --grids 10 --env mainnet --max-position 10 --min-position 0 --lower-price 21.4 --upper-price 21.5
```



## Disclaimer

This is experimental software is for educational purposes only on a developer testnet. USE THIS OPEN SOURCE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR RESULTS.
