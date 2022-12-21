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

post two maker orders for SOL-PERP on devnet (default) with main account (default)

```
python floating_maker.py --amount .1 --market SOL-PERP
```

post two maker orders for SOL/USDC on mainnet with subaccount_id = 2

```
python floating_maker.py --amount .69 --market SOL --env mainnet --subaccount 2
```

## Disclaimer

This is experimental software is for educational purposes only on a developer testnet. USE THIS OPEN SOURCE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR RESULTS.
