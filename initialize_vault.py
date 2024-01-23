from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.sysvar import RENT as SYSVAR_RENT_PUBKEY

from solders.system_program import ID as SYS_PROGRAM_ID
from solana.transaction import Transaction

from anchorpy import Program, Idl, Context
import requests
from anchorpy import Provider, Wallet
from solana.rpc.async_api import AsyncClient
import json
import os
from driftpy.constants.config import configs
from driftpy.drift_client import DriftClient, AccountSubscriptionConfig
from driftpy.accounts import *
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import get_associated_token_address


async def main(keypath,
               env,
               url,
               name,
               ):
    with open(os.path.expanduser(keypath), 'r') as f:
        secret = json.load(f)
    kp = Keypair.from_bytes(bytes(secret))
    print('using public key:', str(kp.pubkey()))
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)

    url = 'https://raw.githubusercontent.com/drift-labs/drift-vaults/master/ts/sdk/src/idl/drift_vaults.json'
    response = requests.get(url)
    data = response.text
    idl_raw = data
    # idl = json.loads(idl_raw)
    pid = 'vAuLTsyrvSfZRuRB3XgvkPwNGgYSs9YRYymVebLKoxR'
    vault_program = Program(
        Idl.from_json(idl_raw),
        Pubkey.from_string(pid),
        provider,
    )
    config = configs[env]
    # drift_client = DriftClient.from_config(config, provider)
    drift_client: DriftClient = DriftClient(provider.connection, provider.wallet, env.split('-')[0], account_subscription=AccountSubscriptionConfig("cached"))
    # Initialize an empty list to store the character number array
    char_number_array = [0] * 32

    # Iterate through each character in the string and get its Unicode code point
    for i in range(32):
        if i < len(name):
            char_number_array[i] = ord(name[i])

    # Print the original string and the character number array
    vault_pubkey = Pubkey.find_program_address(
        [b"vault", bytes(char_number_array)], Pubkey.from_string(pid)
    )[0]
    params = {
        'name': char_number_array,
        'spot_market_index': 0,  # USDC spot market index
        'redeem_period': int(60 * 60 * 24 * 30),  # 30 days
        'max_tokens': int(1_000_000 * 1e6),
        'min_deposit_amount': int(100 * 1e6),
        'management_fee': int(.02 * 1e6),
        'profit_share': int(.2 * 1e6),
        'hurdle_rate': 0,  # no supported atm
        'permissioned': False,
    }
    # ch_signer = get_clearing_house_signer_public_key(drift_client.program_id)
    spot_market = await get_spot_market_account(
        drift_client.program, params['spot_market_index']
    )

    vault_user = get_user_account_public_key(drift_client.program_id, vault_pubkey)
    vault_user_stats = get_user_stats_account_public_key(drift_client.program_id, vault_pubkey)

    # vault_ata = get_associated_token_address(drift_client.authority, spot_market.mint)
    ata = Pubkey.find_program_address(
        [b"vault_token_account", bytes(vault_pubkey)], vault_program.program_id
    )[0]

    instruction = vault_program.instruction['initialize_vault'](
        params,
        ctx=Context(
            accounts={
                'drift_spot_market': spot_market.pubkey,
                'drift_spot_market_mint': spot_market.mint,
                'drift_user_stats': vault_user_stats,
                'drift_user': vault_user,
                'drift_state': drift_client.get_state_public_key(),
                'vault': vault_pubkey,
                'token_account': ata,
                'token_program': TOKEN_PROGRAM_ID,
                'drift_program': drift_client.program_id,
                'manager': drift_client.authority,
                'payer': drift_client.authority,
                "rent": SYSVAR_RENT_PUBKEY,
                "system_program": SYS_PROGRAM_ID,
            }),
    )

    tx = Transaction()
    tx.add(instruction)
    txSig = await vault_program.provider.send(tx)
    print(f"tx sig {txSig}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False, default=os.environ.get('ANCHOR_WALLET'))
    parser.add_argument('--name', type=str, required=True, default='devnet')
    parser.add_argument('--env', type=str, default='devnet')
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
        args.name,
    ))