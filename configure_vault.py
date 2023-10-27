from solana.publickey import PublicKey
from anchorpy import Program, Idl
import requests
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
import json
import os
from driftpy.constants.config import configs
from driftpy.clearing_house import ClearingHouse
from driftpy.accounts import *
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import get_associated_token_address


async def main(keypath,
               env,
               url,
               name,
               action,
               delegate,
               management_fee,
               profit_share,
               redeem_period,
               max_tokens,
               min_deposit_amount,
               permissioned,
               ):
    with open(os.path.expanduser(keypath), 'r') as f:
        secret = json.load(f)
    kp = Keypair.from_secret_key(bytes(secret))
    print('using public key:', kp.public_key)
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)

    url = 'https://raw.githubusercontent.com/drift-labs/drift-vaults/master/ts/sdk/src/idl/drift_vaults.json'
    response = requests.get(url)
    data = response.json()
    idl = data
    pid = 'vAuLTsyrvSfZRuRB3XgvkPwNGgYSs9YRYymVebLKoxR'
    vault_program = Program(
        Idl.from_json(idl),
        PublicKey(pid),
        provider,
    )
    config = configs[env]
    drift_client = ClearingHouse.from_config(config, provider)

    print(f"vault name: {name}")

    # Initialize an empty list to store the character number array
    char_number_array = [0] * 32

    # Iterate through each character in the string and get its Unicode code point
    for i in range(32):
        if i < len(name):
            char_number_array[i] = ord(name[i])

    vault_pubkey = PublicKey.find_program_address(
        [b"vault", bytes(char_number_array)], PublicKey(pid)
    )[0]

    print(f"vault pubkey : {vault_pubkey}")

    vault_user = get_user_account_public_key(drift_client.program_id, vault_pubkey)

    print(f"vault user : {vault_user}")

    vault_user_stats = get_user_stats_account_public_key(drift_client.program_id, vault_pubkey)

    spot_market_index = 0

    spot_market = await get_spot_market_account(
        drift_client.program, spot_market_index
    )

    if action == 'init':
        params = {
            'name': char_number_array,
            'spot_market_index': spot_market_index,  # USDC spot market index
            'redeem_period': redeem_period,  # 30 days
            'max_tokens': max_tokens,
            'min_deposit_amount': min_deposit_amount,
            'management_fee': management_fee,
            'profit_share': profit_share,
            'hurdle_rate': 0,  # no supported atm
            'permissioned': permissioned,
        }

        # vault_ata = get_associated_token_address(drift_client.authority, spot_market.mint)
        ata = PublicKey.find_program_address(
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
                    'manager': drift_client.signer.public_key,
                    'payer': drift_client.signer.public_key,
                    "rent": SYSVAR_RENT_PUBKEY,
                    "system_program": SYS_PROGRAM_ID,
                }),
        )

        tx = Transaction()
        tx.add(instruction)
        txSig = await vault_program.provider.send(tx)
        print(f"tx sig {txSig}")
    elif action == 'update-delegate':
        instruction = vault_program.instruction['update_delegate'](
            PublicKey(delegate),
            ctx=Context(
                accounts={
                    'drift_user': vault_user,
                    'vault': vault_pubkey,
                    'drift_program': drift_client.program_id,
                    'manager': drift_client.signer.public_key,
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
    parser.add_argument('--action', choices=['init', 'update-delegate'], required=True)
    parser.add_argument('--management-fee', type=int, required=False, default=int(.02 * 1e6))
    parser.add_argument('--profit-share', type=int, required=False, default=int(.02 * 1e6))
    parser.add_argument('--redeem-period', type=int, required=False, default=int(60 * 60 * 24 * 30))
    parser.add_argument('--max-tokens', type=int, required=False, default=int(1_000_000 * 1e6))
    parser.add_argument('--min-deposit-amount', type=int, required=False, default=int(100 * 1e6))
    parser.add_argument('--permissioned', type=int, required=False, default=False)
    parser.add_argument('--delegate', type=str, default=None)
    args = parser.parse_args()

    if args.keypath is None:
        if os.environ['ANCHOR_WALLET'] is None:
            raise ValueError("need to provide keypath or set ANCHOR_WALLET")
        else:
            args.keypath = os.environ['ANCHOR_WALLET']

    if args.action == 'update-delegate' and args.delegate is None:
        raise ValueError('update-delegate requires that you pass a delegate')

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
        args.action,
        args.delegate,
        args.management_fee,
        args.profit_share,
        args.redeem_period,
        args.max_tokens,
        args.min_deposit_amount,
        args.permissioned,
    ))