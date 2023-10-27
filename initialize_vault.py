
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
    subaccount_id,
    authority=None,):
    with open(os.path.expanduser(keypath), 'r') as f: secret = json.load(f) 
    kp = Keypair.from_secret_key(bytes(secret))
    print('using public key:', kp.public_key, 'subaccount=', subaccount_id)
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
    drift_client.authority = PublicKey(authority)
    print(vault_program.rpc.keys())
    VAULT_NAME = 'TEST_VAULT'

    # Initialize an empty list to store the character number array
    char_number_array = [0] * 32

    # Iterate through each character in the string and get its Unicode code point
    for i in range(32):
        if i < len(VAULT_NAME):
            char_number_array[i] = ord(VAULT_NAME[i])

    # Print the original string and the character number array
    print(VAULT_NAME)
    print(char_number_array)
    print(VAULT_NAME, char_number_array)
    vault_pubkey = PublicKey.find_program_address(
        [b"vault", bytes(char_number_array)], PublicKey(pid)
    )[0]
    params = {
		'name': char_number_array,
		'spot_market_index': 0, # USDC spot market index
		'redeem_period': int(60 * 60 * 24 * 30), # 30 days
		'max_tokens': int(1_000_000 * 1e6),
		'min_deposit_amount': int(100 * 1e6),
		'management_fee': int(.02 * 1e6),
		'profit_share': int(.2 * 1e6),
		'hurdle_rate': 0, # no supported atm
		'permissioned': False,
	}
    # ch_signer = get_clearing_house_signer_public_key(drift_client.program_id)
    spot_market = await get_spot_market_account(
                    drift_client.program, params['spot_market_index']
                )
    
    print(idl['types'])
    print(drift_client.authority)
    print(vault_program.instruction['initialize_vault'].idl_ix)
    #vault_ata = get_associated_token_address(drift_client.authority, spot_market.mint)
    ata = PublicKey.find_program_address(
        [b"vault_token_account", bytes(vault_pubkey)], vault_program.program_id
    )[0]

    print('ATA:', str(ata))
    drift_client.authority = vault_pubkey
    instruction = vault_program.instruction['initialize_vault'](
        params,
            ctx=Context(
                accounts={
			'drift_spot_market': spot_market.pubkey,
			'drift_spot_market_mint': spot_market.mint,
			'drift_user_stats': drift_client.get_user_stats_public_key(),
			'drift_user': drift_client.get_user_account_public_key(subaccount_id),
			'drift_state': drift_client.get_state_public_key(),
			'vault': vault_pubkey,
			'token_account': ata,
			'token_program': TOKEN_PROGRAM_ID,
			'drift_program': drift_client.program_id,
            'manager': drift_client.authority,
            'payer': drift_client.authority,
            "rent": SYSVAR_RENT_PUBKEY,
            "system_program": SYS_PROGRAM_ID,
	}            ),
        )
    
    print(instruction)
    tx = Transaction()
    tx.add(instruction)
    print(drift_client.signer, drift_client.signers)
    await vault_program.provider.send(tx, signers=[kp])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False, default=os.environ.get('ANCHOR_WALLET'))
    parser.add_argument('--env', type=str, default='devnet')
    parser.add_argument('--subaccount', type=int, required=False, default=0)
    parser.add_argument('--authority', type=str, required=False, default=None)
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
        args.subaccount,
        args.authority
    ))


