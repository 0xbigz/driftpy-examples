from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.sysvar import RENT as SYSVAR_RENT_PUBKEY

from solders.system_program import ID as SYS_PROGRAM_ID
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
               action,
               delegate,
               depositor,
               management_fee,
               profit_share,
               redeem_period,
               max_tokens,
               min_deposit_amount,
               permissioned,
               deposit_amount,
               ):
    with open(os.path.expanduser(keypath), 'r') as f:
        secret = json.load(f)
    kp = Keypair.from_bytes(bytes(secret))
    print('using public key:', kp.pubkey())
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)

    url = 'https://raw.githubusercontent.com/drift-labs/drift-vaults/master/ts/sdk/src/idl/drift_vaults.json'
    response = requests.get(url)
    data = response.json()
    idl = data
    pid = 'vAuLTsyrvSfZRuRB3XgvkPwNGgYSs9YRYymVebLKoxR'
    data = response.text
    idl_raw = data
    # idl = json.loads(idl_raw)
    pid = 'vAuLTsyrvSfZRuRB3XgvkPwNGgYSs9YRYymVebLKoxR'
    vault_program = Program(
        Idl.from_json(idl_raw),
        Pubkey.from_string(pid),
        provider,
    )
    drift_client: DriftClient = DriftClient(provider.connection, provider.wallet, env.split('-')[0], account_subscription=AccountSubscriptionConfig("cached"))
    # Initialize an empty list to store the character number array
    char_number_array = [32] * 32 # 32 is unicode for space

    # Iterate through each character in the string and get its Unicode code point
    for i in range(32):
        if i < len(name):
            char_number_array[i] = ord(name[i])

    vault_pubkey = Pubkey.find_program_address(
        [b"vault", bytes(char_number_array)], Pubkey.from_string(pid)
    )[0]

    print(f"vault pubkey : {vault_pubkey}")

    vault_user = get_user_account_public_key(drift_client.program_id, vault_pubkey)

    print(f"vault user : {vault_user}")

    vault_user_stats = get_user_stats_account_public_key(
        drift_client.program_id, vault_pubkey)

    spot_market_index = 0

    spot_market = await get_spot_market_account(
        drift_client.program, spot_market_index
    )

    print(f"action {action}")

    if action == 'init-vault':
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

        txSig = await drift_client.send_ixs([instruction])
        print(f"tx sig {txSig}")
    if action == 'update-vault':
        params = {
            'redeem_period': redeem_period,  # 30 days
            'max_tokens': max_tokens,
            'min_deposit_amount': min_deposit_amount,
            'management_fee': management_fee,
            'profit_share': profit_share,
            'hurdle_rate': None,  # no supported atm
            'permissioned': permissioned,
        }
        instruction = vault_program.instruction['update_vault'](
            params,
            ctx=Context(
                accounts={
                    'vault': vault_pubkey,
                    'manager': drift_client.authority,
                }),
        )

        txSig = await drift_client.send_ixs([instruction])

        print(f"tx sig {txSig}")
    elif action == 'update-delegate':
        instruction = vault_program.instruction['update_delegate'](
            Pubkey.from_string(delegate),
            ctx=Context(
                accounts={
                    'drift_user': vault_user,
                    'vault': vault_pubkey,
                    'drift_program': drift_client.program_id,
                    'manager': drift_client.authority,
                }),
        )

        txSig = await drift_client.send_ixs([instruction])
        print(f"tx sig {txSig}")
    elif action == 'init-depositor':
        depositor_pubkey = Pubkey.from_string(depositor)
        vault_depositor_pubkey = Pubkey.find_program_address(
            [b"vault_depositor", bytes(vault_pubkey), bytes(
                depositor_pubkey)], Pubkey.from_string(pid)
        )[0]

        print(f"vault depositor pubkey : {vault_depositor_pubkey}")

        instruction = vault_program.instruction['initialize_vault_depositor'](
            ctx=Context(
                accounts={
                    'vault': vault_pubkey,
                    'vault_depositor': vault_depositor_pubkey,
                    'authority': depositor_pubkey,
                    'payer': drift_client.authority,
                    "rent": SYSVAR_RENT_PUBKEY,
                    "system_program": SYS_PROGRAM_ID,
                }),
        )

        txSig = await drift_client.send_ixs([instruction])
        print(f"tx sig {txSig}")
    elif action == 'deposit':
        depositor_pubkey = drift_client.authority
        vault_depositor_pubkey = Pubkey.find_program_address(
            [b"vault_depositor", bytes(vault_pubkey), bytes(
                depositor_pubkey)], Pubkey.from_string(pid)
        )[0]

        vault_ata = Pubkey.find_program_address(
            [b"vault_token_account", bytes(vault_pubkey)], vault_program.program_id
        )[0]

        depositor_ata = get_associated_token_address(drift_client.authority, spot_market.mint)

        print(f"vault depositor pubkey : {vault_depositor_pubkey}")

        remaining_accounts = await drift_client.get_remaining_accounts(writable_spot_market_index=spot_market_index, user_id=[0], authority=vault_pubkey)

        instruction = vault_program.instruction['deposit'](
            deposit_amount,
            ctx=Context(
                accounts={
                    'vault': vault_pubkey,
                    'vault_depositor': vault_depositor_pubkey,
                    'authority': drift_client.authority,
                    'drift_spot_market': spot_market.pubkey,
                    'drift_spot_market_vault': spot_market.vault,
                    'drift_user_stats': vault_user_stats,
                    'drift_user': vault_user,
                    'drift_state': drift_client.get_state_public_key(),
                    'token_program': TOKEN_PROGRAM_ID,
                    'drift_program': drift_client.program_id,
                    'vault_token_account': vault_ata,
                    'user_token_account': depositor_ata,
                },
                remaining_accounts=remaining_accounts
            ),
        )

        txSig = await drift_client.send_ixs([instruction])
        print(f"tx sig {txSig}")

    vault_account = await vault_program.account.get('Vault').fetch(vault_pubkey, "processed")
    print("vault account", vault_account)


def get_fee_param(fee, param_name):
    if fee > 1 or fee < 0:
        raise ValueError(f"{param_name} must be between 0 and 1")
    return int(fee * 1e6)


def get_token_amount_param(amount, param_name):
    if amount < 0:
        raise ValueError(f"{param_name} must be greater than 0")
    return int(amount * 1e6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keypath', type=str, required=False,
                        default=os.environ.get('ANCHOR_WALLET'), help='path to keypair for manager/depositor')
    parser.add_argument('--name', type=str, required=True, default='devnet',
                        help='name of the vault. uniquely identifies the vault')
    parser.add_argument('--cluster', type=str,         choices=[
        'devnet',
        'mainnet-beta'],
        default='devnet', help='the cluster to connect to')
    parser.add_argument(
        '--action',
        choices=[
            'init-vault',
            'init-depositor',
            'update-delegate',
            'update-vault',
            'deposit',
        ],
        required=True,
        help='the action to perform. init-vault will create a new vault, init-depositor will create a depositor for permissioned vault, update-vault will update an existing vault, update-delegate will update the delegate of a vault'
    )
    parser.add_argument('--management-fee', type=float, required=False, default=None,
                        help='the management fee applied to vault deposits (between 0 and 1). 0.2 = 20%%')
    parser.add_argument('--profit-share', type=float, required=False, default=None,
                        help='the profit share applied to vault pnl (between 0 and 1). 0.2 = 20%%')
    parser.add_argument('--redeem-period', type=int, required=False, default=None,
                        help='the redeem period in seconds. eg 1 day redeem period = 86400')
    parser.add_argument('--max-tokens', type=int, required=False, default=None,
                        help='the max number of tokens that can be deposited into the vault')
    parser.add_argument('--min-deposit-amount', type=int, required=False, default=None,
                        help='the minimum amount of tokens that must be deposited into the vault at one time')
    parser.add_argument('--permissioned', type=bool, required=False, default=None,
                        help='whether the vault is permissioned or not. if permissioned, vault manager must initialize depositor account')
    parser.add_argument('--delegate', type=str, default=None,
                        help='the delegate to update the vault to (only used for update-delegate action)')
    parser.add_argument('--depositor', type=str, default=None,
                        help='the depositor to initialize for permisisoned vault (only used for init-depositor action)')
    parser.add_argument('--deposit-amount', type=int, default=None,
                        help='the amount of tokens to deposit into the vault (only used for deposit action)')
    args = parser.parse_args()

    if args.keypath is None:
        if os.environ['ANCHOR_WALLET'] is None:
            raise ValueError("need to provide keypath or set ANCHOR_WALLET")
        else:
            args.keypath = os.environ['ANCHOR_WALLET']

    action = args.action

    management_fee = args.management_fee
    profit_share = args.profit_share
    redeem_period = args.redeem_period
    max_tokens = args.max_tokens
    min_deposit_amount = args.min_deposit_amount
    permissioned = args.permissioned

    if action == 'init-vault':
        if management_fee is None:
            management_fee = .2

        if profit_share is None:
            profit_share = .02

        if redeem_period is None:
            redeem_period = int(60 * 60 * 24 * 30)

        if max_tokens is None:
            max_tokens = int(0)

        if min_deposit_amount is None:
            min_deposit_amount = int(0)

        if permissioned is None:
            permissioned = False

    # handle some santization/formatting
    if action == 'init-vault' or action == 'update-vault':
        if management_fee is not None:
            management_fee = get_fee_param(management_fee, 'management fee')

        if profit_share is not None:
            profit_share = get_fee_param(profit_share, 'profit share')

        if max_tokens is not None:
            max_tokens = get_token_amount_param(max_tokens, 'max tokens')

        if min_deposit_amount is not None:
            min_deposit_amount = get_token_amount_param(
                min_deposit_amount, 'min deposit amount')

    if args.action == 'update-delegate':
        if args.delegate is None:
            raise ValueError('update-delegate requires that you pass a delegate')

    if args.action == 'init-depositor':
        if args.depositor is None:
            raise ValueError('init-depositor requires that you pass a depositor')

    deposit_amount = args.deposit_amount
    if args.action == 'deposit':
        if deposit_amount is None:
            raise ValueError('deposit requires that you pass a deposit amount')
        deposit_amount = get_token_amount_param(deposit_amount, 'deposit amount')

    if args.cluster == 'devnet':
        url = 'https://api.devnet.solana.com'
    elif args.cluster == 'mainnet':
        url = 'https://api.mainnet-beta.solana.com'
    else:
        raise NotImplementedError('only devnet/mainnet env supported')

    import asyncio

    asyncio.run(main(
        args.keypath,
        args.cluster,
        url,
        args.name,
        args.action,
        args.delegate,
        args.depositor,
        management_fee,
        profit_share,
        redeem_period,
        max_tokens,
        min_deposit_amount,
        permissioned,
        deposit_amount
    ))
