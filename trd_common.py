import os
import re
import json
import asyncio
import logging
import base64
import random
from typing import Dict, List
import httpx
from logging.handlers import TimedRotatingFileHandler

from dotenv import load_dotenv
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed

from jup_python_sdk.clients.ultra_api_client import UltraApiClient
from jup_python_sdk.models.ultra_api.ultra_order_request_model import UltraOrderRequest
from datetime import datetime

# Load environment variables
load_dotenv()

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging to rotate daily
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler("logs/trading.log", when="midnight", interval=1, backupCount=7)
file_handler.setFormatter(formatter)
file_handler.suffix = "%Y-%m-%d"

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class TradingBot:
    def __init__(self):
        solana_url = os.getenv("SOLANA_RPC_URL")
        self.client = AsyncClient(solana_url, commitment=Confirmed)

        key_path = os.getenv("PRIVATE_KEY_FILE", "private_key.json")
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Hiányzik a privát kulcs fájl: {key_path}")

        try:
            with open(key_path, "r") as f:
                key_list = json.load(f)

            if not isinstance(key_list, list) or len(key_list) != 64:
                raise ValueError("A privát kulcs JSON tömbként, pontosan 64 elemmel legyen megadva.")
            key_bytes = bytes(key_list)
            self.keypair = Keypair.from_bytes(key_bytes)
            encoded = base64.b64encode(key_bytes).decode()
            os.environ["PRIVATE_KEY"] = str(self.keypair)
        except Exception as e:
            raise ValueError(f"Privát kulcs beolvasási hiba: {e}")

        self.jupiter = UltraApiClient()

        self.active_trades_file = "strategies/trades.json"
        self.active_trades: List[dict] = self.load_trades()

        self.USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        self.strategy = self.load_strategy("strategies/default.json")

    def load_strategy(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Nem sikerült betölteni a stratégiát: {e}")
            return {
                "buy_amount_usdc": 1,
                "slippage": 0.5,
                "check_interval": 30,
                "sell_strategy": []
            }

    def save_trades(self):
        try:
            with open(self.active_trades_file, "w") as f:
                json.dump(self.active_trades, f)
        except Exception as e:
            logging.error(f"Nem sikerült menteni a pozíciókat: {e}")

    def load_trades(self) -> List[dict]:
        if os.path.exists(self.active_trades_file):
            try:
                with open(self.active_trades_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return list(data.values())
                    elif isinstance(data, list):
                        return data
            except Exception as e:
                logging.error(f"Nem sikerült betölteni a pozíciókat: {e}")
        return []

    async def is_valid_token(self, token_address: str) -> bool:
        try:
            resp = await self.client.get_token_supply(Pubkey.from_string(token_address))
            return resp.value is not None
        except Exception as e:
            logging.warning(f"Nem érvényes token cím: {token_address} – {e}")
            return False

    async def fetch_token_price(self, token_address: str) -> float:
        try:
            url = f"https://lite-api.jup.ag/price/v2?ids={token_address}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                data = response.json()
                price_str = data["data"].get(token_address, {}).get("price")
                if price_str:
                    return float(price_str)
        except Exception as e:
            logging.error(f"Árfolyam lekérdezési hiba: {e}")
        return 0.0

    async def process_token(self, token_address: str):
        logging.info(f"📥 Üzenet feldolgozása: {token_address}")

        if not await self.is_valid_token(token_address):
            logging.warning(f"🚫 Nem érvényes token cím: {token_address}")
            await self.send_telegram_message(f"🚫 Nem érvényes token cím: {token_address}")
            return

        if any(t["token"] == token_address for t in self.active_trades):
            logging.warning(f"⚠️ Már létezik pozíció ezzel a tokennel: {token_address}")
            await self.send_telegram_message(f"⚠️ Már nyitott pozíció van: {token_address}")
            return

        await asyncio.sleep(random.uniform(2, 5))  # Random delay before buy

        max_retries = 10
        for attempt in range(max_retries):
            try:
                amount = int(float(self.strategy.get("buy_amount_usdc", 1)) * 1_000_000)
                slippage = float(self.strategy.get("slippage", 0.5))

                order = UltraOrderRequest(
                    input_mint=self.USDC_MINT,
                    output_mint=token_address,
                    amount=amount,
                    taker=str(self.keypair.pubkey()),
                    slippage_bps=int(slippage * 100)
                )

                response = self.jupiter.order_and_execute(order)
                logging.info(f"✅ Vásárlás sikeres: {response}")

                bought_at = await self.fetch_token_price(token_address)
                output_amount_str = response.get("outputAmountResult")
                output_amount = int(output_amount_str) if output_amount_str else amount

                self.active_trades.append({
                    "token": token_address,
                    "bought_at": bought_at,
                    "strategy": self.strategy["sell_strategy"],
                    "amount": output_amount,
                    "steps_executed": []
                })
                self.save_trades()
                break

            except Exception as e:
                logging.error(f"❌ Vásárlási hiba ({attempt + 1}/{max_retries}): {e}")

    async def execute_sell(self, token_address: str, amount: int):
        logging.debug(f"Eladási kérés paraméterei: input={token_address}, output={self.USDC_MINT}, amount={amount}")
        try:
            slippage = float(self.strategy.get("slippage", 0.5))
            order = UltraOrderRequest(
                input_mint=token_address,
                output_mint=self.USDC_MINT,
                amount=amount,
                taker=str(self.keypair.pubkey()),
                slippage_bps=int(slippage * 100)
            )
            response = self.jupiter.order_and_execute(order)
            logging.info(f"✅ Eladás sikeres: {response}")
        except Exception as e:
            logging.error(f"[!] Eladási hiba: {e}")

    async def check_sell_conditions(self):
        remaining_trades = []
        for trade in self.active_trades:
            token = trade["token"]
            strategy_steps = trade["strategy"]
            current_price = await self.fetch_token_price(token)
            bought_at = trade["bought_at"]
            total_amount = trade["amount"]
            steps_executed = trade.get("steps_executed", [])

            if bought_at == 0:
                remaining_trades.append(trade)
                continue

            profit_ratio = current_price / bought_at

            logging.debug(f"[CHECK] Token: {token}, Current: {current_price}, Bought at: {bought_at}, Profit: {profit_ratio:.4f}")

            for i, step in enumerate(strategy_steps):
                if i in steps_executed:
                    continue

                target = step.get("target_multiplier")
                stop_loss = step.get("stop_loss", None)
                percentage = step.get("percentage", 100)

                step_amount = int(total_amount * (percentage / 100))

                logging.debug(f"[STRATEGY] Step {i}: Target={target}, Stop={stop_loss}, Amount={step_amount}, Already executed={i in steps_executed}")

                if profit_ratio >= target:
                    logging.info(f"🎯 Cél elérés, eladás: {token}, lépés: {i}")
                    await self.execute_sell(token, step_amount)
                    steps_executed.append(i)

                elif stop_loss and profit_ratio <= stop_loss:
                    logging.info(f"🔻 Stop-loss trigger, eladás: {token}, lépés: {i}")
                    await self.execute_sell(token, step_amount)
                    steps_executed.append(i)

            if len(steps_executed) != len(strategy_steps):
                trade["steps_executed"] = steps_executed
                remaining_trades.append(trade)

        self.active_trades = remaining_trades
        self.save_trades()

    async def send_telegram_message(self, message: str):
        return  # Üzenetküldés letiltva teljesen

    def extract_token_address(self, text: str) -> str | None:
        match = re.search(r'[1-9A-HJ-NP-Za-km-z]{32,44}', text)
        return match.group(0) if match else None

    async def handle_message(self, update: dict):
        message = update.get("message", {}).get("text")
        if message:
            token_address = self.extract_token_address(message)
            if token_address:
                await self.process_token(token_address)

    async def poll_telegram(self):
        offset = None

        async def poll_updates():
            nonlocal offset
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            params = {"timeout": 100, "offset": offset}
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params)
                    data = response.json()
                    updates = data.get("result", [])
                    for update in updates:
                        offset = update["update_id"] + 1
                        await self.handle_message(update)
            except Exception as e:
                logging.error(f"Hiba a Telegram lekérdezésben: {e}")

        while True:
            await asyncio.gather(
                poll_updates(),
                self.check_sell_conditions()
            )
            await asyncio.sleep(10)  # 10 másodpercenként újra

def main():
    bot = TradingBot()
    asyncio.run(bot.poll_telegram())

if __name__ == "__main__":
    main()
