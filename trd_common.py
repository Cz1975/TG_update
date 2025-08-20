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
                # Telegram több chat ID támogatás
        chat_ids_str = os.getenv("TELEGRAM_NOTIFY_IDS", "")
        self.notify_chat_ids = [cid.strip() for cid in chat_ids_str.split(",") if cid.strip()]

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
                json.dump(self.active_trades, f, indent=4)
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

#        await asyncio.sleep(random.uniform(2, 5))

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
            
                output_amount_str = response.get("outputAmountResult")
                output_amount = int(output_amount_str) if output_amount_str else 0

                # Slippage határ ellenőrzés
                if output_amount == 0 or output_amount < amount * (1 - slippage / 100):
                    msg = f"⚠️ Slippage miatt elutasított vásárlás ({attempt+1}/{max_retries}): {token_address}"
                    logging.warning(msg)
                    await self.send_telegram_message(msg)
                    continue  # újrapróbálkozás
                # Vásárlás sikeres
                #bought_at = await self.fetch_token_price(token_address)
                bought_at = (amount / 1_000_000) / (output_amount / 1_000_000)  # Valós árfolyam: USDC/token
                self.active_trades.append({
                    "token": token_address,
                    "bought_at": bought_at,
                    "strategy": self.strategy["sell_strategy"],
                    "amount": output_amount,
                    "steps_executed": []
                })
                self.save_trades()
                
                msg = (
                    f"✅ Vásárlás sikeres\n"
                    f"Token: {token_address}\n"
                    f"Összeg: {output_amount / 1_000_000:.6f} token\n"
                    f"Árfolyam: {bought_at:.6f} USDC/token"
                )
                logging.info(msg)
                await self.send_telegram_message(msg)
                
                break

            except Exception as e:
                msg = f"❌ Vásárlási hiba ({attempt+1}/{max_retries}): {e}"
                logging.error(msg)
                await self.send_telegram_message(msg)

    def extract_token_addresses(self, text: str) -> List[str]:
        return re.findall(r"[1-9A-HJ-NP-Za-km-z]{32,44}", text)

    async def handle_message(self, update: dict):
        message = update.get("message", {}).get("text")
        if message:
            token_addresses = self.extract_token_addresses(message)
            for token_address in token_addresses:
                await self.process_token(token_address)

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

            # Lekérjük az aktuális árfolyamot
            price = await self.fetch_token_price(token_address)
            msg = (
                  f"✅ Eladás sikeres\n"
                  f"Token: {token_address}\n"
                  f"Eladott összeg: {amount / 1_000_000:.6f} token\n"
                  f"Árfolyam: {price:.6f} USDC/token"
            )
            await self.send_telegram_message(msg)
        
        except Exception as e:
            logging.error(f"[!] Eladási hiba: {e}")
            await self.send_telegram_message(f"❌ Eladási hiba: {token_address} - {e}")

    async def check_sell_conditions(self):
        remaining_trades = []
        for trade in self.active_trades:
            token = trade["token"]
            strategy_steps = trade["strategy"]
            #current_price = await self.fetch_token_price(token)
            bought_at = trade["bought_at"]
            total_amount = trade["amount"]
            steps_executed = trade.get("steps_executed", [])

            if bought_at == 0:
                remaining_trades.append(trade)
                continue

            #profit_ratio = current_price / bought_at

            #logging.debug(f"[CHECK] Token: {token}, Current: {current_price}, Bought at: {bought_at}, Profit: {profit_ratio:.4f}")

            for i, step in enumerate(strategy_steps):
                if i in steps_executed:
                    continue

                target = step.get("target_multiplier")
                stop_loss = step.get("stop_loss", None)
                percentage = step.get("percentage", 100)

                trigger_price = bought_at * target
                step_amount = int(total_amount * (percentage / 100))
                
                try:
                    trigger_payload = {
                        "user": str(self.keypair.pubkey()),
                        "inputMint": token,
                        "outputMint": self.USDC_MINT,
                        "amount": str(step_amount),
                        "triggerCondition": ">",
                        "triggerPrice": str(trigger_price)
                    }
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://lite-api.jup.ag/trigger/v1/createOrder",
                            headers={"Content-Type": "application/json"},
                            json=trigger_payload
                        )
                        response.raise_for_status()
                        resp_json = response.json()
                        logging.debug(f"Trigger order válasz: {resp_json}")

                    msg = (
                        f"⏳ Trigger order elküldve\n"
                        f"Token: {token}\n"
                        f"Eladási ár: {trigger_price:.6f} USDC/token\n"
                        f"Mennyiség: {step_amount / 1_000_000:.6f} token"
                    )
                    logging.info(msg)
                    await self.send_telegram_message(msg)
                              
                    steps_executed.append(i)
                    
                except Exception as e:
                    logging.error(f"Trigger order hiba ({token}): {e}")
                    await self.send_telegram_message(f"❌ Trigger order hiba ({token}): {e}")             

            if len(steps_executed) != len(strategy_steps):
                trade["steps_executed"] = steps_executed
                remaining_trades.append(trade)

        self.active_trades = remaining_trades
        self.save_trades()

    async def send_telegram_message(self, message: str):
        if not self.telegram_token or not self.notify_chat_ids:
            return  # nincs megadva értesítési cím

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            for chat_id in self.notify_chat_ids:
                try:
                    await client.post(url, data={"chat_id": chat_id, "text": message})
                except Exception as e:
                    logging.error(f"Hiba az üzenetküldéskor ({chat_id}): {e}")

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
            await asyncio.sleep(10)

def main():
    bot = TradingBot()
    asyncio.run(bot.poll_telegram())

if __name__ == "__main__":
    main()

