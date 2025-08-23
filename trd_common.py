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
from solders.message import VersionedMessage
from solders.transaction import VersionedTransaction
from solders.signature import Signature

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
           
        # ⛔ Szűrés: USDC és SOL tokeneket ne próbáljuk megvenni
        IGNORED_TOKENS = {
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        }

        if token_address in IGNORED_TOKENS:
            logging.info(f"⛔ Kihagyott token: {token_address} (USDC vagy SOL)")
            await self.send_telegram_message(f"⛔ Kihagyott token: {token_address}")
            return

        if any(t["token"] == token_address for t in self.active_trades):
            logging.warning(f"⚠️ Már létezik pozíció ezzel a tokennel: {token_address}")
            await self.send_telegram_message(f"⚠️ Már nyitott pozíció van: {token_address}")
            return
        
    
#        await asyncio.sleep(random.uniform(1, 3))

        max_retries = 10
        for attempt in range(max_retries):
            try:
                amount = float(self.strategy.get("buy_amount_usdc", 1))
                slippage = float(self.strategy.get("slippage", 0.5))
                raw_amount = int(amount * 1_000_000)
                
                order = UltraOrderRequest(
                    input_mint=self.USDC_MINT,
                    output_mint=token_address,
                    amount=raw_amount,
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
                token_decimals = await self.get_token_decimals(token_address)
                adjusted_amount = output_amount
                bought_at = (amount*1000000) / output_amount
                #bought_at = (amount / 1_000_000) / (output_amount / 1_000_000)  # Valós árfolyam: USDC/token
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
                    f"Összeg: {amount:.12f} token\n"
                    f"Árfolyam: {bought_at:.12f} USDC/token"
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

    async def get_token_decimals(self, mint: str) -> int:
        try:
            pubkey = Pubkey.from_string(mint)
            resp = await self.client.get_token_supply(pubkey)
            if resp.value and resp.value.decimals is not None:
                return resp.value.decimals
        except Exception as e:
            logging.error(f"[!] Hiba a decimális lekérdezésénél: {e}")
        return 6  # alapértelmezett fallback

      
    
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
            token_decimals = await self.get_token_decimals(token_address)
            adjusted_amount = amount / (10 ** token_decimals)
            msg = (
                  f"✅ Eladás sikeres\n"
                  f"Token: {token_address}\n"
                  f"Eladott összeg: {adjusted_amount:.12f} token\n"
                  f"Árfolyam: {price:.12f} USDC/token"
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
                
                # --- DEBUG: logold ki az összes alapváltozót és a képleteket ---
                logging.debug(
                    "[TRIGGER_DBG] token=%s step=%d | bought_at=%.12f target=%.6f -> trigger_price=%.12f | total_amount=%d percentage=%.2f -> step_amount=%d",
                    token, i, float(bought_at), float(target), float(trigger_price), int(total_amount), float(percentage), int(step_amount)
                )

                # Típusok (ha véletlenül string vagy None csúszik be)
                logging.debug(
                    "[TRIGGER_DBG] types | bought_at=%s target=%s percentage=%s total_amount=%s",
                    type(bought_at).__name__, type(target).__name__, type(percentage).__name__, type(total_amount).__name__
                )

                # takingAmount képlete és eredménye
                taking_amount = int(trigger_price * step_amount)
                logging.debug(
                    "[TRIGGER_DBG] taking_amount = int(trigger_price * step_amount) = int(%.12f * %d) = %d",
                    float(trigger_price), int(step_amount), int(taking_amount)
                )

                # Védőkorlátok: ha bármelyik érték 0/negatív, ne küldjünk ki ordert, logoljuk miért
                if bought_at is None or target is None:
                    logging.error("[TRIGGER_DBG] Missing bought_at or target; skip token=%s step=%d", token, i)
                    continue
                if bought_at <= 0 or target <= 0:
                    logging.error("[TRIGGER_DBG] Non-positive bought_at (%.12f) or target (%.6f); skip token=%s step=%d",
                                  float(bought_at), float(target), token, i)
                    continue
                if step_amount <= 0:
                    logging.error("[TRIGGER_DBG] step_amount <= 0 (%d); skip token=%s step=%d", int(step_amount), token, i)
                    continue
                if taking_amount <= 0:
                    logging.error("[TRIGGER_DBG] taking_amount <= 0 (%d); trigger_price*step_amount nem pozitív; skip token=%s step=%d",
                                  int(taking_amount), token, i)
                    continue
                # --- END DEBUG ---

                max_trigger_attempts = 10
                trigger_attempts = 0

                while trigger_attempts < max_trigger_attempts:
                    trigger_attempts += 1
                
                    try:
                        trigger_payload = {
                            "inputMint": token,
                            "outputMint": self.USDC_MINT,
                            "maker": str(self.keypair.pubkey()),
                            "payer": str(self.keypair.pubkey()),
                            "params": {
                                "makingAmount": str(step_amount),  # pl. 1_000_000
                                "takingAmount": str(int(trigger_price * step_amount))  # elérni kívánt USDC mennyiség
                            },
                            "computeUnitPrice": "auto"
                        }
                        
                        logging.debug("[TRIGGER_DBG] POST /createOrder payload:\n%s", json.dumps(trigger_payload, indent=2))

                        
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                "https://lite-api.jup.ag/trigger/v1/createOrder",
                                headers={"Content-Type": "application/json"},
                                json=trigger_payload
                            )
                            
                            
                            try:
                                response_json = response.json()
                            except Exception as e:
                                raw_body = await response.aread()
                                logging.error(f"❌ createOrder válasz nem JSON ({e}): {raw_body.decode('utf-8', errors='ignore')}")
                                await self.send_telegram_message(f"❌ createOrder nem JSON válasz: {e}")
                                return
     
                            # JSON kiírás teljes egészében
                            logging.debug(f"[TRIGGER_DBG] createOrder válasz JSON:\n{json.dumps(response_json, indent=2)}")
      
                            # Ellenőrizzük, van-e benne transaction
                            transaction_b64 = response_json.get("transaction")
                            request_id = response_json.get("requestId")
                              
                            
                            try:
                                response.raise_for_status()
                                request_id = response_json.get("requestId")
                                if request_id:
                                    transaction_base64 = response_json.get("transaction")
                                    if not transaction_base64:
                                        logging.error("❌ Nincs transaction a createOrder válaszban: %s", response_json)
                                        await self.send_telegram_message(f"❌ Nincs transaction a trigger válaszban: {request_id}")
                                        continue


                                    try:
                                        tx_bytes = base64.b64decode(transaction_base64)
                                        versioned_tx = VersionedTransaction.from_bytes(tx_bytes)

                                        # Nem kell külön signature-t létrehozni, helyette új aláírt tranzakciót készítünk
                                        signed_tx = VersionedTransaction(versioned_tx.message, [self.keypair])
                                        signed_tx_b64 = base64.b64encode(bytes(signed_tx)).decode("utf-8")

                                        # Execute trigger order
                                        exec_payload = {
                                            "signedTransaction": signed_tx_b64,
                                            "requestId": request_id
                                        }
                                        exec_response = await client.post(
                                            "https://lite-api.jup.ag/trigger/v1/execute",
                                            headers={"Content-Type": "application/json"},
                                            json=exec_payload
                                        )
                                        exec_response.raise_for_status()

                                        logging.info(f"✅ Trigger order aláírva és végrehajtva: {request_id}")
                                        await self.send_telegram_message(f"✅ Trigger order elküldve: {request_id}")
                                        steps_executed.append(i)
                                        break

                                    except Exception as e:
                                        logging.error(f"❌ Trigger aláírás vagy küldés hiba ({request_id}): {e}")
                                        await self.send_telegram_message(f"❌ Aláírás vagy küldés hiba: {request_id}")
                                        continue
                                                                                               
                                logging.debug(f"Trigger order válasz: {response_json}")

                                msg = (
                                   f"⏳ Trigger order elküldve\n"
                                   f"Token: {token}\n"
                                   f"Eladási ár: {trigger_price:.12f} USDC/token\n"
                                   f"Mennyiség: {step_amount / 1_000_000:.12f} token"
                                )
                                logging.info(msg)
                                await self.send_telegram_message(msg)
                                steps_executed.append(i)
                                                       
                            except httpx.HTTPStatusError as e:
                                try:
                                    error_json = response.json()
                                    error_msg = error_json.get("error", "Ismeretlen hiba")
                                    cause = error_json.get("cause", "Ismeretlen ok")
                                    msg = f"❌ Trigger order hiba ({token}): {error_msg} – {cause}"
                                except Exception:
                                    msg = f"❌ Trigger order HTTP hiba ({token}): {str(e)}"

                                logging.error(msg)
                                await self.send_telegram_message(msg)

                            except Exception as e:
                                logging.error(f"⚠️ Trigger attempt {trigger_attempts}/{max_trigger_attempts} hiba: {e}")
                                await self.send_telegram_message(
                                    f"⚠️ Trigger attempt {trigger_attempts}/{max_trigger_attempts} sikertelen {token} step={i}: {e}"
                                )
                                if trigger_attempts >= max_trigger_attempts:
                                    logging.error(f"❌ Max. trigger próbálkozás elérve a step-re: token={token}, step={i}")
                                await asyncio.sleep(1)  # opcionális várakozás
                     
                    except Exception as e:
                        logging.error(f"⚠️ Trigger külső hiba ({token}): {e}")
                        await self.send_telegram_message(f"⚠️ Trigger külső hiba ({token}): {e}")
                        await asyncio.sleep(1)
                                          
            trade["steps_executed"] = steps_executed
            if len(steps_executed) != len(strategy_steps):
                remaining_trades.append(trade)
            else:
                await self.send_telegram_message(f"✅ Minden eladási lépés teljesült, pozíció lezárva: {token}")
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
