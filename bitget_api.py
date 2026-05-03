"""
bitget_api.py
=============
Modulo de comunicacion con Bitget USDT Futures API v2.
Autenticacion: HMAC-SHA256 sobre (timestamp + method + path + body).
Sin librerias externas de exchange — solo requests + hmac estandar.

Uso:
    from bitget_api import BitgetClient, client_from_env
    client = client_from_env()
    client.set_leverage("BTCUSDT", 3, "isolated")
    client.place_order("BTCUSDT", direction="buy", size_usdt=100,
                       sl_price=80000.0, tp_price=95000.0)
"""

import hashlib
import hmac
import math
import os
import time
import base64
import json
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.bitget.com"
PRODUCT  = "USDT-FUTURES"
SYMBOL   = "BTCUSDT"


class BitgetClient:
    def __init__(self, api_key, secret, passphrase):
        self.api_key    = api_key
        self.secret     = secret
        self.passphrase = passphrase
        self.session    = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "locale":       "en-US",
        })

    # ── Firma HMAC-SHA256 ─────────────────────────────────────────────────────
    def _sign(self, timestamp, method, path, body=""):
        msg = timestamp + method.upper() + path + (body or "")
        return hmac.new(
            self.secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        # Bitget usa base64 del digest binario
    def _sign_b64(self, timestamp, method, path, body=""):
        msg = timestamp + method.upper() + path + (body or "")
        raw = hmac.new(
            self.secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(raw).decode()

    def _headers(self, method, path, body=""):
        ts = str(int(time.time() * 1000))
        sig = self._sign_b64(ts, method, path, body)
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       sig,
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": self.passphrase,
        }

    # ── HTTP helpers ──────────────────────────────────────────────────────────
    def _get(self, path, params=None):
        qs = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        full_path = path + qs
        headers   = self._headers("GET", full_path)
        resp      = self.session.get(BASE_URL + full_path,
                                     headers=headers, timeout=10)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"Bitget GET {path} failed: Invalid JSON response (Status {resp.status_code})")

        if data.get("code") != "00000":
            msg = data.get("msg", "No error message provided")
            raise RuntimeError(f"Bitget GET {path} error: {data.get('code')} - {msg}")
        return data

    def _post(self, path, body_dict):
        body_str = json.dumps(body_dict, separators=(",", ":"))
        headers  = self._headers("POST", path, body_str)
        resp     = self.session.post(BASE_URL + path,
                                     headers=headers,
                                     data=body_str, timeout=10)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"Bitget POST {path} failed: Invalid JSON response (Status {resp.status_code})")

        if data.get("code") != "00000":
            msg = data.get("msg", "No error message provided")
            raise RuntimeError(f"Bitget POST {path} error: {data.get('code')} - {msg}")
        return data

    # ── Precio de mercado ──────────────────────────────────────────────────────
    def get_price(self, symbol=SYMBOL):
        data = self._get("/api/v2/mix/market/ticker",
                         {"symbol": symbol, "productType": PRODUCT})
        ticker = data["data"]
        if isinstance(ticker, list):
            ticker = ticker[0]
        return float(ticker["lastPr"])

    # ── Info del contrato ─────────────────────────────────────────────────────
    def get_contract_info(self, symbol=SYMBOL):
        data = self._get("/api/v2/mix/market/contracts",
                         {"symbol": symbol, "productType": PRODUCT})
        contracts = data["data"]
        if isinstance(contracts, list):
            return contracts[0]
        return contracts

    def get_step_size(self, symbol=SYMBOL):
        info = self.get_contract_info(symbol)
        return float(info.get("sizeMultiplier", 0.001))

    def get_min_size(self, symbol=SYMBOL):
        info = self.get_contract_info(symbol)
        return float(info.get("minTradeNum", 0.001))

    def round_qty(self, qty, step):
        precision = max(0, -int(math.floor(math.log10(step))))
        return round(math.floor(qty / step) * step, precision)

    # ── Balance USDT ──────────────────────────────────────────────────────────
    def get_balance(self):
        """
        Devuelve dict con available, equity, unrealized_pl en USDT.
        """
        data = self._get("/api/v2/mix/account/account",
                         {"symbol": SYMBOL,
                          "productType": PRODUCT,
                          "marginCoin": "USDT"})
        acc = data["data"]
        return {
            "available":     float(acc.get("available", 0)),
            "equity":        float(acc.get("usdtEquity", acc.get("equity", 0))),
            "unrealized_pl": float(acc.get("unrealizedPL", 0)),
        }

    # ── Posiciones abiertas ────────────────────────────────────────────────────
    def get_positions(self, symbol=SYMBOL):
        data = self._get("/api/v2/mix/position/single-position",
                         {"symbol": symbol,
                          "productType": PRODUCT,
                          "marginCoin": "USDT"})
        positions = data.get("data", [])
        return [p for p in positions if float(p.get("total", 0)) > 0]

    def has_open_position(self, symbol=SYMBOL):
        return len(self.get_positions(symbol)) > 0

    # ── Configurar cuenta ──────────────────────────────────────────────────────
    def set_leverage(self, symbol, leverage, margin_mode="isolated"):
        return self._post("/api/v2/mix/account/set-leverage", {
            "symbol":      symbol,
            "productType": PRODUCT,
            "marginCoin":  "USDT",
            "leverage":    str(int(leverage)),
            "holdSide":    "long_short",
        })

    def set_margin_mode(self, symbol, margin_mode="isolated"):
        try:
            return self._post("/api/v2/mix/account/set-margin-mode", {
                "symbol":      symbol,
                "productType": PRODUCT,
                "marginCoin":  "USDT",
                "marginMode":  margin_mode,
            })
        except RuntimeError:
            return {}

    # ── Abrir posicion con TP y SL ───────────────────────────────────────────────
    def place_tpsl(self, symbol, direction, tp_price, sl_price):
        """
        Coloca TP y SL sobre una posicion ya abierta usando place-tpsl.
        direction: "buy" (posicion LONG) o "sell" (posicion SHORT)
        """
        hold_side = "long" if direction == "buy" else "short"
        try:
            self._post("/api/v2/mix/order/place-tpsl", {
                "symbol":           symbol,
                "productType":      PRODUCT,
                "marginCoin":       "USDT",
                "planType":         "pos_profit",
                "triggerPrice":     str(round(tp_price, 1)),
                "triggerType":      "mark_price",
                "executePrice":     "0",
                "holdSide":         hold_side,
                "size":             "",
                "rangeRate":        "",
            })
            print(f"  [BITGET] TP colocado: {tp_price:.0f}")
        except Exception as e:
            print(f"  [BITGET] Aviso TP: {e}")
        try:
            self._post("/api/v2/mix/order/place-tpsl", {
                "symbol":           symbol,
                "productType":      PRODUCT,
                "marginCoin":       "USDT",
                "planType":         "pos_loss",
                "triggerPrice":     str(round(sl_price, 1)),
                "triggerType":      "mark_price",
                "executePrice":     "0",
                "holdSide":         hold_side,
                "size":             "",
                "rangeRate":        "",
            })
            print(f"  [BITGET] SL colocado: {sl_price:.0f}")
        except Exception as e:
            print(f"  [BITGET] Aviso SL: {e}")

    def place_order(self, symbol, direction, size_usdt, sl_price, tp_price, leverage):
        """
        Abre posicion de mercado y luego coloca TP y SL por separado.
        direction: "buy" (LONG) o "sell" (SHORT)
        size_usdt: notional total en USDT (margen * leverage)
        """
        mkt_px = self.get_price(symbol)
        step   = self.get_step_size(symbol)
        min_sz = self.get_min_size(symbol)
        qty    = self.round_qty(size_usdt / mkt_px, step)
        qty    = max(qty, min_sz)

        if qty <= 0:
            raise ValueError(f"Qty={qty} invalida. size_usdt={size_usdt} demasiado pequeno")

        self.set_margin_mode(symbol, "isolated")
        self.set_leverage(symbol, leverage, "isolated")

        ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        client_id = f"mec_{direction[0].upper()}_{ts_str}"

        body = {
            "symbol":      symbol,
            "productType": PRODUCT,
            "marginMode":  "isolated",
            "marginCoin":  "USDT",
            "size":        str(qty),
            "side":        direction,
            "tradeSide":   "open",
            "orderType":   "market",
            "clientOid":   client_id,
        }
        data = self._post("/api/v2/mix/order/place-order", body)
        order_id = data["data"].get("orderId")

        print(f"  [BITGET] Orden OK: {direction.upper()} {qty} BTC @ ~{mkt_px:.0f}"
              f"  lev={leverage}x  orderId={order_id}")

        # Esperar un momento para que la posicion este activa antes de poner TP/SL
        import time; time.sleep(1)
        self.place_tpsl(symbol, direction, tp_price, sl_price)

        print(f"  [BITGET] SL={sl_price:.0f}  TP={tp_price:.0f}")

        return {
            "orderId":  order_id,
            "qty":      qty,
            "entry_px": mkt_px,
        }

    # ── Cerrar posicion completa ───────────────────────────────────────────────
    def close_position(self, symbol=SYMBOL):
        """Cierra toda la posicion abierta con flash close."""
        try:
            data = self._post("/api/v2/mix/order/close-positions", {
                "symbol":      symbol,
                "productType": PRODUCT,
            })
            print(f"  [BITGET] Posicion cerrada para {symbol}")
            return data
        except Exception as e:
            print(f"  [BITGET] Error cerrando posicion: {e}")
            return {}

    # ── PnL realizado ─────────────────────────────────────────────────────────
    def get_realized_pnl(self, symbol=SYMBOL, limit=5):
        try:
            data = self._get("/api/v2/mix/order/fill-history", {
                "symbol":      symbol,
                "productType": PRODUCT,
                "pageSize":    str(limit),
            })
            return data.get("data", {}).get("fillList", [])
        except Exception:
            return []


# ── Fabrica desde variables de entorno ────────────────────────────────────────
def client_from_env():
    key = os.environ.get("BITGET_API_KEY",    "")
    sec = os.environ.get("BITGET_API_SECRET", "")
    pp  = os.environ.get("BITGET_PASSPHRASE", "")
    if not all([key, sec, pp]):
        raise EnvironmentError(
            "Faltan: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE"
        )
    return BitgetClient(key, sec, pp)
