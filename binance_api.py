"""
binance_api.py
==============
Modulo de comunicacion con Binance USDT-M Futures API (fapi).
Autenticacion: HMAC-SHA256 sobre query string (sin libreria externa).

CAMBIO OBLIGATORIO de Binance (efectivo 2025-12-09):
  STOP_MARKET y TAKE_PROFIT_MARKET ya NO van por /fapi/v1/order.
  Ahora usan /fapi/v1/algoOrder. Error -4120 si se usa el viejo.
  Este modulo lo implementa correctamente.

Uso:
    from binance_api import BinanceClient, client_from_env
    client = client_from_env()
    client.set_leverage("BTCUSDT", 3)
    client.place_order("BTCUSDT", side="BUY", size_usdt=300,
                       sl_price=80000.0, tp_price=95000.0, leverage=3)
"""

import hashlib
import hmac
import math
import os
import time
import urllib.parse
from datetime import datetime, timezone

import requests

BASE_URL    = "https://fapi.binance.com"
SYMBOL      = "BTCUSDT"
RECV_WINDOW = 5000


class BinanceClient:
    def __init__(self, api_key, api_secret):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.session    = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })

    # ── Firma HMAC-SHA256 ─────────────────────────────────────────────────────
    def _sign(self, params):
        qs = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            qs.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _ts(self):
        return int(time.time() * 1000)

    # ── HTTP helpers ──────────────────────────────────────────────────────────
    def _get(self, path, params=None, signed=False):
        params = params or {}
        if signed:
            params["timestamp"]  = self._ts()
            params["recvWindow"] = RECV_WINDOW
            params["signature"]  = self._sign(params)
        resp = self.session.get(BASE_URL + path, params=params, timeout=10)
        data = resp.json()
        if isinstance(data, dict) and int(data.get("code", 0)) < 0:
            raise RuntimeError(f"Binance GET {path} error: {data}")
        return data

    def _post(self, path, params):
        params["timestamp"]  = self._ts()
        params["recvWindow"] = RECV_WINDOW
        params["signature"]  = self._sign(params)
        resp = self.session.post(
            BASE_URL + path,
            data=urllib.parse.urlencode(params),
            timeout=10,
        )
        data = resp.json()
        if isinstance(data, dict) and int(data.get("code", 0)) < 0:
            raise RuntimeError(f"Binance POST {path} error: {data}")
        return data

    def _delete(self, path, params):
        params["timestamp"]  = self._ts()
        params["recvWindow"] = RECV_WINDOW
        params["signature"]  = self._sign(params)
        resp = self.session.delete(BASE_URL + path, params=params, timeout=10)
        data = resp.json()
        if isinstance(data, dict) and int(data.get("code", 0)) < 0:
            raise RuntimeError(f"Binance DELETE {path} error: {data}")
        return data

    # ── Precio de mercado ──────────────────────────────────────────────────────
    def get_price(self, symbol=SYMBOL):
        data = self._get("/fapi/v2/ticker/price", {"symbol": symbol})
        return float(data["price"])

    # ── Step size del contrato ─────────────────────────────────────────────────
    def get_step_size(self, symbol=SYMBOL):
        data = self._get("/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return float(f["stepSize"])
        return 0.001

    def round_qty(self, qty, step):
        precision = max(0, -int(math.floor(math.log10(step))))
        return round(math.floor(qty / step) * step, precision)

    # ── Balance USDT ──────────────────────────────────────────────────────────
    def get_balance(self):
        data = self._get("/fapi/v3/account", signed=True)
        for asset in data.get("assets", []):
            if asset["asset"] == "USDT":
                return {
                    "available":     float(asset["availableBalance"]),
                    "wallet":        float(asset["walletBalance"]),
                    "unrealized_pl": float(asset["unrealizedProfit"]),
                }
        return {"available": 0.0, "wallet": 0.0, "unrealized_pl": 0.0}

    # ── Posiciones abiertas ────────────────────────────────────────────────────
    def get_positions(self, symbol=SYMBOL):
        data = self._get("/fapi/v3/positionRisk", {"symbol": symbol}, signed=True)
        return [p for p in data if float(p.get("positionAmt", 0)) != 0]

    def has_open_position(self, symbol=SYMBOL):
        return len(self.get_positions(symbol)) > 0

    # ── Configurar cuenta ──────────────────────────────────────────────────────
    def set_leverage(self, symbol, leverage):
        return self._post("/fapi/v1/leverage", {
            "symbol":   symbol,
            "leverage": str(int(leverage)),
        })

    def set_margin_type(self, symbol, margin_type="ISOLATED"):
        # Binance lanza -4046 si ya esta en ese modo, se ignora
        try:
            return self._post("/fapi/v1/marginType", {
                "symbol":     symbol,
                "marginType": margin_type,
            })
        except RuntimeError as e:
            if "-4046" in str(e):
                return {}
            raise

    # ── Orden de mercado (abre posicion) ──────────────────────────────────────
    def _open_market(self, symbol, side, qty):
        params = {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": str(qty),
        }
        data = self._post("/fapi/v1/order", params)
        print(f"  [BINANCE] MARKET {side} {qty} BTC  orderId={data.get('orderId')}")
        return data

    # ── algoOrder para SL/TP (OBLIGATORIO desde 2025-12-09) ───────────────────
    def _place_algo(self, symbol, side, order_type, qty, trigger_price):
        """
        Coloca SL o TP via /fapi/v1/algoOrder con reduceOnly=true.
        order_type: "STOP_MARKET" (SL) o "TAKE_PROFIT_MARKET" (TP)
        """
        ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        client_id = f"mec_{order_type[:2].lower()}_{ts_str}"
        params = {
            "symbol":       symbol,
            "side":         side,
            "orderType":    order_type,
            "quantity":     str(qty),
            "triggerPrice": f"{trigger_price:.1f}",
            "workingType":  "MARK_PRICE",
            "reduceOnly":   "true",
            "clientAlgoId": client_id,
        }
        data = self._post("/fapi/v1/algoOrder", params)
        algo_id = data.get("algoId") or (data.get("data") or {}).get("algoId")
        print(f"  [BINANCE] {order_type}: trigger={trigger_price:.0f}  algoId={algo_id}")
        return data

    # ── Flujo completo: mercado + SL + TP ─────────────────────────────────────
    def place_order(self, symbol, side, size_usdt, sl_price, tp_price, leverage):
        """
        Abre posicion de mercado y coloca SL + TP automaticos.

        side:      "BUY" para LONG, "SELL" para SHORT
        size_usdt: notional total en USDT (margen * leverage)
        """
        mkt_px = self.get_price(symbol)
        step   = self.get_step_size(symbol)
        qty    = self.round_qty(size_usdt / mkt_px, step)
        if qty <= 0:
            raise ValueError(f"Qty={qty} invalida. size_usdt={size_usdt} demasiado pequeno")

        self.set_margin_type(symbol, "ISOLATED")
        self.set_leverage(symbol, leverage)

        order      = self._open_market(symbol, side, qty)
        close_side = "SELL" if side == "BUY" else "BUY"

        # SL
        sl_data = self._place_algo(symbol, close_side, "STOP_MARKET",         qty, sl_price)
        # TP
        tp_data = self._place_algo(symbol, close_side, "TAKE_PROFIT_MARKET",  qty, tp_price)

        print(f"  [BINANCE] Posicion abierta OK: {side} {qty} BTC @ ~{mkt_px:.0f}"
              f"  SL={sl_price:.0f}  TP={tp_price:.0f}  lev={leverage}x")

        return {
            "orderId":    order.get("orderId"),
            "qty":        qty,
            "entry_px":   mkt_px,
            "sl_algo_id": sl_data.get("algoId"),
            "tp_algo_id": tp_data.get("algoId"),
        }

    # ── Cancelar algoOrders abiertos ──────────────────────────────────────────
    def cancel_algo_orders(self, symbol=SYMBOL):
        try:
            data   = self._get("/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)
            orders = data if isinstance(data, list) else data.get("orders", [])
            for o in orders:
                algo_id = o.get("algoId")
                if algo_id:
                    self._delete("/fapi/v1/algoOrder", {"algoId": str(algo_id)})
                    print(f"  [BINANCE] algoOrder {algo_id} cancelado")
        except Exception as e:
            print(f"  [BINANCE] Error cancelando algoOrders: {e}")

    # ── Cerrar posicion completa ───────────────────────────────────────────────
    def close_position(self, symbol=SYMBOL):
        """Cancela SL/TP pendientes y cierra por mercado."""
        self.cancel_algo_orders(symbol)
        positions = self.get_positions(symbol)
        for pos in positions:
            amt = float(pos.get("positionAmt", 0))
            if amt == 0:
                continue
            close_side = "SELL" if amt > 0 else "BUY"
            step = self.get_step_size(symbol)
            qty  = self.round_qty(abs(amt), step)
            order = self._open_market(symbol, close_side, qty)
            print(f"  [BINANCE] Posicion cerrada: {close_side} {qty} BTC")
            return order
        print("  [BINANCE] No habia posicion que cerrar")
        return {}

    # ── PnL realizado (para sincronizar estado) ────────────────────────────────
    def get_realized_pnl(self, symbol=SYMBOL, limit=5):
        data = self._get("/fapi/v1/income", {
            "symbol":     symbol,
            "incomeType": "REALIZED_PNL",
            "limit":      str(limit),
        }, signed=True)
        return data if isinstance(data, list) else []


# ── Fabrica desde variables de entorno ────────────────────────────────────────
def client_from_env():
    key = os.environ.get("BINANCE_API_KEY",    "")
    sec = os.environ.get("BINANCE_API_SECRET", "")
    if not all([key, sec]):
        raise EnvironmentError(
            "Faltan: BINANCE_API_KEY, BINANCE_API_SECRET"
        )
    return BinanceClient(key, sec)
