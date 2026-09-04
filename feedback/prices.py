"""Prices, for checking predictions against what actually happened.

Binance's public endpoints need no key, which matters: the feedback loop runs on
every open prediction every few minutes, and putting that behind a rate-limited
credential would make the loop the thing that breaks first.

Three problems this module exists to solve.

**The baseline must be the price when the news broke.** If the observer starts
twenty minutes after an article was analysed and uses the current price as the
baseline, the move it was supposed to measure has already happened and is now
invisible — every prediction would score as a miss for the fast events and a hit
for the slow ones, which is exactly backwards. So a baseline is read from the
one-minute candle containing the prediction's timestamp, not from the spot ticker.

**Not every asset the models name is tradeable here.** Stage 3 legitimately
returns things like ``CRYPTO`` for a market-wide event, or a token that is not
listed on Binance. These are not errors and must not be retried forever; they are
recorded as unpriceable and the prediction is closed unscored.

**One request, not forty.** Spot prices for a whole batch come back in a single
call, because forty individual requests every few minutes is how a public
endpoint starts refusing them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from services.config import FeedbackSettings, settings as global_settings

log = logging.getLogger(__name__)

#: Assets that are not a tradeable symbol. The models use these deliberately —
#: "CRYPTO" for a market-wide event is the right answer to "what is affected" —
#: so they are skipped rather than treated as bad output.
UNPRICEABLE = frozenset(
    {
        "CRYPTO", "MARKET", "ALTCOINS", "ALTS", "DEFI", "NFT", "MEMECOINS",
        "STABLECOINS", "TOTAL", "TOTAL3", "L1", "L2", "AI", "RWA", "GAMING",
        "USD", "EUR", "GOLD", "SPX", "NASDAQ", "EQUITIES", "BONDS",
    }
)

#: Assets whose price against the quote asset is ~1 by construction. Scoring a
#: direction call on USDT/USDT is meaningless.
PEGGED = frozenset({"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "PYUSD"})

#: Names the models use that differ from the exchange's ticker.
ALIASES = {
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
    "SOLANA": "SOL",
    "RIPPLE": "XRP",
    "XBT": "BTC",
    "MATIC": "POL",
    "BNB": "BNB",
    "CARDANO": "ADA",
    "DOGECOIN": "DOGE",
    "POLKADOT": "DOT",
    "AVALANCHE": "AVAX",
    "CHAINLINK": "LINK",
    "LITECOIN": "LTC",
    "TONCOIN": "TON",
    "SHIBA": "SHIB",
    "SHIBA INU": "SHIB",
}


class PriceUnavailable(Exception):
    """The price could not be read, and it is not the asset's fault.

    Distinct from an unpriceable asset: this is a network or exchange problem, so
    the caller should try again later rather than closing the prediction.
    """


@dataclass(frozen=True)
class Candle:
    """One minute of trading, which is the resolution a baseline needs."""

    open_time: datetime
    open: float
    high: float
    low: float
    close: float

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2


def normalise_asset(asset: str) -> str:
    """Model output to an exchange base asset."""
    cleaned = (asset or "").strip().upper().replace("$", "")
    cleaned = cleaned.split("/")[0].split("-")[0].strip()
    return ALIASES.get(cleaned, cleaned)


def is_priceable(asset: str) -> bool:
    """Whether a price for this asset exists to be compared against.

    Called before anything is fetched, so a market-wide prediction costs no
    requests at all.
    """
    base = normalise_asset(asset)
    if not base or len(base) > 12:
        return False
    return base not in UNPRICEABLE and base not in PEGGED


class PriceSource:
    """Reads spot and historical prices from a public exchange API."""

    def __init__(
        self,
        config: FeedbackSettings | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or global_settings.feedback
        self.base_url = self.config.binance_base_url.rstrip("/")
        self.quote = self.config.quote_asset.upper()
        self._http = http
        self._owns_http = http is None
        #: Symbols the exchange rejected, so a delisted token is asked for once.
        self._unknown: set[str] = set()
        self.requests = 0

    async def __aenter__(self) -> "PriceSource":
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                limits=httpx.Limits(max_connections=8),
                headers={"accept": "application/json"},
            )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("PriceSource must be used as an async context manager")
        return self._http

    def symbol_for(self, asset: str) -> str:
        """``BTC`` -> ``BTCUSDT``. Empty when there is no such pair to ask for."""
        base = normalise_asset(asset)
        if not is_priceable(base):
            return ""
        symbol = f"{base}{self.quote}"
        return "" if symbol in self._unknown else symbol

    # ------------------------------------------------------------------ requests

    async def _get(self, path: str, params: dict) -> object:
        """One GET with a short retry, translating failure into one exception type."""
        last: str = ""
        for attempt in range(3):
            try:
                self.requests += 1
                response = await self.http.get(f"{self.base_url}{path}", params=params)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise PriceUnavailable(f"malformed price response: {exc}") from None

                if response.status_code in (400, 404):
                    # The exchange saying "no such symbol". Not retryable, and
                    # remembering it keeps a delisted token from being asked for
                    # on every pass forever.
                    symbol = str(params.get("symbol", ""))
                    if symbol:
                        self._unknown.add(symbol)
                    raise PriceUnavailable(f"unknown symbol {symbol or path}")

                if response.status_code in (418, 429):
                    # Binance bans on repeat offence, so this waits properly.
                    delay = float(response.headers.get("retry-after") or 5.0)
                    log.warning("price API rate limited, waiting %.0fs", delay)
                    await asyncio.sleep(min(delay, 30.0))
                    last = f"HTTP {response.status_code}"
                    continue

                last = f"HTTP {response.status_code}"

            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))

        raise PriceUnavailable(last or "no response")

    # -------------------------------------------------------------------- spot

    async def spot(self, asset: str) -> float:
        """Current price of one asset."""
        symbol = self.symbol_for(asset)
        if not symbol:
            raise PriceUnavailable(f"{asset} is not a tradeable symbol")

        payload = await self._get("/api/v3/ticker/price", {"symbol": symbol})
        if isinstance(payload, dict) and payload.get("price"):
            return float(payload["price"])
        raise PriceUnavailable(f"no price in response for {symbol}")

    async def spot_many(self, assets: list[str]) -> dict[str, float]:
        """Prices for a batch in one request, keyed by normalised asset.

        Assets with no symbol are simply absent from the result — the caller has
        to handle a missing price anyway, and raising here would make one
        unlisted token cost the whole batch its prices.
        """
        symbols: dict[str, str] = {}
        for asset in assets:
            symbol = self.symbol_for(asset)
            if symbol:
                symbols[symbol] = normalise_asset(asset)

        if not symbols:
            return {}

        # The batch form takes a JSON array with no spaces.
        payload = await self._get(
            "/api/v3/ticker/price",
            {"symbols": json.dumps(sorted(symbols), separators=(",", ":"))},
        )

        prices: dict[str, float] = {}
        for entry in payload if isinstance(payload, list) else []:
            if not isinstance(entry, dict):
                continue
            asset = symbols.get(str(entry.get("symbol", "")))
            if asset and entry.get("price"):
                prices[asset] = float(entry["price"])

        missing = set(symbols.values()) - set(prices)
        if missing:
            log.debug("no price returned for %s", ", ".join(sorted(missing)))
        return prices

    # -------------------------------------------------------------- historical

    async def candle_at(self, asset: str, when: datetime) -> Candle:
        """The one-minute candle covering ``when``.

        This is what a baseline is read from. The candle's *open* is used rather
        than its close, because the open is the price at the start of the minute
        the news carries — using the close would already include part of the move
        being measured.
        """
        symbol = self.symbol_for(asset)
        if not symbol:
            raise PriceUnavailable(f"{asset} is not a tradeable symbol")

        moment = when.astimezone(timezone.utc).replace(second=0, microsecond=0)
        start = int(moment.timestamp() * 1000)

        payload = await self._get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": start,
                "endTime": start + 60_000,
                "limit": 1,
            },
        )

        rows = payload if isinstance(payload, list) else []
        if not rows or not isinstance(rows[0], list) or len(rows[0]) < 5:
            raise PriceUnavailable(f"no candle for {symbol} at {moment.isoformat()}")

        row = rows[0]
        return Candle(
            open_time=datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
        )

    async def price_at(self, asset: str, when: datetime) -> float:
        """Price at a past moment, or the spot price when the moment is now.

        A prediction made seconds ago has no completed candle yet, so asking for
        one would fail on exactly the predictions that matter most — the fresh
        ones.
        """
        age = datetime.now(timezone.utc) - when.astimezone(timezone.utc)
        if age < timedelta(minutes=2):
            return await self.spot(asset)
        candle = await self.candle_at(asset, when)
        return candle.open

    async def move_since(self, asset: str, baseline: float, when: datetime) -> tuple[float, float]:
        """Price now and the percentage change from ``baseline``."""
        if baseline <= 0:
            raise PriceUnavailable(f"no usable baseline for {asset}")
        price = await self.price_at(asset, when)
        return price, pct_change(baseline, price)


def pct_change(baseline: float, price: float) -> float:
    """Signed percentage move, as the models express magnitudes."""
    if baseline <= 0:
        return 0.0
    return (price - baseline) / baseline * 100.0


__all__ = [
    "ALIASES",
    "PEGGED",
    "UNPRICEABLE",
    "Candle",
    "PriceSource",
    "PriceUnavailable",
    "is_priceable",
    "normalise_asset",
    "pct_change",
]
