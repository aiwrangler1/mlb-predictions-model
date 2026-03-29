"""
Market Data Layer - Integrates The Odds API, Kalshi, and Polymarket
for real sportsbook odds comparison against model predictions.
"""
import httpx
import asyncio
from typing import Optional
from datetime import datetime, date, timedelta

# ─── The Odds API ──────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = None  # Set via API or env

# ─── Kalshi ────────────────────────────────────────────────────
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# ─── Cache ─────────────────────────────────────────────────────
_market_cache = {}


def set_odds_api_key(key: str):
    global ODDS_API_KEY
    ODDS_API_KEY = key


async def _fetch(url: str, timeout: float = 10.0) -> dict | list:
    if url in _market_cache:
        return _market_cache[url]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            _market_cache[url] = data
            return data
    except Exception as e:
        print(f"Market fetch error ({url[:80]}): {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# THE ODDS API
# ═══════════════════════════════════════════════════════════════

async def get_odds_api_mlb(markets: str = "h2h,spreads,totals") -> list:
    """
    Fetch MLB odds from The Odds API.
    Returns list of games with odds from multiple bookmakers.
    
    Markets: h2h (moneyline), spreads (run line), totals (over/under)
    """
    if not ODDS_API_KEY:
        return []
    
    url = (
        f"{ODDS_API_BASE}/sports/baseball_mlb/odds"
        f"?regions=us,us2&markets={markets}"
        f"&oddsFormat=american&apiKey={ODDS_API_KEY}"
    )
    data = await _fetch(url)
    if not isinstance(data, list):
        return []
    
    results = []
    for game in data:
        parsed = {
            "odds_api_id": game.get("id", ""),
            "sport": game.get("sport_title", "MLB"),
            "commence_time": game.get("commence_time", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "bookmakers": [],
            "consensus": {},
        }
        
        all_away_ml = []
        all_home_ml = []
        all_totals = []
        all_away_spread = []
        all_home_spread = []
        
        for bk in game.get("bookmakers", []):
            book = {
                "name": bk.get("title", ""),
                "key": bk.get("key", ""),
                "markets": {}
            }
            
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                outcomes = mkt.get("outcomes", [])
                
                if mkt_key == "h2h":
                    for o in outcomes:
                        if o["name"] == game.get("away_team"):
                            book["markets"]["away_ml"] = o.get("price", 0)
                            all_away_ml.append(o.get("price", 0))
                        elif o["name"] == game.get("home_team"):
                            book["markets"]["home_ml"] = o.get("price", 0)
                            all_home_ml.append(o.get("price", 0))
                
                elif mkt_key == "totals":
                    for o in outcomes:
                        if o["name"] == "Over":
                            book["markets"]["over"] = o.get("price", 0)
                            book["markets"]["total"] = o.get("point", 0)
                            all_totals.append(o.get("point", 0))
                        elif o["name"] == "Under":
                            book["markets"]["under"] = o.get("price", 0)
                
                elif mkt_key == "spreads":
                    for o in outcomes:
                        if o["name"] == game.get("away_team"):
                            book["markets"]["away_spread"] = o.get("point", 0)
                            book["markets"]["away_spread_price"] = o.get("price", 0)
                            all_away_spread.append(o.get("point", 0))
                        elif o["name"] == game.get("home_team"):
                            book["markets"]["home_spread"] = o.get("point", 0)
                            book["markets"]["home_spread_price"] = o.get("price", 0)
                            all_home_spread.append(o.get("point", 0))
            
            parsed["bookmakers"].append(book)
        
        # Consensus (median across books)
        if all_away_ml:
            sorted_away = sorted(all_away_ml)
            sorted_home = sorted(all_home_ml)
            mid = len(sorted_away) // 2
            parsed["consensus"]["away_ml"] = sorted_away[mid]
            parsed["consensus"]["home_ml"] = sorted_home[mid]
            parsed["consensus"]["away_implied"] = american_to_prob(sorted_away[mid])
            parsed["consensus"]["home_implied"] = american_to_prob(sorted_home[mid])
        
        if all_totals:
            parsed["consensus"]["total"] = sorted(all_totals)[len(all_totals) // 2]
        
        if all_away_spread:
            parsed["consensus"]["away_spread"] = sorted(all_away_spread)[len(all_away_spread) // 2]
        
        parsed["consensus"]["num_books"] = len(parsed["bookmakers"])
        
        # Best lines (best odds available)
        if all_away_ml:
            parsed["best_away_ml"] = max(all_away_ml)
            parsed["best_home_ml"] = max(all_home_ml)
            # Find which books have the best lines
            for bk in parsed["bookmakers"]:
                if bk["markets"].get("away_ml") == parsed["best_away_ml"]:
                    parsed["best_away_book"] = bk["name"]
                if bk["markets"].get("home_ml") == parsed["best_home_ml"]:
                    parsed["best_home_book"] = bk["name"]
        
        results.append(parsed)
    
    return results


async def get_odds_api_events() -> list:
    """Get list of upcoming MLB events (for event-specific prop odds)."""
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/events?apiKey={ODDS_API_KEY}"
    return await _fetch(url)


async def get_odds_api_player_props(event_id: str, markets: str = "batter_home_runs,pitcher_strikeouts") -> dict:
    """Get player prop odds for a specific game."""
    if not ODDS_API_KEY:
        return {}
    url = (
        f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds"
        f"?regions=us&markets={markets}"
        f"&oddsFormat=american&apiKey={ODDS_API_KEY}"
    )
    return await _fetch(url)


# ═══════════════════════════════════════════════════════════════
# KALSHI
# ═══════════════════════════════════════════════════════════════

async def get_kalshi_mlb_games() -> list:
    """
    Fetch MLB game winner markets from Kalshi.
    Series ticker: KXMLBGAME
    """
    url = f"{KALSHI_BASE}/markets?limit=100&status=open&series_ticker=KXMLBGAME"
    data = await _fetch(url)
    if isinstance(data, dict):
        markets = data.get("markets", [])
    else:
        return []
    
    # Group by event (each game has YES/NO for each team)
    events = {}
    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        # Extract game identifier from ticker
        # Format: KXMLBGAME-26MAR302210DETAZ-DET
        parts = ticker.rsplit("-", 1)
        if len(parts) == 2:
            event_key = parts[0]
            team_code = parts[1]
        else:
            event_key = ticker
            team_code = ""
        
        if event_key not in events:
            events[event_key] = {
                "event_ticker": event_key,
                "title": title,
                "close_time": m.get("close_time", ""),
                "teams": {}
            }
        
        yes_bid = m.get("yes_bid", 0) or 0
        yes_ask = m.get("yes_ask", 0) or 0
        no_bid = m.get("no_bid", 0) or 0
        no_ask = m.get("no_ask", 0) or 0
        
        events[event_key]["teams"][team_code] = {
            "ticker": ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "volume": m.get("volume", 0),
            "open_interest": m.get("open_interest", 0),
            # Implied probability from mid price (Kalshi prices are in cents)
            "implied_prob": (yes_bid + yes_ask) / 200 if (yes_bid + yes_ask) > 0 else None,
        }
    
    return list(events.values())


async def get_kalshi_mlb_futures() -> list:
    """Get MLB futures (World Series winner, division winners, etc.)."""
    url = f"{KALSHI_BASE}/markets?limit=50&status=open&series_ticker=KXMLB"
    data = await _fetch(url)
    if isinstance(data, dict):
        return data.get("markets", [])
    return []


# ═══════════════════════════════════════════════════════════════
# POLYMARKET
# ═══════════════════════════════════════════════════════════════

async def get_polymarket_mlb() -> list:
    """
    Fetch any active MLB/baseball markets from Polymarket.
    Polymarket's sports coverage for MLB is still limited.
    """
    results = []
    
    # Try the sports games endpoint
    for tag in ["mlb", "baseball", "sports"]:
        url = f"https://gamma-api.polymarket.com/events?limit=20&active=true&tag={tag}"
        try:
            data = await _fetch(url)
            if isinstance(data, list):
                for event in data:
                    title = event.get("title", "")
                    if any(w in title.lower() for w in ["mlb", "baseball", "home run", "world series", "pitcher"]):
                        markets = []
                        for m in event.get("markets", []):
                            markets.append({
                                "question": m.get("question", ""),
                                "best_bid": m.get("bestBid", 0),
                                "best_ask": m.get("bestAsk", 0),
                                "volume": m.get("volumeNum", 0),
                                "liquidity": m.get("liquidityNum", 0),
                            })
                        results.append({
                            "title": title,
                            "slug": event.get("slug", ""),
                            "markets": markets,
                        })
        except Exception:
            continue
    
    return results


# ═══════════════════════════════════════════════════════════════
# AGGREGATION & EDGE FINDING
# ═══════════════════════════════════════════════════════════════

async def get_all_market_odds() -> dict:
    """
    Aggregate odds from all sources for MLB games.
    Returns a unified view combining sportsbook, Kalshi, and Polymarket.
    """
    tasks = [
        get_odds_api_mlb(),
        get_kalshi_mlb_games(),
        get_polymarket_mlb(),
    ]
    
    odds_api, kalshi, polymarket = await asyncio.gather(*tasks, return_exceptions=True)
    
    if isinstance(odds_api, Exception):
        odds_api = []
    if isinstance(kalshi, Exception):
        kalshi = []
    if isinstance(polymarket, Exception):
        polymarket = []
    
    return {
        "sportsbooks": odds_api if isinstance(odds_api, list) else [],
        "kalshi": kalshi if isinstance(kalshi, list) else [],
        "polymarket": polymarket if isinstance(polymarket, list) else [],
        "source_count": {
            "sportsbooks": len(odds_api) if isinstance(odds_api, list) else 0,
            "kalshi": len(kalshi) if isinstance(kalshi, list) else 0,
            "polymarket": len(polymarket) if isinstance(polymarket, list) else 0,
        }
    }


def match_odds_to_game(game: dict, sportsbook_odds: list) -> dict:
    """
    Match sportsbook odds to a specific game from our schedule.
    Uses team name fuzzy matching.
    """
    away = game.get("away_team_name", "").lower()
    home = game.get("home_team_name", "").lower()
    away_abbr = game.get("away_team", "").lower()
    home_abbr = game.get("home_team", "").lower()
    
    for odds_game in sportsbook_odds:
        oa = odds_game.get("away_team", "").lower()
        oh = odds_game.get("home_team", "").lower()
        
        if (away_abbr in oa or oa in away or any(w in oa for w in away.split()[-1:])) and \
           (home_abbr in oh or oh in home or any(w in oh for w in home.split()[-1:])):
            return odds_game
    
    return {}


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def prob_to_american(prob: float) -> str:
    """Convert probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        odds = int(-100 * prob / (1 - prob))
        return str(odds)
    else:
        odds = int(100 * (1 - prob) / prob)
        return f"+{odds}"


def compute_edge(model_prob: float, market_odds: int) -> dict:
    """
    Compute betting edge and Kelly criterion bet size.
    """
    implied = american_to_prob(market_odds)
    edge = model_prob - implied
    
    # Kelly criterion: f* = (bp - q) / b
    # b = decimal odds - 1, p = model_prob, q = 1 - model_prob
    if market_odds > 0:
        b = market_odds / 100
    else:
        b = 100 / abs(market_odds)
    
    q = 1 - model_prob
    kelly = (b * model_prob - q) / b if b > 0 else 0
    
    # Half-Kelly (more conservative)
    half_kelly = kelly / 2
    
    return {
        "edge_pct": round(edge * 100, 2),
        "implied_prob": round(implied, 4),
        "kelly_pct": round(max(0, kelly * 100), 2),
        "half_kelly_pct": round(max(0, half_kelly * 100), 2),
        "is_positive_ev": edge > 0,
        "market_odds": market_odds,
    }


def clear_market_cache():
    global _market_cache
    _market_cache = {}
