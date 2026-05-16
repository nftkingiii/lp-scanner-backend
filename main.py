"""
Polymarket LP Scanner - FastAPI Backend
With Supabase persistence for watchlist and positions
Run locally: uvicorn main:app --reload --port 8000
"""

import json as _json
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Supabase (optional — app works without it, falls back to client localStorage)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase connected")
    except Exception as e:
        print(f"Supabase init failed: {e}")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app = FastAPI(title="Polymarket LP Scanner API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_market(body: dict):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Anthropic API key not configured")
    
    market = body.get("market", {})
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": """You are a sharp Polymarket LP analyst specialising in maker reward farming.
The user places resting limit orders near the midpoint to earn maker rewards without getting filled.
Tone: direct, no fluff, no bullet lists. Format: 3 short paragraphs.
Cover: odds stability assessment, informed trader risk, pool share advantage, and a clear verdict.
End with BUY / PASS / WAIT on its own line.""",
                "messages": [{
                    "role": "user",
                    "content": body.get("prompt", "")
                }]
            }
        )
        resp.raise_for_status()
        data = resp.json()
        text = next((b["text"] for b in data.get("content", []) if b.get("type") == "text"), "")
        return {"analysis": text}

GAMMA_BASE = "https://gamma-api.polymarket.com"
HEADERS    = {"User-Agent": "PolymarketLPScanner/2.0", "Accept": "application/json"}
MAKER_REWARD_DAILY = 0.0005


# ─── helpers ──────────────────────────────────────────────────────────────────

def lp_score(yes, no, liquidity, end_date_iso):
    try:
        end_dt    = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        days_left = max(1, (end_dt - datetime.now(timezone.utc)).days)
    except Exception:
        days_left = 90
    balance    = 1 - abs(yes - 0.5) * 2
    time_score = 1.0 if days_left > 60 else (0.7 if days_left > 30 else 0.3)
    liq_score  = 1.0 if liquidity < 500 else (0.7 if liquidity < 2000 else 0.4)
    return round((balance * 0.5 + time_score * 0.3 + liq_score * 0.2) * 100)


def farm_score(yes, no, liquidity, volume, end_date_iso):
    try:
        end_dt    = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        days_left = max(0, (end_dt - datetime.now(timezone.utc)).days)
    except Exception:
        days_left = 90
    balance = 1 - abs(yes - 0.5) * 2
    if balance < 0.5:
        return 0
    capital    = 50
    pool_share = capital / (liquidity + capital) if liquidity > 0 else 0
    pool_score = min(1.0, pool_share * 8)
    vol_score  = (0.1 if volume < 100 else 0.4 if volume < 500 else
                  0.8 if volume < 5000 else 1.0 if volume < 50000 else
                  0.6 if volume < 250000 else 0.3)
    time_score = (0.0 if days_left < 7 else 0.2 if days_left < 14 else
                  0.5 if days_left < 30 else 0.8 if days_left < 60 else 1.0)
    raw = pool_score * 0.35 + 0.5 * 0.30 + vol_score * 0.15 + time_score * 0.20
    return round(raw * 100)


def parse_str_list(val):
    if isinstance(val, list): return val
    if isinstance(val, str):
        try: return _json.loads(val)
        except: pass
    return []


def get_category(m):
    events = m.get("events", [])
    slug = title = ""
    if isinstance(events, list) and events:
        slug  = events[0].get("slug", "").lower()
        title = events[0].get("title", "").lower()
    else:
        slug  = m.get("slug", "").lower()
        title = m.get("question", "").lower()
    combined = slug + " " + title
    if any(w in combined for w in ["crypto","bitcoin","btc","eth","defi","solana","nft","web3","token","blockchain"]): return "Crypto"
    if any(w in combined for w in ["election","president","vote","politic","congress","senate"]): return "Politics"
    if any(w in combined for w in ["nba","nfl","soccer","football","sport","tennis","ufc","championship","cup"]): return "Sports"
    if any(w in combined for w in ["ai","gpt","science","tech","model","nasa","space"]): return "Science"
    if any(w in combined for w in ["fed","economy","business","stock","rate","gdp","trade"]): return "Business"
    return "Other"


def is_binary(m):
    return len(parse_str_list(m.get("outcomes", []))) == 2


def parse_market(m):
    outcomes  = parse_str_list(m.get("outcomes", ["Yes","No"]))
    prices    = parse_str_list(m.get("outcomePrices", [0.5, 0.5]))
    try:
        yes = float(prices[0]) if prices else 0.5
        no  = float(prices[1]) if len(prices) > 1 else round(1 - yes, 4)
    except: yes, no = 0.5, 0.5
    volume    = float(m.get("volumeNum") or m.get("volume") or 0)
    liquidity = float(m.get("liquidityNum") or m.get("liquidity") or 0)
    end_date  = m.get("endDate") or m.get("endDateIso") or ""
    return {
        "id":         m.get("id", ""),
        "question":   m.get("question", ""),
        "slug":       m.get("slug", ""),
        "yes":        round(yes, 4),
        "no":         round(no, 4),
        "volume":     round(volume, 2),
        "liquidity":  round(liquidity, 2),
        "endDate":    end_date,
        "category":   get_category(m),
        "active":     m.get("active", True),
        "closed":     m.get("closed", False),
        "score":      lp_score(yes, no, liquidity, end_date),
        "farmScore":  farm_score(yes, no, liquidity, volume, end_date),
        "outcomes":   outcomes,
    }


# ─── market routes ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "supabase": supabase is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/markets")
async def get_markets(
    limit:       int           = Query(100, ge=1, le=200),
    offset:      int           = Query(0,   ge=0),
    active_only: bool          = Query(True),
    sort_by:     str           = Query("score"),
):
    params = {"limit": min(limit * 3, 300), "offset": offset,
              "active": "true" if active_only else "false"}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(f"{GAMMA_BASE}/markets", params=params, headers=HEADERS)
            resp.raise_for_status()
            raw = resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Gamma API error")
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    binary = [m for m in raw if is_binary(m)]
    parsed = [parse_market(m) for m in binary]

    if sort_by == "score":      parsed.sort(key=lambda m: m["score"],     reverse=True)
    elif sort_by == "farm":     parsed.sort(key=lambda m: m["farmScore"], reverse=True)
    elif sort_by == "volume":   parsed.sort(key=lambda m: m["volume"],    reverse=True)
    elif sort_by == "liquidity":parsed.sort(key=lambda m: m["liquidity"], reverse=True)

    return {"total": len(parsed), "markets": parsed[:limit],
            "fetched_at": datetime.now(timezone.utc).isoformat()}


# ─── Supabase persistence routes ───────────────────────────────────────────────

def require_supabase():
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")

@app.get("/user/{user_id}/watchlist")
async def get_watchlist(user_id: str):
    require_supabase()
    try:
        res = supabase.table("watchlists").select("market_ids").eq("user_id", user_id).execute()
        if res.data:
            return {"market_ids": res.data[0].get("market_ids", [])}
        return {"market_ids": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/{user_id}/watchlist")
async def save_watchlist(user_id: str, body: dict):
    require_supabase()
    try:
        supabase.table("watchlists").upsert(
            {"user_id": user_id, "market_ids": body.get("market_ids", []),
             "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="user_id"
        ).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/positions")
async def get_positions(user_id: str):
    require_supabase()
    try:
        res = supabase.table("positions").select("*").eq("user_id", user_id).execute()
        return {"positions": res.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/{user_id}/positions")
async def save_positions(user_id: str, body: dict):
    require_supabase()
    try:
        supabase.table("positions").upsert(
            {"user_id": user_id, "data": body.get("positions", []),
             "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="user_id"
        ).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
