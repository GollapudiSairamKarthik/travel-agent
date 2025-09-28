#!/usr/bin/env python3
"""
travel_agent.py

Single-file Travel Planner (LangGraph supervisor + tools).
Outputs weather, POIs, and an itinerary table with columns:
Day | Morning | Afternoon | Evening | Notes

Environment (.env) expected keys:
  GROQ_API_KEY=...             # Groq / OpenAI-compatible API key
  OPENAI_API_BASE=https://api.groq.com/openai/v1
  LLM_MODEL=llama-3.3-70b-versatile
  OPENTRIPMAP_API_KEY=...      # optional, for richer POIs
"""
from __future__ import annotations
import os
import time
import requests
import argparse
import datetime
from dateutil.parser import parse as parse_date
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus

# LangGraph & LangChain wrapper
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Load env
load_dotenv()

OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")


# ----------------------
# Geocoding helpers (now returns category + links)
# ----------------------
def _make_links_for_coords(lat: float, lon: float, name: Optional[str] = None) -> Dict[str, str]:
    """Return helpful links for a lat/lon: OpenStreetMap, Google Maps, Wikipedia search."""
    osm = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=12/{lat}/{lon}"
    gmap = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    wiki_q = quote_plus(name or f"{lat},{lon}")
    wiki = f"https://en.wikipedia.org/w/index.php?search={wiki_q}"
    return {"openstreetmap": osm, "google_maps": gmap, "wikipedia_search": wiki}


def geocode_opentripmap(place: str) -> Optional[Dict[str, Any]]:
    """Try OpenTripMap geoname endpoint (requires OPENTRIPMAP_API_KEY)."""
    if not OPENTRIPMAP_API_KEY:
        return None
    try:
        url = "https://api.opentripmap.com/0.1/en/places/geoname"
        r = requests.get(url, params={"name": place, "apikey": OPENTRIPMAP_API_KEY}, timeout=8)
        r.raise_for_status()
        d = r.json()
        if "lat" in d and "lon" in d:
            lat = float(d["lat"]); lon = float(d["lon"])
            name = d.get("name", place)
            return {
                "lat": lat,
                "lon": lon,
                "name": name,
                "category": d.get("type") or "place",
                "links": _make_links_for_coords(lat, lon, name),
            }
    except Exception:
        return None


def geocode_nominatim_once(place: str, user_agent: str) -> Optional[Dict[str, Any]]:
    """
    Single Nominatim attempt. Returns dict with lat, lon, display_name, class/type, category and links.
    Tries to prefer results containing the query token to avoid noise like 'X Way' when user meant 'X'.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": user_agent}
        r = requests.get(
            url,
            params={"q": place, "format": "json", "limit": 3, "addressdetails": 0, "extratags": 1},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
        if not items:
            return None

        qtoken = place.split(",")[0].strip().lower()
        chosen = None
        # prefer a result whose display_name contains the queried token exactly
        for it in items:
            display = (it.get("display_name") or "").lower()
            if qtoken and qtoken in display:
                chosen = it
                break
        if not chosen:
            chosen = items[0]

        lat = float(chosen["lat"])
        lon = float(chosen["lon"])
        display_name = chosen.get("display_name", place)
        fclass = chosen.get("class", "") or ""
        ftype = chosen.get("type", "") or ""

        # derive a simple category useful for beaches/mountains/hills
        class_type = f"{fclass}:{ftype}".lower()
        category = "other"
        if "beach" in class_type or "beach" in ftype:
            category = "beach"
        elif any(x in class_type for x in ("peak", "mountain", "hill", "ridge", "valley", "mountain_range")):
            category = "mountain/hill"
        elif "city" in class_type or "town" in class_type or "village" in class_type or "county" in class_type:
            category = "city"
        elif "water" in class_type or "river" in class_type:
            category = "water"

        return {
            "lat": lat,
            "lon": lon,
            "name": display_name,
            "class": fclass,
            "type": ftype,
            "category": category,
            "links": _make_links_for_coords(lat, lon, display_name),
        }
    except Exception:
        return None


def geocode_nominatim(place: str) -> Optional[Dict[str, Any]]:
    """Robust Nominatim geocode: try with a couple of user-agents and a fallback of appending 'India'."""
    attempts = ["LangGraph-Travel-Agent/1.0", "Mozilla/5.0 (compatible; LangGraphAgent/1.0)"]
    for ua in attempts:
        res = geocode_nominatim_once(place, ua)
        if res:
            return res
    # if short and likely Indian locality, try adding country hint
    if "," not in place and len(place.split()) <= 3:
        res = geocode_nominatim_once(f"{place}, India", attempts[0])
        if res:
            return res
    return None


def geocode_city(place: str) -> Optional[Dict[str, Any]]:
    """
    Try OpenTripMap first, then Nominatim.
    If Nominatim returns a name that doesn't contain the query token, try disambiguating with country hint.
    Returns: {lat, lon, name, category, links, ...}
    """
    place = (place or "").strip()
    if not place:
        return None

    g = geocode_opentripmap(place)
    if g:
        # opentripmap doesn't always include category/links; add if missing
        if "links" not in g:
            g["links"] = _make_links_for_coords(g["lat"], g["lon"], g.get("name"))
        if "category" not in g:
            g["category"] = "place"
        return g

    g = geocode_nominatim(place)
    if not g:
        return None

    token = place.split(",")[0].strip().lower()
    display_lower = (g.get("name") or "").lower()
    if token and token not in display_lower and "india" not in display_lower:
        alt = geocode_nominatim(f"{place}, India")
        if alt:
            return alt
    return g


# ----------------------
# POI & weather fetchers (unchanged behavior)
# ----------------------
def fetch_pois_opentripmap(lat: float, lon: float, radius: int = 2000, limit: int = 8) -> List[Dict[str, Any]]:
    if not OPENTRIPMAP_API_KEY:
        return []
    try:
        url = "https://api.opentripmap.com/0.1/en/places/radius"
        params = {"apikey": OPENTRIPMAP_API_KEY, "radius": radius, "lon": lon, "lat": lat, "limit": limit, "rate": 3}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        features = r.json().get("features", [])
        results = []
        for p in features:
            props = p.get("properties", {})
            name = props.get("name") or props.get("kinds", "unknown")
            xid = props.get("xid")
            details = {}
            if xid:
                try:
                    dr = requests.get(f"https://api.opentripmap.com/0.1/en/places/xid/{xid}",
                                      params={"apikey": OPENTRIPMAP_API_KEY}, timeout=6)
                    if dr.status_code == 200:
                        details = dr.json()
                except Exception:
                    details = {}
                time.sleep(0.03)
            results.append({"name": name, "dist": props.get("dist"), "kinds": props.get("kinds"), "details": details})
        return results
    except Exception:
        return []


def fetch_pois_overpass(lat: float, lon: float, radius: int = 2000, limit: int = 12) -> List[Dict[str, Any]]:
    try:
        query = f"""
        [out:json][timeout:15];
        (
          node(around:{radius},{lat},{lon})["tourism"];
          way(around:{radius},{lat},{lon})["tourism"];
          node(around:{radius},{lat},{lon})["historic"];
          way(around:{radius},{lat},{lon})["historic"];
          node(around:{radius},{lat},{lon})["leisure"];
          way(around:{radius},{lat},{lon})["leisure"];
          node(around:{radius},{lat},{lon})["amenity"~"museum|theatre|gallery|marketplace|park"];
        );
        out center {limit};
        """
        r = requests.post("https://overpass-api.de/api/interpreter", data=query.encode("utf-8"), timeout=20)
        r.raise_for_status()
        el = r.json().get("elements", [])
        results = []
        seen = set()
        for e in el:
            tags = e.get("tags", {})
            name = tags.get("name")
            if not name:
                continue
            kinds_list = []
            for k in ("tourism", "historic", "leisure", "amenity"):
                if k in tags:
                    kinds_list.append(tags.get(k))
            kinds = ",".join([str(x) for x in kinds_list if x])
            key = name.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            results.append({"name": name, "kinds": kinds or "unknown", "dist": None, "tags": tags})
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def fetch_pois(lat: float, lon: float, radius: int = 2000, limit: int = 8) -> List[Dict[str, Any]]:
    pois = fetch_pois_opentripmap(lat, lon, radius=radius, limit=limit)
    if pois:
        seen = set()
        filtered = []
        for p in pois:
            name = (p.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(p)
        return filtered[:limit]
    pois2 = fetch_pois_overpass(lat, lon, radius=radius, limit=max(limit, 12))
    if not pois2:
        return []
    priority = ("attraction", "museum", "memorial", "monument", "historic", "park", "marketplace", "gallery")

    def score(p: Dict[str, Any]) -> int:
        kinds = (p.get("kinds") or "").lower()
        for i, k in enumerate(priority):
            if k in kinds:
                return -(len(priority) - i)
        return 0

    pois2.sort(key=score)
    return pois2[:limit]


def fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, Any]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                  "start_date": start_date, "end_date": end_date, "timezone": "auto"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("daily", {})
    except Exception:
        return {}


# ----------------------
# Tools
# ----------------------
def weather_agent(city: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    g = geocode_city(city)
    if g is None:
        return f"ERROR: Could not geocode '{city}'."
    lat, lon = g["lat"], g["lon"]
    name = g.get("name", city)
    category = g.get("category", "")
    links = g.get("links", {})
    if not start_date:
        start_date = datetime.date.today().isoformat()
    if not end_date:
        end_date = start_date
    try:
        sd = parse_date(start_date).date().isoformat()
        ed = parse_date(end_date).date().isoformat()
    except Exception:
        return "ERROR: Dates must be YYYY-MM-DD."
    weather = fetch_weather(lat, lon, sd, ed)
    if not weather:
        header = f"WARNING: No weather data for {name} between {sd} and {ed}."
        if links:
            header += f" Links: {links.get('openstreetmap')} | {links.get('google_maps')}"
        return header
    lines = [f"Weather for {name} ({sd} to {ed}):"]
    times = weather.get("time", [])
    tmax = weather.get("temperature_2m_max", [])
    tmin = weather.get("temperature_2m_min", [])
    prec = weather.get("precipitation_sum", [])
    for i, d in enumerate(times):
        mx = tmax[i] if i < len(tmax) else "N/A"
        mn = tmin[i] if i < len(tmin) else "N/A"
        pr = prec[i] if i < len(prec) else "N/A"
        lines.append(f"- {d}: max {mx}°C, min {mn}°C, precipitation {pr} mm")
    # Append geocode links (useful for beaches/mountains)
    if links:
        lines.append("")
        lines.append(f"Location links: {links.get('openstreetmap')} | {links.get('google_maps')} | {links.get('wikipedia_search')}")
    return "\n".join(lines)


def poi_agent(city: str, radius: int = 2000, limit: int = 8) -> str:
    g = geocode_city(city)
    if g is None:
        return f"ERROR: Could not geocode '{city}'."
    pois = fetch_pois(g["lat"], g["lon"], radius=radius, limit=limit)
    if not pois:
        return f"WARNING: No POIs found for {g['name']}."
    lines = [f"Top {len(pois)} POIs near {g['name']}:"] + [
        f"{i+1}. {p.get('name')} — kinds: {p.get('kinds','')} — dist: {p.get('dist','N/A')}" for i, p in enumerate(pois)
    ]
    # Add location links if available (helpful for beaches/mountains)
    links = g.get("links", {})
    if links:
        lines.append("")
        lines.append(f"Location links: {links.get('openstreetmap')} | {links.get('google_maps')} | {links.get('wikipedia_search')}")
    return "\n".join(lines)


def itinerary_agent(city: str, start_date: str, end_date: str, daily_limit: int = 3) -> str:
    g = geocode_city(city)
    if g is None:
        return f"ERROR: Could not geocode '{city}'."
    try:
        sd = parse_date(start_date).date()
        ed = parse_date(end_date).date()
    except Exception:
        return "ERROR: Dates must be YYYY-MM-DD."
    if ed < sd:
        return "ERROR: end_date must be same or after start_date."
    num_days = (ed - sd).days + 1

    pool_pois = fetch_pois(g["lat"], g["lon"], radius=3500, limit=max(20, daily_limit * num_days * 2))
    poi_lines = "\n".join([f"- {p['name']} ({p.get('kinds','')})" for p in pool_pois]) if pool_pois else "No POIs found."

    weather = fetch_weather(g["lat"], g["lon"], sd.isoformat(), ed.isoformat())
    precip = weather.get("precipitation_sum", [])
    tmax = weather.get("temperature_2m_max", [])
    tmin = weather.get("temperature_2m_min", [])

    indoor_kinds = {"museum", "theatre", "gallery", "library", "cinema"}
    pool = []
    for p in pool_pois:
        kinds = (p.get("kinds") or "").lower()
        is_indoor = any(k in kinds for k in indoor_kinds)
        pool.append({"name": p.get("name"), "is_indoor": is_indoor, "kinds": kinds})

    assigned = set()
    rows: List[Dict[str, Any]] = []
    idx = 0

    def weather_note_for_day(i: int) -> str:
        note_parts = []
        if i < len(precip):
            try:
                pr = float(precip[i])
            except Exception:
                pr = 0.0
            if pr >= 10.0:
                note_parts.append("Heavy rain expected — favor indoor activities")
            elif pr >= 2.0:
                note_parts.append("Chance of rain — have indoor alternatives")
            elif pr > 0:
                note_parts.append("Light showers possible")
            else:
                note_parts.append("Good weather for walking")
        if i < len(tmax) and i < len(tmin):
            try:
                mx = float(tmax[i]); mn = float(tmin[i])
                if mx >= 30:
                    note_parts.append("Hot during day")
                elif mx <= 15:
                    note_parts.append("Cool day — bring a jacket")
            except Exception:
                pass
        return "; ".join(note_parts) if note_parts else "No specific weather notes"

    for day_index in range(num_days):
        date = (sd + datetime.timedelta(days=day_index)).isoformat()
        rainy = False
        if day_index < len(precip):
            try:
                rainy = float(precip[day_index]) >= 2.0
            except Exception:
                rainy = False

        day_selected: List[str] = []
        attempts = 0
        while len(day_selected) < daily_limit and attempts < max(1, len(pool)) * 2:
            p = pool[idx % max(1, len(pool))] if pool else None
            idx += 1; attempts += 1
            if p is None:
                break
            if p["name"] in assigned:
                continue
            if rainy and not p["is_indoor"]:
                continue
            day_selected.append(p["name"])
            assigned.add(p["name"])
        if len(day_selected) < daily_limit:
            for p in pool:
                if p["name"] in assigned:
                    continue
                day_selected.append(p["name"])
                assigned.add(p["name"])
                if len(day_selected) >= daily_limit:
                    break

        morning = day_selected[0] if len(day_selected) > 0 else "Free / explore locally"
        afternoon = day_selected[1] if len(day_selected) > 1 else "Free / explore locally"
        evening = day_selected[2] if len(day_selected) > 2 else "Dinner / relax"
        notes = weather_note_for_day(day_index)
        rows.append({"date": date, "morning": morning, "afternoon": afternoon, "evening": evening, "notes": notes})

    md: List[str] = []
    md.append(f"# Itinerary for {g.get('name')}")
    md.append(f"Dates: {sd.isoformat()} to {ed.isoformat()}")
    md.append("")
    # include category and helpful links near header (useful for beaches/hills/mountains)
    cat = g.get("category")
    if cat:
        md.append(f"**Place type:** {cat}")
    links = g.get("links", {})
    if links:
        md.append(f"**Links:** {links.get('openstreetmap')} | {links.get('google_maps')} | {links.get('wikipedia_search')}")
    md.append("")
    md.append("## Travel Itinerary")
    md.append("")
    md.append("| Day | Morning | Afternoon | Evening | Notes |")
    md.append("|---:|---|---|---|---|")
    for i, r in enumerate(rows, start=1):
        md.append(f"| {i} | {r['morning']} | {r['afternoon']} | {r['evening']} | {r['notes']} |")
    md.append("")
    md.append("## POIs considered")
    md.append(poi_lines)
    return "\n".join(md)


# ----------------------
# Adapter wrappers (docstrings required so LangChain converts them to structured tools)
# ----------------------
def weather_tool(city: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Return weather summary for a city and date range. Args: city (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)."""
    return weather_agent(str(city), str(start_date) if start_date else None, str(end_date) if end_date else None)


def poi_tool(city: str, radius: Optional[object] = 2000, limit: Optional[object] = 8) -> str:
    """Return POIs near a city. Args: city (str), radius (int meters), limit (int)."""
    try:
        radius_i = int(radius) if radius is not None and str(radius) != "" else 2000
    except Exception:
        radius_i = 2000
    try:
        limit_i = int(limit) if limit is not None and str(limit) != "" else 8
    except Exception:
        limit_i = 8
    return poi_agent(str(city), radius=radius_i, limit=limit_i)


def itinerary_tool(city: str, start_date: str, end_date: str, daily_limit: Optional[object] = 3) -> str:
    """Return a markdown itinerary. Args: city (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD), daily_limit (int)."""
    try:
        daily_limit_i = int(daily_limit) if daily_limit is not None and str(daily_limit) != "" else 3
    except Exception:
        daily_limit_i = 3
    return itinerary_agent(str(city), str(start_date), str(end_date), daily_limit=daily_limit_i)


# ----------------------
# Build supervisor agent
# ----------------------
def build_agent():
    """Create and return the LangGraph REACT-style supervisor agent."""
    tools = [weather_tool, poi_tool, itinerary_tool]
    llm = ChatOpenAI(model=LLM_MODEL, api_key=GROQ_API_KEY, base_url=OPENAI_API_BASE, temperature=0.2)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a travel-planning supervisor. Tools available:\n"
            "- weather_tool(city, start_date, end_date) -> returns weather summary\n"
            "- poi_tool(city, radius, limit) -> returns POIs list\n"
            "- itinerary_tool(city, start_date, end_date, daily_limit) -> returns itinerary\n\n"
            "When calling tools, send city names and dates as strings and numeric values as integers.\n"
            "Produce a final reply containing weather, POIs, and a day-by-day itinerary table with a Notes column."
        ),
    )
    return agent


# ----------------------
# CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=False)
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    args = parser.parse_args()

    if not args.city:
        city = input("Enter place: ").strip()
    else:
        city = args.city
    if not city:
        print("No place provided. Exiting.")
        return

    start = args.start or input("Start date (YYYY-MM-DD) [default today]: ").strip() or datetime.date.today().isoformat()
    end = args.end or input("End date (YYYY-MM-DD) [default same as start]: ").strip() or start

    user_query = f"I want a trip plan to {city} from {start} to {end}. Provide weather, POIs, and an itinerary table with Notes."
    print("Invoking supervisor agent...")

    final = ""
    try:
        agent = build_agent()
        state = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

        # Extract messages robustly (dicts, strings, or langchain message objects)
        msgs = []
        if isinstance(state, dict):
            msgs = state.get("messages") or state.get("output") or state.get("result") or []
        else:
            try:
                msgs = state.get("messages") or []
            except Exception:
                msgs = []

        extracted: List[str] = []
        for m in msgs:
            if isinstance(m, dict) and m.get("content"):
                extracted.append(m.get("content"))
            elif isinstance(m, str):
                extracted.append(m)
            elif isinstance(m, (AIMessage, HumanMessage, ToolMessage)) and getattr(m, "content", None):
                extracted.append(m.content)

        final = "\n\n".join([s for s in extracted if s and s.strip()]).strip()

        # fallback: check other fields in state dict
        if not final and isinstance(state, dict):
            alt_texts = []
            for k in ("assistant", "response", "final_output", "output_text", "result"):
                v = state.get(k)
                if isinstance(v, str) and v.strip():
                    alt_texts.append(v.strip())
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            alt_texts.append(item.strip())
                        elif isinstance(item, dict) and item.get("content"):
                            alt_texts.append(item.get("content"))
            if alt_texts:
                final = "\n\n".join(alt_texts).strip()

        if not final:
            raise RuntimeError("Supervisor returned no usable messages.")
    except Exception:
        import traceback
        print("\n--- SUPERVISOR EXCEPTION ---")
        traceback.print_exc()
        print("--- END ---\n")
        final = "NOTE: Supervisor failed — using direct tools.\n\n"
        final += weather_agent(city, start, end) + "\n\n"
        final += poi_agent(city, radius=2500, limit=10) + "\n\n"
        final += itinerary_agent(city, start, end)

    print("\n=== AGENT OUTPUT ===\n")
    print(final)
    print("\n=== END ===\n")


if __name__ == "__main__":
    main()
