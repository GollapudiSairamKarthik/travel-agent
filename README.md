🌍 Travel Agent (AI Trip Planner)

An AI-powered travel planning assistant that generates personalized itineraries including weather forecasts, points of interest (POIs), and day-by-day schedules.
Built using LangGraph, Groq LLM API, and OpenTripMap / Open-Meteo APIs.

🚀 Features

🗺️ Geocoding & Location Info: Converts city names into latitude/longitude with helpful links (Google Maps, OSM, Wikipedia).

🌦️ Weather Forecasts: Uses Open-Meteo API to fetch accurate daily forecasts.

🏛️ Points of Interest (POIs): Fetches attractions, museums, parks, beaches, mountains, etc. using OpenTripMap API and Overpass (OSM).

📅 Itinerary Generator: Automatically builds a markdown travel itinerary with:

Morning, Afternoon, Evening activities

Weather-aware suggestions (indoor if rainy, outdoor if sunny)

Beach-specific activities (sunrise, sunset walks)

Mountain/Hill-specific activities (hiking, trekking recommendations)

🤖 LangGraph Supervisor Agent: Orchestrates multiple tools (weather, POIs, itinerary) with an LLM (LLaMA-3 via Groq).

📌 CLI (Command Line Interface): Users can provide city & dates directly in the terminal.

🏗️ Tech Stack

Python 3.9+

LangGraph + LangChain → agent orchestration

Groq LLM API (LLaMA-3.3-70B) → reasoning + itinerary generation

OpenTripMap API → points of interest

Open-Meteo API → weather forecasts

Overpass API (OSM) → backup POIs

dateutil, requests, dotenv → utilities
