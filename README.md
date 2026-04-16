# 💬 WhatsApp Intelligent Analyzer

An interactive web dashboard that processes WhatsApp chat exports (`.txt`) and generates rich visual analytics powered by AI.

## Features

- **5 Analysis Tabs**: Summary, Members, Spelling, Gossip Bursts, Interaction Dynamics
- **AI Insights**: Vibe analysis, topic mapping, and personality profiles via Claude
- **Activity Heatmap**: Hour × Day visualization of chat activity
- **Interaction Network**: D3.js force-directed response graph
- **Spelling Analysis**: Bilingual (ES + EN) error detection with word cloud
- **Burst Detection**: Identifies peak conversation moments
- **Ghosting & Triple-Texting**: Behavioral dynamics metrics
- Supports iPhone and Android WhatsApp export formats (English & Spanish)

## Deploy to Railway (1-click)

1. **Fork this repository** on GitHub (click the Fork button at the top-right)

2. **Create a Railway account** at [railway.app](https://railway.app) if you don't have one

3. **Create a new project**:
   - Click **"New Project"**
   - Select **"Deploy from GitHub repo"**
   - Authorize Railway and select your forked repository

4. **Railway auto-detects** the `Procfile` and configures the deployment automatically

5. **Get your public URL**:
   - Go to your project settings → Domains
   - Click **"Generate Domain"**
   - Your app is now live at `https://your-app.up.railway.app`

## How to Use

1. Open the Railway URL in your browser
2. Export a WhatsApp chat: **Chat → ⋮ menu → More → Export chat → Without Media**
3. Get a [Claude API Key](https://console.anthropic.com/) (optional, needed for AI insights)
4. Upload the `.txt` file and paste your API key
5. Click **"Analizar"** and wait for your dashboard

## Privacy

- Chat files are processed in memory and immediately discarded
- API keys are never stored or logged
- No data persists between requests

## Tech Stack

- **Backend**: Python, Flask, pandas, pyspellchecker, Anthropic SDK
- **Frontend**: Vanilla JS, Chart.js, D3.js, wordcloud2.js
- **Deployment**: Railway, Gunicorn
