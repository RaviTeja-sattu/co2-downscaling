# CO2 Downscaling Dashboard (PAVAN)

India CO₂ satellite data visualisation and downscaling app built with Flask + Plotly + Gemini AI.

## Deploy to Render (free, recommended)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set these values:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Environment Variable:** `GEMINI_API_KEY` = your Gemini API key

Done. Render will auto-deploy on every git push.

> ⚠️ Render's free tier sleeps after 15 min of inactivity (cold start ~30s).
> Upgrade to Starter ($7/mo) for always-on.

## Deploy to Railway (alternative)

1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add env var: `GEMINI_API_KEY`
4. Railway auto-detects the Procfile

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python app.py
```
