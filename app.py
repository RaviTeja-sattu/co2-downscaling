import os, io, warnings, traceback, json, re
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from flask import Flask, render_template, jsonify, send_file, request
import google.generativeai as genai
warnings.filterwarnings('ignore')

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── AI CONFIG ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
AI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

IMD_SEASONS = {
    'Winter':      [1, 2],
    'Premonsoon':  [3, 4, 5],
    'Monsoon':     [6, 7, 8, 9],
    'Postmonsoon': [10, 11, 12],
}
SEASON_ORDER  = ['Winter', 'Premonsoon', 'Monsoon', 'Postmonsoon']
SEASON_COLORS = {
    'Winter':      '#1e88e5',
    'Premonsoon':  '#43a047',
    'Monsoon':     '#8e24aa',
    'Postmonsoon': '#fb8c00',
}
MONTH_LABELS = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

ENV_VARS     = ['blh', 'temperature', 'pressure', 'specific_humidity']
FEATURE_COLS = ['latitude','longitude','ct_co2','blh','temperature',
                'pressure','specific_humidity','lat_norm','lon_norm',
                'lat_lon_interact','dist_from_center']

DS_COLORS = {
    'CarbonTracker': '#4361ee',
    'GOSAT':         '#16a34a',
    'OCO-2':         '#9333ea',
    'OCO-3':         '#ea580c',
}

CO2_BG = 280.0  # pre-industrial baseline (ppm)

# ── LAYOUT / GEO DEFAULTS ─────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#f8faff',
    font=dict(family='Inter, system-ui, sans-serif', size=12, color='#1e2338'),
    margin=dict(l=12, r=12, t=52, b=12),
    legend=dict(
        bgcolor='rgba(255,255,255,0.97)',
        bordercolor='#e2e6f0',
        borderwidth=1,
        font=dict(size=11),
        itemsizing='constant',
    ),
)

GEO_STYLE = dict(
    visible=True,
    resolution=50,
    showcountries=True,  countrycolor='#94a3b8', countrywidth=1.2,
    showcoastlines=True, coastlinecolor='#6b8fa8', coastlinewidth=1.0,
    showland=True,       landcolor='#f0ece4',
    showocean=True,      oceancolor='#d0e8f5',
    showlakes=True,      lakecolor='#d8ecf5',
    showrivers=False,
    lataxis_range=[6, 38],
    lonaxis_range=[66, 100],
    projection_type='natural earth',
    showframe=False,
    bgcolor='#e8f2fa',
)

# ── AI STYLE HELPER ───────────────────────────────────────────────────────────
def apply_style(fig, title='', height=420, showlegend=True):
    fig.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8faff',
        font=dict(family='Inter, system-ui, sans-serif', size=12, color='#1e2338'),
        margin=dict(l=12, r=12, t=52, b=12),
        height=height,
        showlegend=showlegend,
        legend=dict(
            bgcolor='rgba(255,255,255,0.97)',
            bordercolor='#e2e6f0',
            borderwidth=1,
            font=dict(size=11),
        ),
        title=dict(
            text=title,
            font=dict(size=13, color='#1e2338'),
            x=0.02, xanchor='left',
        ),
    )
    fig.update_xaxes(gridcolor='#e2e6f0', zeroline=False)
    fig.update_yaxes(gridcolor='#e2e6f0', zeroline=False)
    return fig

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_season(month):
    for s, months in IMD_SEASONS.items():
        if month in months:
            return s
    return None

def tight_range(series, nsigma=2.5, pad=0.5):
    m, sd = float(series.median()), float(series.std())
    return m - nsigma*sd - pad, m + nsigma*sd + pad

def hex_to_rgba(hex_color, alpha=0.18):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def choropleth_box_map(df, value_col, title,
                       colorscale='Plasma', zmin=None, zmax=None,
                       zmid=None, height=460):
    if zmin is None:
        zmin, zmax = tight_range(df[value_col])
    z = df[value_col].astype(float).values
    cbar = dict(
        thickness=14, len=0.7,
        title=dict(text='ppm', side='right', font=dict(size=11)),
        tickfont=dict(size=10, family='JetBrains Mono, monospace'),
        outlinewidth=0,
    )
    marker_kw = dict(
        size=16, color=z, colorscale=colorscale,
        cmin=zmin, cmax=zmax, symbol='square',
        colorbar=cbar,
        line=dict(width=0.5, color='rgba(255,255,255,0.6)'),
        opacity=0.93,
    )
    if zmid is not None:
        marker_kw['cmid'] = zmid
    fig = go.Figure(go.Scattergeo(
        lat=df['latitude'], lon=df['longitude'],
        mode='markers', marker=marker_kw,
        text=[f'{v:.3f} ppm' for v in z],
        hovertemplate='<b>%{text}</b><br>Lat %{lat:.2f}°N  Lon %{lon:.2f}°E<extra></extra>',
    ))
    fig.update_geos(**GEO_STYLE)
    fig.update_layout(
        paper_bgcolor='#ffffff',
        font=dict(family='Inter, system-ui, sans-serif', size=12, color='#1e2338'),
        margin=dict(l=0, r=0, t=46, b=0),
        height=height,
        title=dict(text=title, font=dict(size=13, color='#1e2338'),
                   x=0.02, xanchor='left'),
    )
    return fig

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    d = {}
    ct = pd.read_csv(os.path.join(DATA_DIR, "ct_1330_india.csv"))
    if 'geometry' in ct.columns:
        ct = ct.drop(columns=['geometry'])
    ct['time']   = pd.to_datetime(ct['time'], dayfirst=True, errors='coerce')
    ct['year']   = ct['time'].dt.year
    ct['month']  = ct['time'].dt.month
    ct['season'] = ct['month'].apply(get_season)
    d['ct'] = ct

    gosat = pd.read_csv(os.path.join(DATA_DIR, "india_xco2_gosat.csv"))
    gosat['time']   = pd.to_datetime(gosat['date'], errors='coerce')
    gosat['year']   = gosat['time'].dt.year
    gosat['month']  = gosat['time'].dt.month
    gosat['season'] = gosat['month'].apply(get_season)
    gosat = gosat.rename(columns={'xco2_bias_corrected': 'co2'})
    d['gosat'] = gosat[['latitude','longitude','co2','month','year','season','time']].dropna()

    oco2 = pd.read_csv(os.path.join(DATA_DIR, "india_xco2_oco2.csv"), dtype={'time': str})
    oco2['time']   = pd.to_datetime(oco2['time'], errors='coerce')
    oco2['year']   = oco2['time'].dt.year
    oco2['month']  = oco2['time'].dt.month
    oco2['season'] = oco2['month'].apply(get_season)
    oco2 = oco2.rename(columns={'xco2': 'co2'})
    oco2 = oco2[oco2['qf'] == 0]
    d['oco2'] = oco2[['latitude','longitude','co2','month','year','season','time']].dropna()

    oco3 = pd.read_csv(os.path.join(DATA_DIR, "india_xco2_oco3.csv"), dtype={'time': str})
    oco3['time']   = pd.to_datetime(oco3['time'], errors='coerce')
    oco3['year']   = oco3['time'].dt.year
    oco3['month']  = oco3['time'].dt.month
    oco3['season'] = oco3['month'].apply(get_season)
    oco3 = oco3.rename(columns={'xco2': 'co2'})
    oco3 = oco3[oco3['qf'] == 0]
    d['oco3'] = oco3[['latitude','longitude','co2','month','year','season','time']].dropna()

    return d

def load_models():
    models = {}
    for s in IMD_SEASONS:
        p = os.path.join(MODEL_DIR, f"rf_{s}.pkl")
        if os.path.exists(p):
            models[s] = joblib.load(p)
    return models

# ── DOWNSCALING ───────────────────────────────────────────────────────────────
def interpolate_env(mlat, mlon, ct_df):
    coords = ct_df[['latitude','longitude']].values
    result = {}
    for v in ENV_VARS:
        if v not in ct_df.columns:
            result[v] = np.nan; continue
        val = griddata(coords, ct_df[v].values, (mlat, mlon),
                       method='linear',
                       fill_value=float(np.nanmean(ct_df[v].values)))
        result[v] = float(val) if isinstance(val, np.ndarray) else float(val)
    return result

def build_miniboxes(ct_df, season_name):
    LAT_H, LON_H = 0.5, 0.75
    offsets = [(-LAT_H,-LON_H),(-LAT_H,LON_H),(LAT_H,-LON_H),(LAT_H,LON_H)]
    rows = []
    for idx, row in ct_df.iterrows():
        for mb_id, (dlat, dlon) in enumerate(offsets):
            mlat = row['latitude']  + dlat
            mlon = row['longitude'] + dlon
            env  = interpolate_env(mlat, mlon, ct_df)
            rows.append({'original_ct_idx': idx, 'minibox_id': mb_id,
                         'latitude': mlat, 'longitude': mlon,
                         'lat_min': mlat-LAT_H, 'lat_max': mlat+LAT_H,
                         'lon_min': mlon-LON_H, 'lon_max': mlon+LON_H,
                         'ct_co2': row['co2'], 'season': season_name, **env})
    df = pd.DataFrame(rows)
    clat, clon = df['latitude'].mean(), df['longitude'].mean()
    df['dist_from_center'] = np.sqrt((df['latitude']-clat)**2 + (df['longitude']-clon)**2)
    df['lat_norm'] = (df['latitude']-df['latitude'].min()) / (df['latitude'].max()-df['latitude'].min()+1e-9)
    df['lon_norm'] = (df['longitude']-df['longitude'].min()) / (df['longitude'].max()-df['longitude'].min()+1e-9)
    df['lat_lon_interact'] = df['lat_norm'] * df['lon_norm']
    return df

def prepare_X(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X

def run_downscaling(season, ct_df, model):
    mb = build_miniboxes(ct_df, season)
    X  = prepare_X(mb)
    mb['predicted_residual'] = model.predict(X)
    mb['downscaled_co2']     = mb['ct_co2'] + mb['predicted_residual']
    return mb

# ── VIZ BUILDERS ─────────────────────────────────────────────────────────────
def figs_ct(ct):
    out = {}

    ct_mean = ct.groupby(['latitude','longitude'], as_index=False).agg(
        co2=('co2','mean'), blh=('blh','mean'),
        temperature=('temperature','mean'),
        specific_humidity=('specific_humidity','mean'),
        pressure=('pressure','mean'),
    )
    zmin, zmax = tight_range(ct_mean['co2'])
    out['map'] = choropleth_box_map(
        ct_mean, 'co2',
        'CarbonTracker — Annual Mean XCO₂ (2°×3° Grid)',
        colorscale='Plasma', zmin=zmin, zmax=zmax, height=520,
    ).to_json()

    monthly = ct.groupby('month').agg(mean=('co2','mean'), std=('co2','std')).reset_index()
    monthly['lbl'] = monthly['month'].map(MONTH_LABELS)
    out['monthly'] = go.Figure([
        go.Scatter(x=monthly['lbl'], y=monthly['mean']+monthly['std'],
                   mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=monthly['lbl'], y=monthly['mean']-monthly['std'],
                   fill='tonexty', mode='lines', line=dict(width=0),
                   fillcolor='rgba(67,97,238,0.10)', name='±1 σ'),
        go.Scatter(x=monthly['lbl'], y=monthly['mean'], mode='lines+markers',
                   line=dict(color='#4361ee', width=2.5),
                   marker=dict(size=7, color='#4361ee', line=dict(width=2, color='#fff')),
                   name='Monthly mean'),
    ]).update_layout(
        **BASE_LAYOUT, height=330,
        title=dict(text='Monthly Mean CO₂ — CarbonTracker (±1σ band)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    ).to_json()

    fig_v = go.Figure()
    for s in SEASON_ORDER:
        sub = ct[ct['season'] == s]['co2'].dropna()
        fig_v.add_trace(go.Violin(
            y=sub, name=s, box_visible=True, meanline_visible=True,
            fillcolor=SEASON_COLORS[s], line_color='#555',
            opacity=0.82, points=False,
        ))
    fig_v.update_layout(
        **BASE_LAYOUT, height=330, showlegend=False,
        title=dict(text='CO₂ Distribution by IMD Season',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
    )
    out['violin'] = fig_v.to_json()

    sub = ct.dropna(subset=['blh','co2']).sample(min(len(ct), 5000), random_state=42)
    fig_d = go.Figure(go.Histogram2dContour(
        x=sub['blh'], y=sub['co2'], colorscale='Blues', ncontours=18,
        contours=dict(showlabels=False), line=dict(width=0.5),
        colorbar=dict(thickness=12, title=dict(text='density', side='right'),
                      tickfont=dict(size=10), outlinewidth=0),
    ))
    for s in SEASON_ORDER:
        ss = ct[ct['season']==s].dropna(subset=['blh','co2'])
        ss = ss.sample(min(len(ss), 200), random_state=1)
        fig_d.add_trace(go.Scatter(
            x=ss['blh'], y=ss['co2'], mode='markers', name=s,
            marker=dict(color=SEASON_COLORS[s], size=4, opacity=0.72),
        ))
    fig_d.update_layout(
        **BASE_LAYOUT, height=330,
        title=dict(text='BLH vs CO₂ — Density Contour + Season Overlay',
                   font=dict(size=13), x=0.02, xanchor='left'),
        xaxis=dict(title='Boundary Layer Height (m)', gridcolor='#e2e6f0', zeroline=False),
        yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
    )
    out['extra'] = fig_d.to_json()

    if 'temperature' in ct.columns:
        fig_tc = go.Figure()
        for s in SEASON_ORDER:
            ss = ct[ct['season']==s].dropna(subset=['temperature','co2'])
            ss = ss.sample(min(len(ss), 600), random_state=2)
            fig_tc.add_trace(go.Scatter(
                x=ss['temperature'], y=ss['co2'], mode='markers', name=s,
                marker=dict(color=SEASON_COLORS[s], size=4, opacity=0.70),
            ))
        fig_tc.update_layout(
            **BASE_LAYOUT, height=340,
            title=dict(text='Surface Temperature vs CO₂ by Season',
                       font=dict(size=13), x=0.02, xanchor='left'),
            xaxis=dict(title='Temperature (K)', gridcolor='#e2e6f0', zeroline=False),
            yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        )
        out['temp_co2'] = fig_tc.to_json()

    if 'specific_humidity' in ct.columns:
        fig_hc = go.Figure()
        for s in SEASON_ORDER:
            ss = ct[ct['season']==s].dropna(subset=['specific_humidity','co2'])
            ss = ss.sample(min(len(ss), 600), random_state=3)
            fig_hc.add_trace(go.Scatter(
                x=ss['specific_humidity'], y=ss['co2'], mode='markers', name=s,
                marker=dict(color=SEASON_COLORS[s], size=4, opacity=0.70),
            ))
        fig_hc.update_layout(
            **BASE_LAYOUT, height=340,
            title=dict(text='Specific Humidity vs CO₂ Relationship',
                       font=dict(size=13), x=0.02, xanchor='left'),
            xaxis=dict(title='Specific Humidity (kg/kg)', gridcolor='#e2e6f0', zeroline=False),
            yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        )
        out['humidity_co2'] = fig_hc.to_json()

    ann_mean = float(ct['co2'].mean())
    anom = ct.groupby('month')['co2'].mean().reset_index()
    anom['anomaly'] = anom['co2'] - ann_mean
    anom['lbl'] = anom['month'].map(MONTH_LABELS)
    bar_colors = ['#e53e3e' if v > 0 else '#4361ee' for v in anom['anomaly']]
    fig_an = go.Figure(go.Bar(
        x=anom['lbl'], y=anom['anomaly'],
        marker_color=bar_colors, opacity=0.88,
        text=anom['anomaly'].round(2), textposition='outside',
        textfont=dict(size=10),
    ))
    fig_an.add_hline(y=0, line_color='#8b93b0', line_width=1, line_dash='dot')
    fig_an.update_layout(
        **BASE_LAYOUT, height=340, showlegend=False,
        title=dict(text='Monthly CO₂ Anomaly (Δ from Annual Mean)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='Δ CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['anomaly'] = fig_an.to_json()

    if 'blh' in ct.columns:
        mon_blh = ct.groupby('month').agg(
            co2_mean=('co2','mean'), blh_mean=('blh','mean')).reset_index()
        mon_blh['lbl'] = mon_blh['month'].map(MONTH_LABELS)
        fig_bd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bd.add_trace(go.Scatter(
            x=mon_blh['lbl'], y=mon_blh['co2_mean'], name='CO₂ (ppm)',
            mode='lines+markers', line=dict(color='#4361ee', width=2.5),
            marker=dict(size=7, color='#4361ee', line=dict(width=2, color='#fff')),
        ), secondary_y=False)
        fig_bd.add_trace(go.Scatter(
            x=mon_blh['lbl'], y=mon_blh['blh_mean'], name='BLH (m)',
            mode='lines+markers', line=dict(color='#e8622a', width=2.5, dash='dot'),
            marker=dict(size=7, color='#e8622a', line=dict(width=2, color='#fff')),
        ), secondary_y=True)
        fig_bd.update_layout(
            **BASE_LAYOUT, height=360,
            title=dict(text='BLH & CO₂ Monthly Co-variation — Dual Axis',
                       font=dict(size=13), x=0.02, xanchor='left'),
        )
        fig_bd.update_yaxes(title_text='CO₂ (ppm)', secondary_y=False, gridcolor='#e2e6f0')
        fig_bd.update_yaxes(title_text='BLH (m)', secondary_y=True)
        out['blh_dual'] = fig_bd.to_json()

    if 'pressure' in ct.columns:
        samp = ct.dropna(subset=['pressure','co2']).sample(min(len(ct), 3000), random_state=5)
        fig_pc = go.Figure()
        for s in SEASON_ORDER:
            ss = samp[samp['season'] == s]
            fig_pc.add_trace(go.Scatter(
                x=ss['pressure'], y=ss['co2'], mode='markers', name=s,
                marker=dict(color=SEASON_COLORS[s], size=4, opacity=0.72),
            ))
        fig_pc.update_layout(
            **BASE_LAYOUT, height=360,
            title=dict(text='Surface Pressure vs CO₂ Scatter',
                       font=dict(size=13), x=0.02, xanchor='left'),
            xaxis=dict(title='Pressure (Pa)', gridcolor='#e2e6f0', zeroline=False),
            yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        )
        out['pressure_co2'] = fig_pc.to_json()

    # SPEED FIX: seasonal_maps Plotly figure removed — Leaflet handles this via /api/geodata
    # out['seasonal_maps'] = fig_sm.to_json()  ← deleted, saves ~1-2s per CT load

    return out


# ─── Satellite shared helpers ─────────────────────────────────────────────────
def _sat_base(df, name, color):
    out = {}
    band_rgba = hex_to_rgba(color, 0.15)

    # SPEED FIX: Plotly scatter_geo map removed — Leaflet handles this via /api/geodata
    # Building it took 2-3s and was never rendered. Frontend handles missing figs['map'].
    # out['map'] = fig_map.to_json()  ← deleted

    mon = df.groupby('month').agg(
        mean=('co2','mean'), std=('co2','std'), n=('co2','size')).reset_index()
    mon['lbl'] = mon['month'].map(MONTH_LABELS)
    sem = mon['std'] / np.sqrt(mon['n'])
    out['monthly'] = go.Figure([
        go.Scatter(x=mon['lbl'], y=mon['mean']+sem,
                   mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=mon['lbl'], y=mon['mean']-sem,
                   fill='tonexty', mode='lines', line=dict(width=0),
                   fillcolor=band_rgba, name='±SE'),
        go.Scatter(x=mon['lbl'], y=mon['mean'], mode='lines+markers',
                   line=dict(color=color, width=2.5),
                   marker=dict(size=7, color=color, line=dict(width=2, color='#fff')),
                   name='Monthly mean'),
    ]).update_layout(
        **BASE_LAYOUT, height=330,
        title=dict(text=f'{name} — Monthly Mean XCO₂ (±SE band)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    ).to_json()

    fig_box = go.Figure()
    for s in SEASON_ORDER:
        sub = df[df['season']==s]['co2'].dropna()
        fig_box.add_trace(go.Box(
            y=sub, name=s, marker_color=SEASON_COLORS[s],
            boxmean='sd', line_color='#666', opacity=0.85,
        ))
    fig_box.update_layout(
        **BASE_LAYOUT, height=330, showlegend=False,
        title=dict(text=f'{name} — Seasonal CO₂ Distribution',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
    )
    out['seasonal'] = fig_box.to_json()

    df2 = df.copy()
    df2['lat_band'] = (df2['latitude'] // 2) * 2
    lp = df2.groupby('lat_band')['co2'].agg(['mean','std']).reset_index()
    out['extra'] = go.Figure([
        go.Scatter(x=lp['mean']+lp['std'], y=lp['lat_band'].astype(str),
                   mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=lp['mean']-lp['std'], y=lp['lat_band'].astype(str),
                   fill='tonextx', mode='lines', line=dict(width=0),
                   fillcolor=band_rgba, name='±1 σ'),
        go.Scatter(x=lp['mean'], y=lp['lat_band'].astype(str),
                   mode='markers+lines',
                   line=dict(color=color, width=2.5),
                   marker=dict(size=8, color=color, line=dict(width=2, color='#fff')),
                   name='Mean XCO₂'),
    ]).update_layout(
        **BASE_LAYOUT, height=330,
        title=dict(text=f'{name} — CO₂ Meridional Profile (2° Lat Bins)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        xaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        yaxis=dict(title='Latitude band (°N)', gridcolor='#e2e6f0', zeroline=False),
    ).to_json()

    return out


def _sat_year_compare(df, name):
    years = sorted(df['year'].dropna().unique().astype(int))
    year_pal = ['#4361ee','#e8622a','#16a34a','#9333ea','#0ea5e9']
    fig = go.Figure()
    for i, yr in enumerate(years):
        sub = df[df['year']==yr].groupby('month')['co2'].mean().reset_index()
        sub['lbl'] = sub['month'].map(MONTH_LABELS)
        col = year_pal[i % len(year_pal)]
        fig.add_trace(go.Scatter(
            x=sub['lbl'], y=sub['co2'], name=str(yr),
            mode='lines+markers',
            line=dict(color=col, width=2.5),
            marker=dict(size=7, color=col, line=dict(width=2, color='#fff')),
        ))
    fig.update_layout(
        **BASE_LAYOUT, height=360,
        title=dict(text=f'{name} — Year-over-Year Monthly Comparison',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    return fig.to_json()


def _sat_histogram(df, name):
    years = sorted(df['year'].dropna().unique().astype(int))
    year_pal = ['#4361ee','#e8622a','#16a34a','#9333ea']
    fig = go.Figure()
    for i, yr in enumerate(years):
        sub = df[df['year']==yr]['co2'].dropna()
        fig.add_trace(go.Histogram(
            x=sub, name=str(yr), opacity=0.72,
            marker_color=year_pal[i % len(year_pal)],
            nbinsx=60,
        ))
    fig.update_layout(
        **BASE_LAYOUT, height=360, barmode='overlay',
        title=dict(text=f'{name} — XCO₂ Frequency Distribution by Year',
                   font=dict(size=13), x=0.02, xanchor='left'),
        xaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0'),
        yaxis=dict(title='Count', gridcolor='#e2e6f0', zeroline=False),
    )
    return fig.to_json()


def _sat_month_heat(df, name):
    pivot = df.groupby(['year','month'])['co2'].mean().unstack(level='month')
    pivot.columns = [MONTH_LABELS[c] for c in pivot.columns]
    z_vals    = pivot.values
    z_display = np.where(np.isnan(z_vals), None, np.round(z_vals, 1))
    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=list(pivot.columns),
        y=[str(y) for y in pivot.index],
        colorscale='RdYlBu_r',
        text=z_display,
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorbar=dict(
            thickness=12,
            title=dict(text='ppm', side='right'),
            tickfont=dict(size=9),
            outlinewidth=0,
        ),
    ))
    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8faff',
        font=dict(family='Inter, system-ui, sans-serif', size=11, color='#1e2338'),
        margin=dict(l=12, r=12, t=48, b=12),
        height=340,
        title=dict(text=f'{name} — Month × Year Mean XCO₂ Heatmap',
                   font=dict(size=13), x=0.02, xanchor='left'),
    )
    return fig.to_json()


def _sat_anomaly(df, name, color):
    grand = float(df['co2'].mean())
    anom  = df.groupby('month')['co2'].mean().reset_index()
    anom['anomaly'] = anom['co2'] - grand
    anom['lbl'] = anom['month'].map(MONTH_LABELS)
    bar_colors = ['#e53e3e' if v > 0 else '#4361ee' for v in anom['anomaly']]
    fig = go.Figure(go.Bar(
        x=anom['lbl'], y=anom['anomaly'],
        marker_color=bar_colors, opacity=0.88,
        text=anom['anomaly'].round(2), textposition='outside',
        textfont=dict(size=10),
    ))
    fig.add_hline(y=0, line_color='#8b93b0', line_width=1, line_dash='dot')
    fig.update_layout(
        **BASE_LAYOUT, height=340, showlegend=False,
        title=dict(text=f'{name} — Monthly XCO₂ Anomaly (Δ from Multi-Year Mean)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='Δ XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    return fig.to_json()


def _sat_obs_density(df, name):
    fig = go.Figure(go.Histogram2d(
        x=df['longitude'], y=df['latitude'],
        colorscale='Hot_r', nbinsx=35, nbinsy=35,
        colorbar=dict(
            thickness=12,
            title=dict(text='obs', side='right'),
            tickfont=dict(size=9),
            outlinewidth=0,
        ),
    ))
    fig.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text=f'{name} — Observation Spatial Density (2D Histogram)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        xaxis=dict(title='Longitude (°E)', gridcolor='#e2e6f0', range=[66,100]),
        yaxis=dict(title='Latitude (°N)', gridcolor='#e2e6f0', range=[6,38]),
    )
    return fig.to_json()


def _sat_lon_profile(df, name, color):
    df2 = df.copy()
    df2['lon_band'] = (df2['longitude'] // 2) * 2
    lp = df2.groupby('lon_band')['co2'].agg(['mean','std']).reset_index()
    band_rgba = hex_to_rgba(color, 0.15)
    fig = go.Figure([
        go.Scatter(x=lp['lon_band'], y=lp['mean']+lp['std'],
                   mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=lp['lon_band'], y=lp['mean']-lp['std'],
                   fill='tonexty', mode='lines', line=dict(width=0),
                   fillcolor=band_rgba, name='±1 σ'),
        go.Scatter(x=lp['lon_band'], y=lp['mean'], mode='lines+markers',
                   line=dict(color=color, width=2.5),
                   marker=dict(size=7, color=color, line=dict(width=2, color='#fff')),
                   name='Mean XCO₂'),
    ])
    fig.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text=f'{name} — CO₂ Zonal Profile (2° Lon Bins)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        xaxis=dict(title='Longitude (°E)', gridcolor='#e2e6f0', zeroline=False),
        yaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
    )
    return fig.to_json()


# ─── GOSAT ────────────────────────────────────────────────────────────────────
def figs_gosat(df):
    color = '#16a34a'
    out   = _sat_base(df, 'GOSAT', color)
    out['year_compare'] = _sat_year_compare(df, 'GOSAT')
    out['histogram']    = _sat_histogram(df, 'GOSAT')
    out['month_heat']   = _sat_month_heat(df, 'GOSAT')
    out['anomaly']      = _sat_anomaly(df, 'GOSAT', color)
    out['obs_density']  = _sat_obs_density(df, 'GOSAT')
    out['lon_profile']  = _sat_lon_profile(df, 'GOSAT', color)
    return out


# ─── OCO-2 ────────────────────────────────────────────────────────────────────
def figs_oco2(df):
    color = '#9333ea'
    out   = _sat_base(df, 'OCO-2', color)
    out['year_compare'] = _sat_year_compare(df, 'OCO-2')
    out['month_heat']   = _sat_month_heat(df, 'OCO-2')
    out['histogram']    = _sat_histogram(df, 'OCO-2')
    out['anomaly']      = _sat_anomaly(df, 'OCO-2', color)
    out['obs_density']  = _sat_obs_density(df, 'OCO-2')

    df_s = df.dropna(subset=['time','co2']).sort_values('time').set_index('time')
    if len(df_s) > 90:
        rolled = df_s['co2'].rolling('90D').mean().reset_index()
        raw_df = df_s['co2'].reset_index()
        fig_r  = go.Figure([
            go.Scatter(x=raw_df['time'], y=raw_df['co2'], mode='markers',
                       marker=dict(color=color, size=2, opacity=0.25),
                       name='Observations'),
            go.Scatter(x=rolled['time'], y=rolled['co2'],
                       mode='lines', name='90-day rolling mean',
                       line=dict(color='#1e2338', width=2.5)),
        ])
        fig_r.update_layout(
            **BASE_LAYOUT, height=360,
            title=dict(text='OCO-2 — Rolling 90-Day Smoothed CO₂ Trend',
                       font=dict(size=13), x=0.02, xanchor='left'),
            xaxis=dict(title='Date', gridcolor='#e2e6f0'),
            yaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        )
        out['rolling'] = fig_r.to_json()

    return out


# ─── OCO-3 ────────────────────────────────────────────────────────────────────
def figs_oco3(df):
    color = '#ea580c'
    out   = _sat_base(df, 'OCO-3', color)
    out['year_compare'] = _sat_year_compare(df, 'OCO-3')
    out['month_heat']   = _sat_month_heat(df, 'OCO-3')
    out['histogram']    = _sat_histogram(df, 'OCO-3')
    out['anomaly']      = _sat_anomaly(df, 'OCO-3', color)
    out['obs_density']  = _sat_obs_density(df, 'OCO-3')

    df2     = df.copy()
    lat_mid = float(df2['latitude'].median())
    lon_mid = float(df2['longitude'].median())
    def _region(row):
        ns = 'N' if row['latitude'] >= lat_mid else 'S'
        ew = 'E' if row['longitude'] >= lon_mid else 'W'
        return f'{ns}{ew} India'
    df2['region'] = df2.apply(_region, axis=1)
    reg_colors = {'NE India':'#4361ee','NW India':'#16a34a',
                  'SE India':'#9333ea','SW India':'#ea580c'}
    fig_reg = go.Figure()
    for reg, col in reg_colors.items():
        sub = df2[df2['region']==reg].groupby('month')['co2'].mean().reset_index()
        sub['lbl'] = sub['month'].map(MONTH_LABELS)
        fig_reg.add_trace(go.Scatter(
            x=sub['lbl'], y=sub['co2'], name=reg,
            mode='lines+markers',
            line=dict(color=col, width=2.5),
            marker=dict(size=7, color=col, line=dict(width=2, color='#fff')),
        ))
    fig_reg.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text='OCO-3 — Regional CO₂: N/S/E/W India Monthly Comparison',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['regional'] = fig_reg.to_json()

    return out


# ─── Compare All ──────────────────────────────────────────────────────────────
def figs_compare(d):
    ds_cfg = [
        ('CarbonTracker', 'ct',    '#4361ee'),
        ('GOSAT',         'gosat', '#16a34a'),
        ('OCO-2',         'oco2',  '#9333ea'),
        ('OCO-3',         'oco3',  '#ea580c'),
    ]
    out = {}

    fig_t = go.Figure()
    for name, key, col in ds_cfg:
        mon = d[key].groupby('month')['co2'].mean().reset_index()
        mon['lbl'] = mon['month'].map(MONTH_LABELS)
        fig_t.add_trace(go.Scatter(
            x=mon['lbl'], y=mon['co2'], mode='lines+markers', name=name,
            line=dict(color=col, width=2.5),
            marker=dict(size=7, color=col, line=dict(width=2, color='#fff')),
        ))
    fig_t.update_layout(
        **BASE_LAYOUT, height=400,
        title=dict(text='Monthly CO₂ Trend — All 4 Datasets Overlaid',
                   font=dict(size=14), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ / XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['trend'] = fig_t.to_json()

    fig_v = go.Figure()
    for name, key, col in ds_cfg:
        sub = d[key]['co2'].sample(min(len(d[key]),3000), random_state=1)
        fig_v.add_trace(go.Violin(
            y=sub, name=name, box_visible=True, meanline_visible=True,
            fillcolor=col, line_color='#aaa', opacity=0.8, points=False,
        ))
    fig_v.update_layout(
        **BASE_LAYOUT, height=360, showlegend=False,
        title=dict(text='CO₂ Distribution Comparison — Violin',
                   font=dict(size=14), x=0.02, xanchor='left'),
        yaxis=dict(title='ppm', gridcolor='#e2e6f0', zeroline=False),
    )
    out['violin'] = fig_v.to_json()

    counts = []
    for name, key, _ in ds_cfg:
        for s in SEASON_ORDER:
            counts.append({'Dataset': name, 'Season': s,
                           'Count': int((d[key]['season']==s).sum())})
    pivot_h = pd.DataFrame(counts).pivot(index='Dataset', columns='Season', values='Count')[SEASON_ORDER]
    fig_h = go.Figure(go.Heatmap(
        z=pivot_h.values, x=SEASON_ORDER, y=pivot_h.index.tolist(),
        colorscale='Blues',
        text=pivot_h.values, texttemplate='%{text:,.0f}',
        textfont=dict(size=11, color='#1e2338'),
        colorbar=dict(
            thickness=12,
            title=dict(text='obs', side='right'),
            tickfont=dict(size=10),
            outlinewidth=0,
        ),
    ))
    fig_h.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8faff',
        font=dict(family='Inter, system-ui, sans-serif', size=12, color='#1e2338'),
        margin=dict(l=12, r=12, t=52, b=12), height=360,
        title=dict(text='Observation Count — Dataset × Season',
                   font=dict(size=13), x=0.02, xanchor='left'),
    )
    out['heatmap'] = fig_h.to_json()

    rows = []
    for name, key, col in ds_cfg:
        sm = d[key].groupby('season')['co2'].mean()
        for s in SEASON_ORDER:
            if s in sm.index:
                rows.append({'Dataset': name, 'Season': s, 'CO2': sm[s]})
    df_bar = pd.DataFrame(rows)
    fig_bar = go.Figure()
    for name, key, col in ds_cfg:
        sub = df_bar[df_bar['Dataset']==name]
        fig_bar.add_trace(go.Bar(
            x=sub['Season'], y=sub['CO2'], name=name,
            marker_color=col, opacity=0.88,
            text=sub['CO2'].round(2), textposition='outside',
            textfont=dict(size=10, color='#4a5278'),
        ))
    fig_bar.update_layout(
        **BASE_LAYOUT, height=360, barmode='group',
        title=dict(text='Seasonal Mean CO₂ — Grouped Bar',
                   font=dict(size=14), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ (ppm)',
                   range=[df_bar['CO2'].min()-3, df_bar['CO2'].max()+5],
                   gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Season', gridcolor='#e2e6f0'),
        bargap=0.18, bargroupgap=0.05,
    )
    out['bar'] = fig_bar.to_json()

    fig_rng = go.Figure()
    for name, key, col in ds_cfg:
        vals = d[key]['co2'].dropna()
        mn, mx, mu = float(vals.min()), float(vals.max()), float(vals.mean())
        fig_rng.add_trace(go.Bar(
            x=[name], y=[mu],
            error_y=dict(type='data', array=[mx-mu], arrayminus=[mu-mn],
                         visible=True, color='#555', thickness=2, width=8),
            marker_color=col, opacity=0.85, name=name,
        ))
    fig_rng.update_layout(
        **BASE_LAYOUT, height=360, showlegend=False,
        title=dict(text='CO₂ Value Range — Min / Mean / Max per Dataset',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Dataset'),
    )
    out['range_compare'] = fig_rng.to_json()

    fig_lat = go.Figure()
    for name, key, col in ds_cfg:
        df2 = d[key].copy()
        df2['lat_band'] = (df2['latitude'] // 2) * 2
        lp = df2.groupby('lat_band')['co2'].mean().reset_index()
        fig_lat.add_trace(go.Scatter(
            x=lp['co2'], y=lp['lat_band'].astype(str),
            mode='lines+markers', name=name,
            line=dict(color=col, width=2.5),
            marker=dict(size=6, color=col, line=dict(width=1.5, color='#fff')),
        ))
    fig_lat.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text='Latitude CO₂ Profile — All Datasets Overlaid',
                   font=dict(size=14), x=0.02, xanchor='left'),
        xaxis=dict(title='XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        yaxis=dict(title='Latitude band (°N)', gridcolor='#e2e6f0', zeroline=False),
    )
    out['lat_all'] = fig_lat.to_json()

    fig_anorm = go.Figure()
    for name, key, col in ds_cfg:
        mon = d[key].groupby('month')['co2'].mean()
        norm = (mon - mon.mean()) / (mon.std() + 1e-9)
        norm = norm.reset_index()
        norm['lbl'] = norm['month'].map(MONTH_LABELS)
        fig_anorm.add_trace(go.Scatter(
            x=norm['lbl'], y=norm['co2'], name=name,
            mode='lines+markers',
            line=dict(color=col, width=2.5),
            marker=dict(size=6, color=col, line=dict(width=1.5, color='#fff')),
        ))
    fig_anorm.add_hline(y=0, line_color='#8b93b0', line_width=1, line_dash='dot')
    fig_anorm.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text='Normalised Monthly CO₂ Anomaly — All Datasets',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='Normalised Anomaly (σ)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['anomaly_norm'] = fig_anorm.to_json()

    fig_y24 = go.Figure()
    for name, key, col in ds_cfg:
        sub = d[key][d[key]['year']==2024].groupby('month')['co2'].mean().reset_index()
        if sub.empty:
            continue
        sub['lbl'] = sub['month'].map(MONTH_LABELS)
        fig_y24.add_trace(go.Scatter(
            x=sub['lbl'], y=sub['co2'], name=name,
            mode='lines+markers',
            line=dict(color=col, width=2.5),
            marker=dict(size=7, color=col, line=dict(width=2, color='#fff')),
        ))
    fig_y24.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text='2024 Monthly CO₂ — Dataset Overlap Trend',
                   font=dict(size=14), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ / XCO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['year_2024'] = fig_y24.to_json()

    fig_ridge = go.Figure()
    for s in SEASON_ORDER:
        vals = pd.concat([d[key][d[key]['season']==s]['co2']
                          for _, key, _ in ds_cfg]).dropna()
        fig_ridge.add_trace(go.Violin(
            x=vals, name=s, orientation='h',
            side='positive', width=1.8, points=False,
            fillcolor=SEASON_COLORS[s], line_color='#666',
            opacity=0.78, meanline_visible=True,
        ))
    fig_ridge.update_layout(
        **BASE_LAYOUT, height=380,
        title=dict(text='Seasonal CO₂ Ridge Distribution (All Datasets)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        xaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False),
        yaxis=dict(title='Season', gridcolor='#e2e6f0'),
        violingap=0.05, violingroupgap=0.0,
    )
    out['season_ridge'] = fig_ridge.to_json()

    pairs = [
        ('CarbonTracker','ct',  '#4361ee', 'GOSAT', 'gosat','#16a34a'),
        ('CarbonTracker','ct',  '#4361ee', 'OCO-2', 'oco2', '#9333ea'),
        ('CarbonTracker','ct',  '#4361ee', 'OCO-3', 'oco3', '#ea580c'),
    ]
    fig_sp = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{a} vs {b}' for a,_,_,b,_,_ in pairs],
        horizontal_spacing=0.07,
    )
    for ci, (n1,k1,c1,n2,k2,c2) in enumerate(pairs, 1):
        m1 = d[k1].groupby('month')['co2'].mean().reset_index().rename(columns={'co2':'co2_x'})
        m2 = d[k2].groupby('month')['co2'].mean().reset_index().rename(columns={'co2':'co2_y'})
        merged = m1.merge(m2, on='month')
        show_cb = (ci == 3)
        fig_sp.add_trace(go.Scatter(
            x=merged['co2_x'], y=merged['co2_y'], mode='markers',
            marker=dict(
                color=merged['month'], colorscale='Rainbow', size=10,
                line=dict(width=1, color='#fff'), opacity=0.88,
                showscale=show_cb,
                colorbar=dict(
                    thickness=10,
                    title=dict(text='month', side='right'),
                    tickfont=dict(size=9), x=1.02,
                ) if show_cb else {},
            ),
            name=f'{n1} vs {n2}',
        ), row=1, col=ci)
        mn = min(merged['co2_x'].min(), merged['co2_y'].min()) - 1
        mx = max(merged['co2_x'].max(), merged['co2_y'].max()) + 1
        fig_sp.add_trace(go.Scatter(
            x=[mn,mx], y=[mn,mx], mode='lines',
            line=dict(color='#8b93b0', width=1, dash='dot'),
            showlegend=False,
        ), row=1, col=ci)
        fig_sp.update_xaxes(title_text=f'{n1} (ppm)', gridcolor='#e2e6f0', row=1, col=ci)
        fig_sp.update_yaxes(title_text=f'{n2} (ppm)', gridcolor='#e2e6f0', row=1, col=ci)
    fig_sp.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8faff',
        font=dict(family='Inter, system-ui, sans-serif', size=11, color='#1e2338'),
        margin=dict(l=12, r=50, t=52, b=12),
        height=440, showlegend=False,
        title=dict(text='Dataset vs Dataset Scatter — Monthly Mean Pairs',
                   font=dict(size=14), x=0.02, xanchor='left'),
    )
    out['scatter_pairs'] = fig_sp.to_json()

    return out


# ── ATTRIBUTION ANALYSIS ──────────────────────────────────────────────────────
def figs_attribution(attr_df):
    """Build attribution analysis charts from india_ct_co2_with_attribution.csv."""
    out = {}
    df = attr_df.copy()
    if 'season' not in df.columns:
        df['season'] = df['month'].apply(get_season)

    # Monthly India means
    monthly = df.groupby('month').agg(
        co2            =('co2',             'mean'),
        co2_anthro     =('co2_anthro_only',  'mean'),
        co2_natural    =('co2_natural_only', 'mean'),
        co2_anthro_std =('co2_anthro_only',  'std'),
        co2_natural_std=('co2_natural_only', 'std'),
        f_anthro       =('f_anthro',         'mean'),
        f_natural      =('f_natural',        'mean'),
    ).reset_index().sort_values('month')
    monthly['lbl'] = monthly['month'].map(MONTH_LABELS)

    # ── 1. Monthly separation — 3 lines with ±1σ bands ────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=monthly['lbl'],
        y=monthly['co2_anthro']+monthly['co2_anthro_std'],
        mode='lines', line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=monthly['lbl'],
        y=monthly['co2_anthro']-monthly['co2_anthro_std'],
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(213,94,0,0.10)', name='Anthro ±1σ'))
    fig1.add_trace(go.Scatter(x=monthly['lbl'],
        y=monthly['co2_natural']+monthly['co2_natural_std'],
        mode='lines', line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=monthly['lbl'],
        y=monthly['co2_natural']-monthly['co2_natural_std'],
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(0,119,187,0.10)', name='Natural ±1σ'))
    fig1.add_trace(go.Scatter(x=monthly['lbl'], y=monthly['co2'],
        mode='lines+markers', name='Observed CO₂',
        line=dict(color='#222222', width=2.5),
        marker=dict(size=7, color='#222222', line=dict(width=2, color='#fff'))))
    fig1.add_trace(go.Scatter(x=monthly['lbl'], y=monthly['co2_anthro'],
        mode='lines+markers', name='Anthropogenic (fossil fuel)',
        line=dict(color='#D55E00', width=2.2, dash='dash'),
        marker=dict(size=6, color='#D55E00', symbol='square',
                    line=dict(width=1.5, color='#fff'))))
    fig1.add_trace(go.Scatter(x=monthly['lbl'], y=monthly['co2_natural'],
        mode='lines+markers', name='Natural (bio + ocean + fire)',
        line=dict(color='#0077BB', width=2.2, dash='dashdot'),
        marker=dict(size=6, color='#0077BB', symbol='triangle-up',
                    line=dict(width=1.5, color='#fff'))))
    fig1.add_hline(y=CO2_BG, line_color='#a0a0a0', line_width=1, line_dash='dot',
                   annotation_text='280 ppm pre-industrial',
                   annotation_font_size=9, annotation_font_color='#a0a0a0')
    fig1.update_layout(**BASE_LAYOUT, height=380,
        title=dict(text='India CO₂ Attribution — Observed vs Anthropogenic vs Natural (2024)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False, range=[270, 445]),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    fig1.update_layout(legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', font=dict(size=10)))
    out['separation'] = fig1.to_json()

    # ── 2. Enhancement bars ────────────────────────────────────────────────
    ea = monthly['co2_anthro']  - CO2_BG
    en = monthly['co2_natural'] - CO2_BG
    et = monthly['co2']         - CO2_BG
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly['lbl'], y=ea, name='Anthropogenic',
        marker_color='#D55E00', opacity=0.82, offsetgroup=1))
    fig2.add_trace(go.Bar(x=monthly['lbl'], y=en, name='Natural',
        marker_color='#0077BB', opacity=0.82, offsetgroup=2))
    fig2.add_trace(go.Scatter(x=monthly['lbl'], y=et, name='Total',
        mode='lines+markers', line=dict(color='#222222', width=2),
        marker=dict(size=6, color='#222222')))
    fig2.update_layout(**BASE_LAYOUT, height=340, barmode='group',
        title=dict(text='CO₂ Enhancement above Pre-industrial Baseline (280 ppm)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='Enhancement (ppm)', gridcolor='#e2e6f0', zeroline=False),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['enhancement'] = fig2.to_json()

    # ── 3. Attribution fractions — stacked bar ─────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=monthly['lbl'], y=monthly['f_anthro']*100,
        name='Anthropogenic (%)', marker_color='#D55E00', opacity=0.83))
    fig3.add_trace(go.Bar(x=monthly['lbl'], y=monthly['f_natural']*100,
        name='Natural (%)', marker_color='#0077BB', opacity=0.83))
    for _, row in monthly.iterrows():
        fig3.add_annotation(x=row['lbl'], y=row['f_anthro']*50,
            text=f"{row['f_anthro']*100:.0f}%",
            showarrow=False, font=dict(color='white', size=9, family='JetBrains Mono'))
    fig3.add_hline(y=50, line_color='rgba(255,255,255,0.6)', line_width=1, line_dash='dot')
    fig3.update_layout(**BASE_LAYOUT, height=340, barmode='stack',
        title=dict(text='Monthly Attribution Fractions — Anthropogenic vs Natural (%)',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='Attribution (%)', gridcolor='#e2e6f0', zeroline=False, range=[0,100]),
        xaxis=dict(title='Month', gridcolor='#e2e6f0'),
    )
    out['fractions'] = fig3.to_json()

    # ── 4. Seasonal grouped bar ────────────────────────────────────────────
    seas = df.groupby('season').agg(
        co2        =('co2',             'mean'),
        co2_anthro =('co2_anthro_only',  'mean'),
        co2_natural=('co2_natural_only', 'mean'),
    ).reset_index()
    seas = seas[seas['season'].isin(SEASON_ORDER)].copy()
    seas['season'] = pd.Categorical(seas['season'], categories=SEASON_ORDER, ordered=True)
    seas = seas.sort_values('season')
    ylo = seas[['co2','co2_anthro','co2_natural']].min().min() - 5
    yhi = seas[['co2','co2_anthro','co2_natural']].max().max() + 12
    fig4 = go.Figure()
    for var, name_lbl, col in [
        ('co2',        'Observed',      '#222222'),
        ('co2_anthro', 'Anthropogenic', '#D55E00'),
        ('co2_natural','Natural',       '#0077BB'),
    ]:
        fig4.add_trace(go.Bar(x=seas['season'], y=seas[var], name=name_lbl,
            marker_color=col, opacity=0.85,
            text=seas[var].round(1), textposition='outside',
            textfont=dict(size=9, color='#444')))
    fig4.update_layout(**BASE_LAYOUT, height=320, barmode='group',
        title=dict(text='Seasonal Mean CO₂ — Observed vs Anthropogenic vs Natural',
                   font=dict(size=13), x=0.02, xanchor='left'),
        yaxis=dict(title='CO₂ (ppm)', gridcolor='#e2e6f0', zeroline=False, range=[ylo, yhi]),
        xaxis=dict(title='Season', gridcolor='#e2e6f0'),
        bargap=0.15, bargroupgap=0.05,
    )
    out['seasonal'] = fig4.to_json()

    # ── 5. Spatial anthropogenic fraction map ──────────────────────────────
    if 'latitude' in df.columns and 'longitude' in df.columns:
        sp = df.groupby(['latitude','longitude']).agg(
            f_anthro=('f_anthro', 'mean')
        ).reset_index()
        fig5 = go.Figure(go.Scattergeo(
            lat=sp['latitude'], lon=sp['longitude'],
            mode='markers',
            marker=dict(
                size=16, color=sp['f_anthro']*100,
                colorscale='RdYlGn_r', cmin=40, cmax=75,
                symbol='square', opacity=0.90,
                colorbar=dict(
                    thickness=12, len=0.65,
                    title=dict(text='% Fossil', side='right', font=dict(size=10)),
                    tickfont=dict(size=9), outlinewidth=0,
                ),
                line=dict(width=0.5, color='rgba(255,255,255,0.5)'),
            ),
            text=[f'{v*100:.1f}% fossil' for v in sp['f_anthro']],
            hovertemplate='<b>%{text}</b><br>%{lat:.1f}°N  %{lon:.1f}°E<extra></extra>',
        ))
        fig5.update_geos(**GEO_STYLE)
        fig5.update_layout(
            paper_bgcolor='#ffffff',
            font=dict(family='Inter, system-ui, sans-serif', size=11, color='#1e2338'),
            margin=dict(l=0, r=0, t=44, b=0), height=360,
            title=dict(text='Spatial Anthropogenic Fraction — % Fossil Fuel CO₂ (Annual Mean 2024)',
                       font=dict(size=12), x=0.02, xanchor='left'),
        )
        out['spatial'] = fig5.to_json()

    return out


# ── AI ANALYTICS ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are PAVAN AI — a data analytics assistant for India CO₂ satellite data.
You have direct access to 4 pandas DataFrames already loaded in memory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATAFRAMES:

ct  ·  CarbonTracker model  ·  Jan–Dec 2024  ·  2°×3° India grid
  Columns: latitude, longitude, co2 (ppm), blh (m), temperature (K),
           pressure (Pa), specific_humidity, month (1-12), year, season, time (datetime)

gosat  ·  GOSAT satellite  ·  Jan 2024 – Dec 2025
  Columns: latitude, longitude, co2 (ppm), month, year, season, time (datetime)

oco2  ·  OCO-2 satellite qf=0  ·  Dec 2023 – Oct 2025
  Columns: latitude, longitude, co2 (ppm), month, year, season, time (datetime)

oco3  ·  OCO-3 satellite qf=0  ·  Jul 2024 – Dec 2025
  Columns: latitude, longitude, co2 (ppm), month, year, season, time (datetime)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MONTH NUMBERS: Jan=1 Feb=2 Mar=3 Apr=4 May=5 Jun=6 Jul=7 Aug=8 Sep=9 Oct=10 Nov=11 Dec=12
SEASONS: Winter=[1,2]  Premonsoon=[3,4,5]  Monsoon=[6,7,8,9]  Postmonsoon=[10,11,12]
COLORS: CT=#4361ee  GOSAT=#16a34a  OCO-2=#9333ea  OCO-3=#ea580c

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE VARIABLES AVAILABLE (no imports needed):
  ct, gosat, oco2, oco3        — the DataFrames
  pd, np                        — pandas / numpy
  go                            — plotly.graph_objects
  px                            — plotly.express
  make_subplots                 — from plotly.subplots
  SEASON_COLORS                 — dict: season → hex colour
  MONTH_LABELS                  — dict: int → month name string
  apply_style(fig, title='', height=420, showlegend=True)
                                — PAVAN chart styling helper (see rules below)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATE FILTERING EXAMPLES:
  # date range
  oco2[(oco2['time'] >= '2024-08-01') & (oco2['time'] <= '2025-08-01')]
  # single month + year
  ct[(ct['year']==2024) & (ct['month']==6)]
  # season
  gosat[gosat['season']=='Monsoon']

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY CHART RULES — code that violates these will crash:

RULE 1 — ALWAYS call apply_style() to style a chart. NEVER spread BASE_LAYOUT.
  ✓ CORRECT:
      fig = go.Figure(...)
      apply_style(fig, 'My Chart Title', height=420)
      fig.update_xaxes(title_text='Month')
      fig.update_yaxes(title_text='CO₂ (ppm)')

  ✗ WRONG (causes "multiple values for keyword argument" crash):
      fig.update_layout(**BASE_LAYOUT, height=420, legend=dict(...))

RULE 2 — Colorbar title MUST use dict format:
  ✓ colorbar=dict(thickness=12, title=dict(text='ppm', side='right'))
  ✗ colorbar=dict(title='ppm')          ← crashes
  ✗ colorbar=dict(titleside='right')    ← deprecated, crashes

RULE 3 — After apply_style(), add axis labels with update_xaxes / update_yaxes:
      fig.update_xaxes(title_text='Longitude (°E)')
      fig.update_yaxes(title_text='CO₂ (ppm)')

RULE 4 — NEVER pass the same key twice to update_layout(). For example:
  ✗ fig.update_layout(height=400, height=400)      ← crash
  ✗ fig.update_layout(paper_bgcolor=…, paper_bgcolor=…) ← crash

RULE 5 — Tables: always call .dropna() or .fillna('') before assigning to result.
  result = df.groupby('month')['co2'].mean().reset_index().dropna()

RULE 6 — Geo maps (px.scatter_geo):
      fig = px.scatter_geo(df, lat='latitude', lon='longitude',
                           color='co2', color_continuous_scale='Turbo',
                           range_color=[zmin, zmax])
      fig.update_geos(showcountries=True, showland=True, landcolor='#f0ece4',
                      showocean=True, oceancolor='#d0e8f5',
                      lataxis_range=[6,38], lonaxis_range=[66,100])
      apply_style(fig, 'Map Title', height=480)
      fig.update_coloraxes(colorbar=dict(thickness=14,
                           title=dict(text='ppm', side='right')))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — respond with ONLY valid JSON. No markdown fences. No text outside the JSON.

Table (user wants data rows or statistics):
{
  "type": "table",
  "message": "One sentence description of the data shown.",
  "code": "Python code. Must assign a clean DataFrame to variable 'result'. Max 500 rows. Call .dropna()."
}

Chart (user wants a visualisation):
{
  "type": "chart",
  "message": "One sentence description of what is plotted.",
  "code": "Python code. Must assign a plotly Figure to variable 'fig'. Must call apply_style()."
}

Text (general question, stats, explanation — no code needed):
{
  "type": "text",
  "message": "Answer with specific numbers where relevant."
}"""


def _patch_ai_code(code: str) -> str:
    def fix_base_layout(m):
        before = m.group(1)
        rest   = m.group(2)
        rest = re.sub(r'^\s*,\s*', '', rest).strip()
        if rest:
            return f'apply_style({before})\n{before}.update_layout({rest}'
        return f'apply_style({before}'

    code = re.sub(
        r'(\w+)\.update_layout\(\s*\*\*BASE_LAYOUT\s*,?\s*(.*?)\)',
        fix_base_layout,
        code,
        flags=re.DOTALL,
    )
    code = code.replace('**BASE_LAYOUT,', '').replace('**BASE_LAYOUT', '')
    return code


def run_analytics_code(code_str, result_var='result'):
    code_str = _patch_ai_code(code_str)
    namespace = {
        'ct':            APP_DATA['ct'],
        'gosat':         APP_DATA['gosat'],
        'oco2':          APP_DATA['oco2'],
        'oco3':          APP_DATA['oco3'],
        'pd':            pd,
        'np':            np,
        'go':            go,
        'px':            px,
        'make_subplots': make_subplots,
        'SEASON_COLORS': SEASON_COLORS,
        'MONTH_LABELS':  MONTH_LABELS,
        'apply_style':   apply_style,
        'BASE_LAYOUT':   {},
    }
    exec(compile(code_str, '<pavan_ai>', 'exec'), namespace)
    return namespace.get(result_var)


def _strip_fences(raw: str) -> str:
    raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',     '', raw, flags=re.MULTILINE)
    return raw.strip()


def _call_gemini(prompt: str):
    response = AI_MODEL.generate_content(prompt)
    raw      = _strip_fences(response.text.strip())
    return json.loads(raw), raw


def _retry_prompt(user_msg: str, prev_code: str, error: str, attempt: int) -> str:
    return (
        SYSTEM_PROMPT
        + f"\n\nUser: {user_msg}"
        + f"\n\n\u2501\u2501\u2501 RETRY {attempt}/3 \u2014 FIX THE CODE BELOW \u2501\u2501\u2501"
        + "\nThe code you generated previously raised an error. "
          "Rewrite it so it runs without errors."
        + f"\n\nPrevious code:\n```python\n{prev_code}\n```"
        + f"\n\nError received:\n{error}"
        + "\n\nRespond with corrected JSON only. Same format as before."
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    MAX_RETRIES = 3
    try:
        user_msg = request.json.get('message', '').strip()
        if not user_msg:
            return jsonify({'error': 'Empty message'}), 400

        first_prompt = SYSTEM_PROMPT + f"\n\nUser: {user_msg}"
        try:
            parsed, raw = _call_gemini(first_prompt)
        except json.JSONDecodeError:
            print(f"[CHAT] Non-JSON from Gemini: {raw[:300]}")
            return jsonify({'type': 'text', 'message': raw[:800]})

        resp_type  = parsed.get('type', 'text')
        message    = parsed.get('message', '')
        code       = parsed.get('code', '')

        if resp_type == 'text':
            return jsonify({'type': 'text', 'message': message})

        result_var = 'fig' if resp_type == 'chart' else 'result'
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = run_analytics_code(code, result_var)
                last_error = None
                break
            except Exception as exec_err:
                last_error = str(exec_err)
                print(f"[EXEC ERROR attempt {attempt}] {last_error}\n--- Code ---\n{code}\n---")
                if attempt < MAX_RETRIES:
                    retry_p = _retry_prompt(user_msg, code, last_error, attempt)
                    try:
                        parsed, raw = _call_gemini(retry_p)
                        message    = parsed.get('message', message)
                        code       = parsed.get('code', code)
                        resp_type  = parsed.get('type', resp_type)
                        result_var = 'fig' if resp_type == 'chart' else 'result'
                    except json.JSONDecodeError:
                        break

        if last_error:
            return jsonify({
                'type':    'text',
                'message': f'Could not generate the {resp_type} after {MAX_RETRIES} attempts. '
                           f'Last error: {last_error}',
            })

        if resp_type == 'table':
            if result is None or not isinstance(result, pd.DataFrame):
                return jsonify({'type': 'text', 'message': 'No data returned for that query.'})
            result = result.head(300)
            return jsonify({
                'type':    'table',
                'message': message,
                'columns': list(result.columns),
                'rows':    result.round(4).values.tolist(),
                'shape':   list(result.shape),
                'csv':     result.to_csv(index=False),
            })

        elif resp_type == 'chart':
            if result is None:
                return jsonify({'type': 'text', 'message': 'Chart could not be generated.'})
            return jsonify({
                'type':    'chart',
                'message': message,
                'fig':     result.to_json(),
            })

        return jsonify({'type': 'text', 'message': message})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[CHAT ERROR]\n{tb}")
        return jsonify({'type': 'error', 'message': str(e)}), 500


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/viz/<dataset>')
def viz_dataset(dataset):
    try:
        d = APP_DATA
        if   dataset == 'ct':      return jsonify(figs_ct(d['ct']))
        elif dataset == 'gosat':   return jsonify(figs_gosat(d['gosat']))
        elif dataset == 'oco2':    return jsonify(figs_oco2(d['oco2']))
        elif dataset == 'oco3':    return jsonify(figs_oco3(d['oco3']))
        elif dataset == 'compare': return jsonify(figs_compare(d))
        return jsonify({'error': 'unknown dataset'}), 404
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR /api/viz/{dataset}]\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/attribution')
def attribution():
    try:
        if APP_ATTR is None:
            return jsonify({'error': (
                'Attribution data not loaded. '
                'Copy india_ct_co2_with_attribution.csv into the data/ folder '
                'then restart the server.'
            )}), 404
        return jsonify(figs_attribution(APP_ATTR))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR /api/attribution]\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


# ── DAILY RANGE DOWNSCALING ───────────────────────────────────────────────────
import threading
_range_progress = {'pct': 0, 'label': '', 'sub': '', 'done': True}
_range_lock     = threading.Lock()

def _set_progress(pct, label, sub=''):
    with _range_lock:
        _range_progress.update({'pct': pct, 'label': label, 'sub': sub, 'done': False})

def _set_done():
    with _range_lock:
        _range_progress['done'] = True


@app.route('/api/ct_date_range')
def ct_date_range():
    ct   = APP_DATA['ct']
    dmin = ct['time'].min()
    dmax = ct['time'].max()
    return jsonify({'min': dmin.strftime('%Y-%m-%d'), 'max': dmax.strftime('%Y-%m-%d')})


@app.route('/api/downscale/range/progress')
def range_progress():
    with _range_lock:
        return jsonify(dict(_range_progress))


@app.route('/api/downscale/range', methods=['POST'])
def downscale_range():
    try:
        body      = request.json or {}
        start_str = body.get('start', '')
        end_str   = body.get('end', '')

        if not start_str or not end_str:
            return jsonify({'error': 'start and end dates are required'}), 400

        start_dt = pd.to_datetime(start_str)
        end_dt   = pd.to_datetime(end_str)

        if start_dt > end_dt:
            return jsonify({'error': 'start date must be before end date'}), 400

        ct = APP_DATA['ct'].copy()
        ct['date'] = ct['time'].dt.normalize()

        ct_range = ct[(ct['date'] >= start_dt) & (ct['date'] <= end_dt)]
        if ct_range.empty:
            return jsonify({'error': 'No CT data found in the selected date range.'}), 404

        unique_days = sorted(ct_range['date'].unique())
        n_days      = len(unique_days)

        if n_days == 0:
            return jsonify({'error': 'No days found in range.'}), 404

        _set_progress(0, f'Starting — {n_days} days to process…', '')

        all_frames   = []
        models_used  = set()
        skipped_days = 0

        for i, day in enumerate(unique_days):
            day_label = pd.Timestamp(day).strftime('%Y-%m-%d')
            pct       = int((i / n_days) * 100)
            _set_progress(pct, f'Processing {day_label}  ({i+1}/{n_days})',
                          f'{len(all_frames)} rows accumulated so far')

            day_ct = ct_range[ct_range['date'] == day].groupby(
                ['latitude','longitude'], as_index=False
            ).agg(
                co2=('co2','mean'),
                blh=('blh','mean'),
                temperature=('temperature','mean'),
                pressure=('pressure','mean'),
                specific_humidity=('specific_humidity','mean'),
                season=('season','first'),
            )

            if day_ct.empty or day_ct['co2'].isna().all():
                skipped_days += 1
                continue

            season = day_ct['season'].iloc[0]
            if season not in APP_MODELS:
                day_ct['date']               = day_label
                day_ct['ct_co2']             = day_ct['co2']
                day_ct['downscaled_co2']     = np.nan
                day_ct['predicted_residual'] = np.nan
                all_frames.append(day_ct)
                continue

            models_used.add(season)
            try:
                mb = run_downscaling(season, day_ct, APP_MODELS[season])
                mb['date'] = day_label
                all_frames.append(mb)
            except Exception as day_err:
                print(f"[RANGE] Skipping {day_label}: {day_err}")
                skipped_days += 1
                continue

        _set_progress(98, 'Merging results…', f'{len(all_frames)} daily frames')

        if not all_frames:
            _set_done()
            return jsonify({'error': 'No days could be downscaled. Check that models are loaded.'}), 500

        merged     = pd.concat(all_frames, ignore_index=True)
        out_cols   = ['date','latitude','longitude','season',
                      'ct_co2','downscaled_co2','predicted_residual',
                      'blh','temperature','pressure','specific_humidity']
        merged_out = merged[[c for c in out_cols if c in merged.columns]]

        ct_mean = round(float(merged_out['ct_co2'].mean()), 3) \
                  if 'ct_co2' in merged_out else None
        ds_mean = round(float(merged_out['downscaled_co2'].dropna().mean()), 3) \
                  if 'downscaled_co2' in merged_out else None

        csv_str = merged_out.to_csv(index=False)
        _set_progress(100, 'Done!', '')
        _set_done()

        return jsonify({
            'days_processed': n_days - skipped_days,
            'days_skipped':   skipped_days,
            'total_rows':     len(merged_out),
            'ct_mean':        ct_mean,
            'ds_mean':        ds_mean,
            'models_used':    sorted(models_used),
            'date_range':     f"{start_str} → {end_str}",
            'csv':            csv_str,
        })

    except Exception as e:
        _set_done()
        tb = traceback.format_exc()
        print(f"\n[ERROR /api/downscale/range]\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/downscale/<season>')
def downscale_season(season):
    try:
        if season not in APP_MODELS:
            return jsonify({'error': f'Model for {season} not found in models/ folder.'}), 404
        ct = APP_DATA['ct']
        ct_s = ct[ct['season'] == season].groupby(
            ['latitude','longitude'], as_index=False
        ).agg(co2=('co2','mean'), blh=('blh','mean'),
              temperature=('temperature','mean'),
              pressure=('pressure','mean'),
              specific_humidity=('specific_humidity','mean'))
        mb = run_downscaling(season, ct_s, APP_MODELS[season])
        ct_boxes = ct_s.copy().reset_index(drop=True)
        ct_boxes['lat_min'] = ct_boxes['latitude'] - 1.0
        ct_boxes['lat_max'] = ct_boxes['latitude'] + 1.0
        ct_boxes['lon_min'] = ct_boxes['longitude'] - 1.5
        ct_boxes['lon_max'] = ct_boxes['longitude'] + 1.5
        all_co2 = pd.concat([ct_boxes['co2'], mb['downscaled_co2']])
        zmin, zmax = tight_range(all_co2, nsigma=2.5)
        f_ct  = choropleth_box_map(ct_boxes, 'co2',
                    f'{season} — CarbonTracker Input (Coarse 2°×3°)',
                    colorscale='Plasma', zmin=zmin, zmax=zmax, height=460)
        mb_plot = mb.reset_index(drop=True)
        f_ds  = choropleth_box_map(mb_plot, 'downscaled_co2',
                    f'{season} — Downscaled Output (Fine 1°×1.5°)',
                    colorscale='Plasma', zmin=zmin, zmax=zmax, height=460)
        res_abs = float(mb_plot['predicted_residual'].abs().quantile(0.97))
        f_res = choropleth_box_map(mb_plot, 'predicted_residual',
                    f'{season} — Model Residual Correction (ppm)',
                    colorscale='RdBu_r', zmin=-res_abs, zmax=res_abs, zmid=0, height=460)
        stats = {
            'ct_mean':       round(float(ct_boxes['co2'].mean()), 3),
            'ds_mean':       round(float(mb['downscaled_co2'].mean()), 3),
            'residual_mean': round(float(mb['predicted_residual'].mean()), 3),
            'residual_std':  round(float(mb['predicted_residual'].std()), 3),
            'miniboxes':     len(mb),
            'ct_cells':      len(ct_boxes),
        }
        return jsonify({'fig_ct': f_ct.to_json(), 'fig_ds': f_ds.to_json(),
                        'fig_res': f_res.to_json(), 'stats': stats})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR /api/downscale/{season}]\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/download/<season>')
def download_season(season):
    if season not in APP_MODELS:
        return jsonify({'error': 'Model not found'}), 404
    ct = APP_DATA['ct']
    ct_s = ct[ct['season'] == season].groupby(
        ['latitude','longitude'], as_index=False
    ).agg(co2=('co2','mean'), blh=('blh','mean'),
          temperature=('temperature','mean'),
          pressure=('pressure','mean'),
          specific_humidity=('specific_humidity','mean'))
    mb = run_downscaling(season, ct_s, APP_MODELS[season])
    cols = ['latitude','longitude','ct_co2','downscaled_co2',
            'predicted_residual','blh','temperature','pressure',
            'specific_humidity','season']
    out = mb[[c for c in cols if c in mb.columns]]
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name=f'pavan_downscaled_{season}.csv')


# ── GEODATA ───────────────────────────────────────────────────────────────────
@app.route('/api/geodata/<dataset>')
def geodata(dataset):
    MAX_PTS = 900
    try:
        d = APP_DATA

        if dataset == 'ct':
            df = d['ct'].groupby(['latitude','longitude'], as_index=False)['co2'].mean()
            pts = df[['latitude','longitude','co2']].copy()
            seasons_out = {}
            per_s = max(1, MAX_PTS // 4)
            for s in SEASON_ORDER:
                sub = (d['ct'][d['ct']['season'] == s]
                       .groupby(['latitude','longitude'], as_index=False)['co2'].mean())
                if len(sub) > per_s:
                    sub = sub.sample(per_s, random_state=42)
                seasons_out[s] = sub[['latitude','longitude','co2']].to_dict('records')

        elif dataset in ('gosat', 'oco2', 'oco3'):
            df = d[dataset][['latitude','longitude','co2']].copy().dropna()
            if len(df) > MAX_PTS:
                df = df.sample(MAX_PTS, random_state=42)
            pts = df
            seasons_out = {}

        elif dataset == 'compare':
            n_each = MAX_PTS // 4
            frames = []
            for key, name in [('ct','CarbonTracker'), ('gosat','GOSAT'),
                               ('oco2','OCO-2'),       ('oco3','OCO-3')]:
                sub = d[key][['latitude','longitude','co2']].copy().dropna()
                if len(sub) > n_each:
                    sub = sub.sample(n_each, random_state=42)
                sub = sub.copy()
                sub['dataset'] = name
                frames.append(sub)
            pts = pd.concat(frames, ignore_index=True)
            seasons_out = {}

        else:
            return jsonify({'error': 'unknown dataset'}), 404

        if len(pts) > MAX_PTS and dataset != 'compare':
            pts = pts.sample(MAX_PTS, random_state=42)

        vmin = float(pts['co2'].quantile(0.02))
        vmax = float(pts['co2'].quantile(0.98))

        return jsonify({
            'points':  pts.to_dict('records'),
            'vmin':    vmin,
            'vmax':    vmax,
            'seasons': seasons_out,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR /api/geodata/{dataset}]\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


# ── STARTUP ───────────────────────────────────────────────────────────────────
print("Loading data...")
APP_DATA   = load_data()
print("Loading models...")
APP_MODELS = load_models()
print(f"Ready ✓  Models: {list(APP_MODELS.keys())}")

# Load optional attribution dataset
_attr_path = os.path.join(DATA_DIR, 'india_ct_co2_with_attribution.csv')
if os.path.exists(_attr_path):
    APP_ATTR = pd.read_csv(_attr_path)
    if 'month' not in APP_ATTR.columns and 'time' in APP_ATTR.columns:
        APP_ATTR['time']  = pd.to_datetime(APP_ATTR['time'], errors='coerce')
        APP_ATTR['month'] = APP_ATTR['time'].dt.month
    APP_ATTR['month'] = pd.to_numeric(APP_ATTR['month'], errors='coerce')
    if 'season' not in APP_ATTR.columns:
        APP_ATTR['season'] = APP_ATTR['month'].apply(get_season)
    print(f"Attribution data ✓  shape: {APP_ATTR.shape}")
else:
    APP_ATTR = None
    print("Attribution CSV not found — copy india_ct_co2_with_attribution.csv to data/ to enable attribution analysis")

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)