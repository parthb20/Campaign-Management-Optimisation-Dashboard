# dashboard_enhanced.py
# Requirements:
import gc
import os
import time
import pandas as pd
from functools import lru_cache
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State,dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
if os.environ.get('RENDER'):
    print("Running on Render - limiting data size")
    LIMIT_ROWS = 100000  # Process only first 190k rows
else:
    LIMIT_ROWS = None 
# -----------------------------
# CONFIG
# -----------------------------
KEYWORD_DATA_FILE = "Max Learning_5Dec202517_54_48_27Nov2025_03Dec2025.csv"
DOMAIN_DATA_FILE = "Domain Analysis_27Nov2025_03Dec2025.csv"
PORT = 8050

@lru_cache(maxsize=1)
def load_keyword_data():
    """Cache loaded data to prevent reloading"""
    try:
        df = pd.read_csv(KEYWORD_DATA_FILE, low_memory=False)
        print(f"Loaded {len(df)} keyword rows")
        return df
    except Exception as e:
        print(f"Error loading keyword data: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=1)
def load_domain_data():
    """Cache loaded domain data"""
    try:
        df = pd.read_csv(DOMAIN_DATA_FILE, low_memory=False)
        print(f"Loaded {len(df)} domain rows")
        return df
    except Exception as e:
        print(f"Error loading domain data: {e}")
        return pd.DataFrame()

COLORS = {
    'primary': '#00D9FF',
    'secondary': '#FF6B9D',
    'success': '#00F5A0',
    'warning': '#FFD93D',
    'danger': '#FF6B6B',
    'info': '#A78BFA',
    'background': '#0F1419',
    'card_bg': '#121419',
    'text': '#E5E7EB',
    'muted': '#9CA3AF'
}
PLOT_TEMPLATE = {
    'layout': {
        'font': {'color': 'white', 'family': 'Arial, sans-serif'},
        'xaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'color': 'white'},
        'yaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'color': 'white'},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(30,30,40,0.3)'
    }
}
# -----------------------------
# UTILS
# -----------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None
def split_multi(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    txt = str(cell)
    parts = re.split(r'[;,]\s*', txt)
    return [p.strip() for p in parts if p.strip()!='']
def cvr_color(val, metric='CVR'):
    if metric == 'CVR':
        if val >= 1.0:
            return COLORS['success']
        if val < 0.5:
            return COLORS['danger']
        return COLORS['warning']
    elif metric == 'CPA':
        if val <= 50:
            return COLORS['success']
        if val > 150:
            return COLORS['danger']
        return COLORS['warning']
    elif metric == 'ROAS':
        if val >= 3.0:
            return COLORS['success']
        if val < 1.5:
            return COLORS['danger']
        return COLORS['warning']
    return COLORS['info']
# -----------------------------
# LOAD DATA
# -----------------------------
df_keyword = load_keyword_data()
df_domain = load_domain_data()
df = df_keyword
# Column mapping
COL_CAMPAIGN_OBJ = find_col(df, ['[Learning] Campaign Objective', 'Campaign Objective'])
COL_ADVERTISER = find_col(df, ['Advertiser', 'Advertiser.'])
COL_CAMPAIGN_TYPE = find_col(df, ['Campaign Type'])
COL_CAMPAIGN = find_col(df, ['Campaign'])
COL_KEYWORD = find_col(df, ['Keyword', '.Keyword'])
COL_KEYWORD_CAT = find_col(df, ['Keyword Category', 'Keyword Category.'])
COL_QUERY_TYPE = find_col(df, ['Query_Type', 'Query_Type.'])
COL_EMOT = find_col(df, ['Emotional_Intent', 'Emotional Intent'])
COL_PHRASE = find_col(df, ['Individual_Words', 'Phrase Components'])
COL_WORDCOUNT = find_col(df, ['Number_of_Words', 'Word Count'])
COL_CHARCOUNT = find_col(df, ['Number_of_Characters', 'Character Count'])
COL_IS_QUESTION = find_col(df, ['Is_Question'])
COL_SPECIFICITY = find_col(df, ['Specificity_Score', 'Specificity Score'])
COL_URGENCY = find_col(df, ['Urgency_Level', 'Urgency Level'])
COL_NUMBER = find_col(df, ['Is_Number_Present', 'Number_Present'])
COL_NUMBER_POS = find_col(df, ['Position_of_Number'])
COL_IMPRESSIONS = find_col(df, ['Ad Impressions', 'Ad Impressions.'])
COL_CLICKS = find_col(df, ['Clicks', 'Clicks.'])
COL_CTR = find_col(df, ['CTR', 'CTR.'])
COL_CVR = find_col(df, ['CVR', 'CVR,'])
COL_CPA = find_col(df, ['CPA', 'CPA.'])
COL_ROAS = find_col(df, ['roas', 'roas.', 'ROAS'])
COL_MAX_COST = find_col(df, ['Max System Cost', 'Max System Cost.'])
COL_WEIGHTED_CONV = find_col(df, ['Weighted Conversion', 'Weighted Conversion.'])
if df.empty or COL_KEYWORD is None:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.layout = dbc.Container([
        html.H2("Dashboard - Data Load Error", style={'color': COLORS['text']}),
        html.P("Could not load data. Check DATA_FILE path.", style={'color': COLORS['muted']}),
        dbc.Tabs([
        dbc.Tab(label="📊 Keyword Analysis", tab_id="keyword-tab"),
        dbc.Tab(label="🌐 Domain Analysis", tab_id="domain-tab"),
    ], id="analysis-tabs", active_tab="keyword-tab", className="mb-3"),
    
    # Add this store to trigger initial load
    dcc.Store(id='trigger-load', data=0),
    
    html.Div(id="tab-content")
    ], fluid=True)
    if __name__ == '__main__':
        app.run_server(debug=True, port=PORT)
    raise SystemExit("Data not loaded")
# -----------------------------
# PREPROCESS
# -----------------------------
work = df.copy()
if LIMIT_ROWS:
    work = work.head(LIMIT_ROWS)
    print(f"Limited to {len(work)} rows for production")
rename_map = {}
if COL_CAMPAIGN_OBJ: rename_map[COL_CAMPAIGN_OBJ] = 'Campaign_Objective'
if COL_ADVERTISER: rename_map[COL_ADVERTISER] = 'Advertiser'
if COL_CAMPAIGN_TYPE: rename_map[COL_CAMPAIGN_TYPE] = 'Campaign_Type'
if COL_CAMPAIGN: rename_map[COL_CAMPAIGN] = 'Campaign'
if COL_KEYWORD: rename_map[COL_KEYWORD] = 'Keyword'
if COL_KEYWORD_CAT: rename_map[COL_KEYWORD_CAT] = 'Keyword_Category'
if COL_QUERY_TYPE: rename_map[COL_QUERY_TYPE] = 'Query_Type'
if COL_EMOT: rename_map[COL_EMOT] = 'Emotional_Intent'
if COL_PHRASE: rename_map[COL_PHRASE] = 'Phrase_Components'
if COL_WORDCOUNT: rename_map[COL_WORDCOUNT] = 'Word_Count'
if COL_CHARCOUNT: rename_map[COL_CHARCOUNT] = 'Character_Count'
if COL_IS_QUESTION: rename_map[COL_IS_QUESTION] = 'Is_Question'
if COL_SPECIFICITY: rename_map[COL_SPECIFICITY] = 'Specificity_Score'
if COL_URGENCY: rename_map[COL_URGENCY] = 'Urgency_Level'
if COL_NUMBER: rename_map[COL_NUMBER] = 'Is_Number_Present'
if COL_NUMBER_POS: rename_map[COL_NUMBER_POS] = 'Position_of_Number'
if COL_IMPRESSIONS: rename_map[COL_IMPRESSIONS] = 'Impressions'
if COL_CLICKS: rename_map[COL_CLICKS] = 'Clicks'
if COL_CTR: rename_map[COL_CTR] = 'CTR'
if COL_CVR: rename_map[COL_CVR] = 'CVR'
if COL_CPA: rename_map[COL_CPA] = 'CPA'
if COL_ROAS: rename_map[COL_ROAS] = 'ROAS'
if COL_MAX_COST: rename_map[COL_MAX_COST] = 'Max_System_Cost'
if COL_WEIGHTED_CONV: rename_map[COL_WEIGHTED_CONV] = 'Weighted_Conversion'
work = work.rename(columns=rename_map)

# DEBUG - Check what's in the data after loading
    
for col in ['Impressions', 'Clicks', 'CTR', 'CVR', 'CPA', 'ROAS', 'Max_System_Cost', 'Weighted_Conversion', 'Word_Count', 'Character_Count']:
    if col in work.columns:
        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0)
# Keep Specificity_Score as text - don't convert to numeric!
if 'Specificity_Score' in work.columns:
    work['Specificity_Score'] = work['Specificity_Score'].fillna('Unknown').astype(str)
    # Clean up any weird values like ','
    work['Specificity_Score'] = work['Specificity_Score'].replace(',', 'Unknown')

# Keep Urgency_Level as text - don't convert to numeric!
if 'Urgency_Level' in work.columns:
    work['Urgency_Level'] = work['Urgency_Level'].fillna('Unknown').astype(str)
    # Clean up any weird values
    work['Urgency_Level'] = work['Urgency_Level'].replace(',', 'Unknown')
for c in ['Campaign_Objective', 'Advertiser', 'Campaign_Type', 'Campaign', 'Keyword', 'Query_Type', 'Emotional_Intent', 'Phrase_Components', 'Keyword_Category', 'Impressions', 'Clicks', 'CTR', 'CVR', 'CPA', 'ROAS', 'Max_System_Cost', 'Weighted_Conversion', 'Is_Question', 'Is_Number_Present', 'Position_of_Number', 'Word_Count', 'Character_Count']:
    if c not in work.columns:
        work[c] = np.nan if c not in ['Impressions', 'Clicks'] else 0
# -----------------------------
# PREPROCESS DOMAIN DATA
# -----------------------------
work_domain = df_domain.copy()
COL_DOM_CAMPAIGN_OBJ = find_col(df_domain, ['[Learning] Campaign Objective', 'Campaign Objective'])
COL_DOM_ADVERTISER = find_col(df_domain, ['Advertiser'])
COL_DOM_CAMPAIGN_TYPE = find_col(df_domain, ['Campaign Type'])
COL_DOM_CAMPAIGN = find_col(df_domain, ['Campaign'])
COL_DOM_DOMAIN = find_col(df_domain, ['Domain'])
COL_DOM_CATEGORY = find_col(df_domain, ['Sprig Domain Category'])
COL_DOM_IMPRESSIONS = find_col(df_domain, ['Ad Impressions'])
COL_DOM_CLICKS = find_col(df_domain, ['Clicks'])
COL_DOM_CTR = find_col(df_domain, ['CTR'])
COL_DOM_CVR = find_col(df_domain, ['CVR'])
COL_DOM_CPA = find_col(df_domain, ['CPA'])
COL_DOM_ROAS = find_col(df_domain, ['roas', 'ROAS'])
COL_DOM_MAX_COST = find_col(df_domain, ['Max System Cost'])
COL_DOM_WEIGHTED_CONV = find_col(df_domain, ['Weighted Conversion'])
dom_rename_map = {}
if COL_DOM_CAMPAIGN_OBJ: dom_rename_map[COL_DOM_CAMPAIGN_OBJ] = 'Campaign_Objective'
if COL_DOM_ADVERTISER: dom_rename_map[COL_DOM_ADVERTISER] = 'Advertiser'
if COL_DOM_CAMPAIGN_TYPE: dom_rename_map[COL_DOM_CAMPAIGN_TYPE] = 'Campaign_Type'
if COL_DOM_CAMPAIGN: dom_rename_map[COL_DOM_CAMPAIGN] = 'Campaign'
if COL_DOM_DOMAIN: dom_rename_map[COL_DOM_DOMAIN] = 'Domain'
if COL_DOM_CATEGORY: dom_rename_map[COL_DOM_CATEGORY] = 'Domain_Category'
if COL_DOM_IMPRESSIONS: dom_rename_map[COL_DOM_IMPRESSIONS] = 'Impressions'
if COL_DOM_CLICKS: dom_rename_map[COL_DOM_CLICKS] = 'Clicks'
if COL_DOM_CTR: dom_rename_map[COL_DOM_CTR] = 'CTR'
if COL_DOM_CVR: dom_rename_map[COL_DOM_CVR] = 'CVR'
if COL_DOM_CPA: dom_rename_map[COL_DOM_CPA] = 'CPA'
if COL_DOM_ROAS: dom_rename_map[COL_DOM_ROAS] = 'ROAS'
if COL_DOM_MAX_COST: dom_rename_map[COL_DOM_MAX_COST] = 'Max_System_Cost'
if COL_DOM_WEIGHTED_CONV: dom_rename_map[COL_DOM_WEIGHTED_CONV] = 'Weighted_Conversion'
work_domain = work_domain.rename(columns=dom_rename_map)
for col in ['Impressions', 'Clicks', 'CTR', 'CVR', 'CPA', 'ROAS', 'Max_System_Cost', 'Weighted_Conversion']:
    if col in work_domain.columns:
        work_domain[col] = pd.to_numeric(work_domain[col], errors='coerce').fillna(0)
for c in ['Campaign_Objective', 'Advertiser', 'Campaign_Type', 'Campaign', 'Domain', 'Domain_Category', 'Impressions', 'Clicks', 'CTR', 'CVR', 'CPA', 'ROAS', 'Max_System_Cost', 'Weighted_Conversion']:
    if c not in work_domain.columns:
        work_domain[c] = np.nan if c not in ['Impressions', 'Clicks'] else 0
# -----------------------------
# AGGREGATION FUNCTIONS
# -----------------------------
def weighted_ctr(group):
    if group['Impressions'].sum() == 0:
        return 0
    return (group['CTR'] * group['Impressions']).sum() / group['Impressions'].sum()
def weighted_cvr(group):
    if group['Clicks'].sum() == 0:
        return 0
    return (group['CVR'] * group['Clicks']).sum() / group['Clicks'].sum()
def weighted_cpa(group):
    if group['Weighted_Conversion'].sum() == 0:
        return 0
    return (group['CPA'] * group['Weighted_Conversion']).sum() / group['Weighted_Conversion'].sum()
def weighted_roas(group):
    if group['Max_System_Cost'].sum() == 0:
        return 0
    return (group['ROAS'] * group['Max_System_Cost']).sum() / group['Max_System_Cost'].sum()
# -----------------------------
# DASH APP
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Campaign Analytics Dashboard"
app.config.suppress_callback_exceptions = True
server = app.server
app.index_string = '''

<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .big-number { font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }
            .small-muted { font-size: 0.85rem; color: #9CA3AF; font-weight: 500; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    html.H1("✅ CPA Campaign Management Dashboard", className="text-center mt-4 mb-2", style={'color':COLORS['primary'], 'fontWeight': '700'}),
    dbc.Progress(id="loading-progress", value=0, striped=True, animated=True, 
                 color="primary", className="mb-3", style={'height': '5px'}),
    dcc.Store(id='loading-store', data={'loaded': False}),

    
    # Subtitle/description
     dbc.Row([
        dbc.Col([
            html.P("Comprehensive analysis of keyword and domain performance across campaigns. Use filters to drill down into specific segments.", 
                   className="text-center mb-2", 
                   style={'color':COLORS['muted'], 'fontSize': '1.1rem'}),
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-external-link-alt me-2"),  # Icon (if you have FontAwesome)
                    "Data Link"
                ], 
                href="https://max.analytics.mn/reports/analyse?hash=c579974824d8cec5679d31ca98d96dcf",
                target="_blank",
                color="primary",
                size="sm",
                className="mb-3")
            ], className="text-center")
        ], width=12)
    ]),
    
    dbc.Card([
        dbc.CardBody([
            html.H5("🔍 Global Filters", className="mb-3", style={'color': COLORS['primary']}),
            dbc.Row([
                dbc.Col([
                    html.Label("Campaign Objective", style={'color':COLORS['muted'],'fontWeight':'600'}),
                    dcc.Dropdown(id='objective-dropdown', clearable=True, placeholder="Select Objective...")
                ], md=3),
                dbc.Col([
                    html.Label("Advertiser", style={'color':COLORS['muted'],'fontWeight':'600'}),
                    dcc.Dropdown(id='advertiser-dropdown', clearable=True, placeholder="Select Advertiser...")
                ], md=3),
                dbc.Col([
                    html.Label("Campaign Type", style={'color':COLORS['muted'],'fontWeight':'600'}),
                    dcc.Dropdown(id='campaign-type-dropdown', clearable=True, placeholder="Select Campaign Type...")
                ], md=3),
                dbc.Col([
                    html.Label("Campaign", style={'color':COLORS['muted'],'fontWeight':'600'}),
                    dcc.Dropdown(id='campaign-dropdown', clearable=True, placeholder="Select Campaign...")
                ], md=3),
            ], className='g-2')
        ])
    ], className='mb-4'),
 
     
     
    dbc.Tabs([
        dbc.Tab(label="Keyword Analysis", tab_id="keyword-tab"),
        dbc.Tab(label="Domain Analysis", tab_id="domain-tab"),
    ], id="analysis-tabs", active_tab="keyword-tab", className="mb-3"),
    html.Div(id="tab-content")
], fluid=True)
@app.callback(
    Output("tab-content", "children"),
    Input("analysis-tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "keyword-tab":
        return html.Div([
            html.Div(id='stats'),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("💡 Word Performance: CTR/CVR", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Individual word analysis showing click-through and conversion rates. Larger bubbles = more clicks.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']})
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='treemap_ctr_cvr', config={'displayModeBar': False}), type='default')])
                ]), md=12),
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("💰 Word Performance: CPA/ROAS", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Cost efficiency and return on ad spend by word. ", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']})
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='treemap_cpa_roas', config={'displayModeBar': False}), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("📂 Keyword Category Performance", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Performance metrics by keyword category.", className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-keyword-category-btn", color="primary", size="sm", className="mt-2")
                        ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='keyword_category_analysis'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-keyword-category"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("📊 Category Performance Overview", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("All metrics normalized to 0-100 scale for easy comparison across query types.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']})
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='category_overview', config={'displayModeBar': False}), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("🎯 Emotional Intent: CTR vs CVR", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("How different emotional intents perform. Hover to see top performing keywords.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-emotion-ctr-cvr-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='emotion_bubble_ctr_cvr'), type='default')])
                ]), md=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("💵 Emotional Intent: ROAS vs CPA", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Cost efficiency by emotional intent. Top-left quadrant = ideal.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-emotion-roas-cpa-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='emotion_bubble_roas_cpa'), type='default')])
                ]), md=6)
            ], className='mb-4'),
            dcc.Download(id="download-emotion-ctr-cvr"),
            dcc.Download(id="download-emotion-roas-cpa"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("📏 Character Length Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("How keyword length impacts performance. Hover for top keywords.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-char-analysis-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='char_analysis'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-char-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("🎯 Specificity Score Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Performance by keyword specificity level (Low/Medium/High).", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-specificity-analysis-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='specificity_analysis'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-specificity-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("⚡ Urgency Level Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Performance by urgency level (Low/Medium/High).", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-urgency-analysis-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='urgency_analysis'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-urgency-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("📝 Word Count Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("How the number of words in a keyword affects performance.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-word-count-analysis-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='word_count_analysis'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-word-count-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔢 Number Present Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Keywords with vs without numbers.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-number-analysis-btn", color="primary", size="sm", className="mt-2")]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='number_analysis'), type='default')])
                    ]), md=12)], className='mb-4'),
            dcc.Download(id="download-number-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("📍 Position of Number Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Where numbers appear in keywords.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-number-position-analysis-btn", color="primary", size="sm", className="mt-2")]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='number_position_analysis'), type='default')])]), md=12)
                ], className='mb-4'),
            dcc.Download(id="download-number-position-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader([
                        html.H5("❓ Question Type Analysis", className="mb-1", style={'color': COLORS['primary']}),
                        html.P("Question-based vs statement-based keywords.", 
                               className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                        dbc.Button("Download Data", id="download-question-analysis-btn", color="primary", size="sm", className="mt-2")
                    ]),
                    dbc.CardBody([dcc.Loading(dcc.Graph(id='question_analysis'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-question-analysis"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Div([
                        "Data Preview (Top 30 Keywords by Clicks)",
                        dbc.Button("Download CSV", id="download-btn", color="primary", size="sm", style={'float':'right'})
                    ])),
                    dbc.CardBody([dcc.Loading(html.Div(id='table_preview'), type='default')])
                ]), md=12)
            ], className='mb-4'),
            dcc.Download(id="download-data"),
        ])
    
    elif active_tab == "domain-tab":
     return html.Div([
        html.Div(id='domain-stats'),
        # ✅ REMOVED THE DOMAIN FILTER CARD - Using global filters only
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("🌐 Domain Performance: CTR/CVR", className="mb-1", style={'color': COLORS['primary']}),
                    html.P("Domain-level analysis. Bubble size represents total clicks.", 
                           className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                    dbc.Button("Download Data", id="download-domain-treemap-ctr-cvr-btn", color="primary", size="sm", className="mt-2")
                ]),
                dbc.CardBody([dcc.Loading(dcc.Graph(id='domain_treemap_ctr_cvr', config={'displayModeBar': False}), type='default')])
            ]), md=12),
        ], className='mb-4'),
        dcc.Download(id="download-domain-treemap-ctr-cvr-data"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("💰 Domain Performance: CPA/ROAS", className="mb-1", style={'color': COLORS['primary']}),
                    html.P("Cost efficiency and return on ad spend by domain.", 
                           className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']}),
                    dbc.Button("Download Data", id="download-domain-treemap-cpa-roas-btn", color="primary", size="sm", className="mt-2")
                ]),
                dbc.CardBody([dcc.Loading(dcc.Graph(id='domain_treemap_cpa_roas', config={'displayModeBar': False}), type='default')])
            ]), md=12)
        ], className='mb-4'),
        dcc.Download(id="download-domain-treemap-cpa-roas-data"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("📈 Domain Category Performance Overview", className="mb-1", style={'color': COLORS['primary']}),
                    html.P("All metrics normalized to 0-100 scale for easy comparison.", 
                           className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']})
                ]),
                dbc.CardBody([dcc.Loading(dcc.Graph(id='domain_category_overview', config={'displayModeBar': False}), type='default')])
            ]), md=12)
        ], className='mb-4'),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("🎯 Domain Category: CTR vs CVR", className="mb-1", style={'color': COLORS['primary']}),
                    html.P("Performance by domain category.", 
                           className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']})
                ]),
                dbc.CardBody([dcc.Loading(dcc.Graph(id='domain_category_ctr_cvr'), type='default')])
            ]), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("💰 Domain Category: ROAS vs CPA", className="mb-1", style={'color': COLORS['primary']}),
                    html.P("Cost efficiency by domain category.", 
                           className="mb-0", style={'fontSize': '0.9rem', 'color': COLORS['muted']})
                ]),
                dbc.CardBody([dcc.Loading(dcc.Graph(id='domain_category_roas_cpa'), type='default')])
            ]), md=6)
        ], className='mb-4'),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Div([
                    "Domain Data Preview (Top 30 by Clicks)",
                    dbc.Button("Download CSV", id="download-domain-btn", color="primary", size="sm", style={'float':'right'})
                ])),
                dbc.CardBody([dcc.Loading(html.Div(id='domain_table_preview'), type='default')])
            ]), md=12)
        ], className='mb-4'),
        dcc.Download(id="download-domain-data"),
    ])
# KEYWORD CALLBACKS - WITH prevent_initial_call=True ADDED
@app.callback(
    Output('objective-dropdown','options'),
    Output('objective-dropdown','value'),
    Input('objective-dropdown','id')
)
def init_objective(_):
    opts = sorted(work['Campaign_Objective'].dropna().astype(str).unique())
    return [{'label': o, 'value': o} for o in opts], None
@app.callback(
    Output('advertiser-dropdown','options'),
    Input('objective-dropdown','value'),
    prevent_initial_call=True
)
def load_advertisers(obj):
    data = work if not obj else work[work['Campaign_Objective'] == obj]
    opts = sorted(data['Advertiser'].dropna().astype(str).unique())
    return [{'label': a, 'value': a} for a in opts]
@app.callback(
    Output('campaign-type-dropdown','options'),
    Input('advertiser-dropdown','value'),
    prevent_initial_call=True
)
def load_campaign_types(adv):
    data = work if not adv else work[work['Advertiser']==adv]
    opts = sorted(data['Campaign_Type'].dropna().astype(str).unique())
    return [{'label': c, 'value': c} for c in opts]
@app.callback(
    Output('campaign-dropdown','options'),
    Input('campaign-type-dropdown','value'),
    prevent_initial_call=True
)
def load_campaigns(obj, adv, ctype):
    data = work.copy()
    if obj: data = data[data['Campaign_Objective'].astype(str).str.strip().str.lower() == str(obj).strip().lower()]
    if adv: data = data[data['Advertiser'].astype(str).str.strip().str.lower() == str(adv).strip().lower()]
    if ctype: data = data[data['Campaign_Type'].astype(str).str.strip().str.lower() == str(ctype).strip().lower()]
    opts = sorted(data['Campaign'].dropna().astype(str).unique())
    return [{'label': c, 'value': c} for c in opts]
# MAIN KEYWORD DASHBOARD - WITH prevent_initial_call=True ADDED
@app.callback(
    Output('stats','children'),
    Output('treemap_ctr_cvr','figure'),
    Output('treemap_cpa_roas','figure'),
    Output('category_overview','figure'),
    Output('keyword_category_analysis','figure'),
    Output('emotion_bubble_ctr_cvr','figure'),
    Output('emotion_bubble_roas_cpa','figure'),
    Output('char_analysis','figure'),
    Output('specificity_analysis','figure'),
    Output('urgency_analysis','figure'),
    Output('word_count_analysis','figure'),
    Output('number_analysis','figure'),
    Output('number_position_analysis','figure'),
    Output('question_analysis','figure'),
    Output('table_preview','children'),
    Input('objective-dropdown','value'),
    Input('advertiser-dropdown','value'),
    Input('campaign-type-dropdown','value'),
    Input('campaign-dropdown','value'),
    Input('analysis-tabs', 'active_tab'),
    prevent_initial_call=True
)
def update_dashboard(obj, adv, ctype, camp, active_tab):
    if active_tab != "keyword-tab":
        raise PreventUpdate
    
    d = work.copy()
    #if obj:
    #    d = d[d['Campaign_Objective'] == obj]
    #if adv:
    #    d = d[d['Advertiser'] == adv]
    #if ctype:
    #    d = d[d['Campaign_Type'] == ctype]
    #if camp:
    #    d = d[d['Campaign'] == camp]            
    d = work.copy()
    if obj: d = d[d['Campaign_Objective'].astype(str).str.strip().str.lower() == str(obj).strip().lower()]
    if adv: d = d[d['Advertiser'].astype(str).str.strip().str.lower() == str(adv).strip().lower()]
    if ctype: d = d[d['Campaign_Type'].astype(str).str.strip().str.lower() == str(ctype).strip().lower()]
    if camp: d = d[d['Campaign'].astype(str).str.strip().str.lower() == str(camp).strip().lower()]
    if d.shape[0] == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'), xaxis=dict(color='white'),yaxis=dict(color='white'))
        return (html.Div("No data"), empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No data"))
    total_clicks = int(d['Clicks'].sum())
    total_impressions = int(d['Impressions'].sum())
    avg_ctr = weighted_ctr(d)
    avg_cvr = weighted_cvr(d)
    avg_cpa = weighted_cpa(d)
    avg_roas = weighted_roas(d)
    # Stats row - 6 metrics
    stat_row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Total Impressions", className='small-muted'),
            html.Div(f"{total_impressions:,}", className='big-number')
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Total Clicks", className='small-muted'),
            html.Div(f"{total_clicks:,}", className='big-number', style={'color':COLORS['secondary']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg CTR (%)", className='small-muted'),
            html.Div(f"{avg_ctr:.2f}", className='big-number', style={'color':COLORS['info']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg CVR (%)", className='small-muted'),
            html.Div(f"{avg_cvr:.2f}", className='big-number', style={'color':COLORS['success']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg CPA", className='small-muted'),
            html.Div(f"${avg_cpa:.2f}", className='big-number', style={'color':COLORS['warning']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg ROAS", className='small-muted'),
            html.Div(f"{avg_roas:.2f}x", className='big-number', style={'color':COLORS['primary']})
        ])), md=2),
    ], className='mb-3')
    # Prepare word-level aggregated df
    rows = []
    for _, r in d.iterrows():
        pcs = split_multi(r['Phrase_Components'])
        if not pcs:
            tokens = re.findall(r"[A-Za-z0-9']+", str(r['Keyword']).lower())
            pcs = [t for t in tokens if len(t)>1][:5]
        for w in set(pcs):
            rows.append({
                'word': w,
                'Clicks': r['Clicks'],
                'Impressions': r['Impressions'],
                'CTR': r['CTR'],
                'CVR': r['CVR'],
                'CPA': r['CPA'],
                'ROAS': r['ROAS'],
                'Max_System_Cost': r['Max_System_Cost'],
                'Weighted_Conversion': r['Weighted_Conversion']
            })
    if rows:
        word_df = pd.DataFrame(rows)
        word_agg = word_df.groupby('word').apply(lambda g: pd.Series({
            'Clicks': g['Clicks'].sum(),
            'Impressions': g['Impressions'].sum(),
            'CTR': weighted_ctr(g),
            'CVR': weighted_cvr(g),
            'CPA': weighted_cpa(g),
            'ROAS': weighted_roas(g)
        })).reset_index()
        word_agg = word_agg.sort_values('Clicks', ascending=False).head(30)
    else:
        word_agg = pd.DataFrame(columns=['word','Clicks','CTR','CVR','CPA','ROAS'])
    # 1. TREEMAP CTR/CVR
    if not word_agg.empty:
        text_labels = word_agg.apply(
            lambda r: f"<b>{r['word']}</b><br>CTR: {r['CTR']:.1f}% | CVR: {r['CVR']:.1f}%", axis=1
        )
        treemap_ctr_cvr = go.Figure(go.Treemap(
            labels=word_agg['word'],
            parents=[''] * len(word_agg),
            values=word_agg['Clicks'],
            text=text_labels,
            textposition="middle center",
            marker=dict(
                colors=word_agg['CVR'].apply(lambda x: cvr_color(x, 'CVR')),
                line=dict(width=2, color='#1a1a1a')
            ),
            textfont=dict(size=12, color='white')
        ))
        treemap_ctr_cvr.update_layout(
            
            margin=dict(l=5,r=5,t=5,b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,font=dict(color='white'), xaxis=dict(color='white'),yaxis=dict(color='white')
        )
    else:
        treemap_ctr_cvr = go.Figure()
    # 2. TREEMAP CPA/ROAS
    if not word_agg.empty:
        text_labels = word_agg.apply(
            lambda r: f"<b>{r['word']}</b><br>CPA: ${r['CPA']:.1f} | ROAS: {r['ROAS']:.1f}x", axis=1
        )
        treemap_cpa_roas = go.Figure(go.Treemap(
            labels=word_agg['word'],
            parents=[''] * len(word_agg),
            values=word_agg['Clicks'],
            text=text_labels,
            textposition="middle center",
            marker=dict(
                colors=word_agg['CPA'].apply(lambda x: cvr_color(x, 'CPA')),
                line=dict(width=2, color='#1a1a1a')
            ),
            textfont=dict(size=12, color='white')
        ))
        treemap_cpa_roas.update_layout(
            margin=dict(l=5,r=5,t=5,b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,font=dict(color='white'), xaxis=dict(color='white'),yaxis=dict(color='white')
        )
    else:
        treemap_cpa_roas = go.Figure()
    # 3. WORD BUBBLE CTR/CVR
    # 5. CATEGORY OVERVIEW - All 4 metrics normalized
    # KEYWORD CATEGORY ANALYSIS - Add after category_overview
    
    
    
    if 'Query_Type' in d.columns and d['Query_Type'].notna().any():
    # Get top 3 keywords per category
        cat_top_kw = {}
        for cat in d['Query_Type'].dropna().unique():
            cat_data = d[d['Query_Type'] == cat]
            if len(cat_data) > 0 and 'Keyword' in cat_data.columns:
                top_kws = cat_data.nlargest(3, 'Clicks')
                cat_top_kw[cat] = '<br>'.join([f"• {kw} ({int(c)} clicks)" 
                                           for kw, c in zip(top_kws['Keyword'], top_kws['Clicks'])])
            else:
                cat_top_kw[cat] = 'N/A'

    
        cat_grp = d.groupby('Query_Type').apply(lambda g: pd.Series({
                'Clicks': g['Clicks'].sum(),
                'CTR': weighted_ctr(g),
                'CVR': weighted_cvr(g),
                'CPA': weighted_cpa(g),
                'ROAS': weighted_roas(g)
    })).reset_index().sort_values('Clicks', ascending=False).head(10)
    
        if not cat_grp.empty:
        # Normalize all 5 to 0-100
                    for col in ['Clicks', 'CTR', 'CVR', 'CPA', 'ROAS']:
                       min_val = cat_grp[col].min()
                       max_val = cat_grp[col].max()
                       cat_grp[f'{col}_norm'] = (cat_grp[col] - min_val) / (max_val - min_val + 1e-9) * 100
        
        # Add top keywords column
                    cat_grp['top_keywords'] = cat_grp['Query_Type'].map(cat_top_kw)
        
                    fig_cat = go.Figure()
                    fig_cat.add_trace(go.Bar(
                        x=cat_grp['Query_Type'], 
                        y=cat_grp['Clicks_norm'],
                        name='Clicks', 
                        marker_color=COLORS['secondary'],
                        customdata=list(zip(cat_grp['Clicks'], cat_grp['top_keywords'])),
                        hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual Clicks: %{customdata[0]:,}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
                    fig_cat.add_trace(go.Bar(
                        x=cat_grp['Query_Type'], 
                        y=cat_grp['CTR_norm'], 
                        name='CTR', 
                        marker_color=COLORS['info'],
                        customdata=list(zip(cat_grp['CTR'], cat_grp['top_keywords'])),
                        hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual CTR: %{customdata[0]:.2f}%</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'
                        ))
                    fig_cat.add_trace(go.Bar(
                        x=cat_grp['Query_Type'], 
                        y=cat_grp['CVR_norm'], 
                        name='CVR', 
                        marker_color=COLORS['success'],
                        customdata=list(zip(cat_grp['CVR'], cat_grp['top_keywords'])),
                        hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual CVR: %{customdata[0]:.2f}%</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
                    fig_cat.add_trace(go.Bar(
                        x=cat_grp['Query_Type'], 
                        y=cat_grp['CPA_norm'], 
                        name='CPA', 
                        marker_color=COLORS['warning'],
                        customdata=list(zip(cat_grp['CPA'], cat_grp['top_keywords'])),
                        hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual CPA: $%{customdata[0]:.2f}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'
                        ))
                    fig_cat.add_trace(go.Bar(
                        x=cat_grp['Query_Type'], 
                        y=cat_grp['ROAS_norm'], 
                        name='ROAS', 
                        marker_color=COLORS['primary'],
                        customdata=list(zip(cat_grp['ROAS'], cat_grp['top_keywords'])),
                        hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual ROAS: %{customdata[0]:.2f}x</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'
                        ))
                    fig_cat.update_layout(
                        barmode='group', height=500,
                        xaxis_tickangle=-20,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        yaxis_title="Normalized Score (0-100)",font=dict(color='white'), xaxis=dict(color='white'),yaxis=dict(color='white')
            )
        else:
            fig_cat = go.Figure()
            print("cat_grp is empty, creating empty fig_cat")
    else:
        fig_cat = go.Figure()
        print("Query_Type column missing or all null")
        
    if 'Keyword_Category' in d.columns and d['Keyword_Category'].notna().any():
        kw_cat_top = {}
        for cat in d['Keyword_Category'].dropna().unique():
            cat_data = d[d['Keyword_Category'] == cat]
            if len(cat_data) > 0:
                top_kws = cat_data.nlargest(3, 'Clicks')
                kw_cat_top[cat] = '<br>'.join([f"• {kw} ({int(c)} clicks)" 
                                          for kw, c in zip(top_kws['Keyword'], top_kws['Clicks'])])
    
        kw_cat_grp = d.groupby('Keyword_Category').apply(lambda g: pd.Series({
        'Clicks': g['Clicks'].sum(),
        'CTR': weighted_ctr(g),
        'CVR': weighted_cvr(g),
        'CPA': weighted_cpa(g),
        'ROAS': weighted_roas(g)
    })).reset_index().sort_values('Clicks', ascending=False).head(10)
    
        if not kw_cat_grp.empty:
            for col in ['Clicks', 'CTR', 'CVR', 'CPA', 'ROAS']:
                min_val = kw_cat_grp[col].min()
                max_val = kw_cat_grp[col].max()
                kw_cat_grp[f'{col}_norm'] = (kw_cat_grp[col] - min_val) / (max_val - min_val + 1e-9) * 100
        
            kw_cat_grp['top_keywords'] = kw_cat_grp['Keyword_Category'].map(kw_cat_top)
            keyword_category_fig = go.Figure()
            keyword_category_fig.add_trace(go.Bar(
                x=kw_cat_grp['Keyword_Category'], y=kw_cat_grp['Clicks_norm'],
                name='Clicks', marker_color=COLORS['secondary'],
                customdata=list(zip(kw_cat_grp['Clicks'], kw_cat_grp['top_keywords'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Clicks: %{customdata[0]:,}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
            keyword_category_fig.add_trace(go.Bar(
                x=kw_cat_grp['Keyword_Category'], y=kw_cat_grp['CTR_norm'],
                name='CTR', marker_color=COLORS['info'],
                customdata=list(zip(kw_cat_grp['CTR'], kw_cat_grp['top_keywords'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>CTR: %{customdata[0]:.2f}%</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
            keyword_category_fig.add_trace(go.Bar(
                x=kw_cat_grp['Keyword_Category'], y=kw_cat_grp['CVR_norm'],
                name='CVR', marker_color=COLORS['success'],
                customdata=list(zip(kw_cat_grp['CVR'], kw_cat_grp['top_keywords'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>CVR: %{customdata[0]:.2f}%</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
            keyword_category_fig.add_trace(go.Bar(
                x=kw_cat_grp['Keyword_Category'], y=kw_cat_grp['CPA_norm'],
                name='CPA', marker_color=COLORS['warning'],
                customdata=list(zip(kw_cat_grp['CPA'], kw_cat_grp['top_keywords'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>CPA: $%{customdata[0]:.2f}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
            keyword_category_fig.add_trace(go.Bar(
                x=kw_cat_grp['Keyword_Category'], y=kw_cat_grp['ROAS_norm'],
                name='ROAS', marker_color=COLORS['primary'],
                customdata=list(zip(kw_cat_grp['ROAS'], kw_cat_grp['top_keywords'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>ROAS: %{customdata[0]:.2f}x</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>'))
            keyword_category_fig.update_layout(
                barmode='group', height=500,
                xaxis_tickangle=-45,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Normalized Score (0-100)",
                font=dict(color='white'), xaxis=dict(color='white'), yaxis=dict(color='white')
        )
        else:
            keyword_category_fig = go.Figure()
                
    else:
            keyword_category_fig = go.Figure()    
    # 6. EMOTION BUBBLE CTR/CVR
    # 6. EMOTION BUBBLE CTR/CVR
    # 6. EMOTION BUBBLE CTR/CVR with top keywords
    emo_rows = []
    for _, row in d.iterrows():
        emo_str = str(row['Emotional_Intent'])
        if pd.isna(row['Emotional_Intent']) or emo_str.lower() in ['', 'nan', 'none']:
          emos = ['neutral']
        else:
          emos = [e.strip().lower() for e in re.split(r'[;,]\s*', emo_str) if e.strip()]
        if not emos:
          emos = ['neutral']
    
        for e in emos:
            emo_rows.append({
            'emotion': e,
            'Keyword': row['Keyword'],
            'Clicks': row['Clicks'],
            'Impressions': row['Impressions'],
            'CTR': row['CTR'],
            'CVR': row['CVR'],
            'CPA': row['CPA'],
            'ROAS': row['ROAS'],
            'Max_System_Cost': row['Max_System_Cost'],
            'Weighted_Conversion': row['Weighted_Conversion']
        })

    emo_ctr_cvr = go.Figure()
    if emo_rows:
        emo_df = pd.DataFrame(emo_rows)
    
        emo_top_keywords = emo_df.groupby('emotion').apply(
        lambda g: '<br>'.join([f"• {kw} ({int(c)} clicks)" 
                               for kw, c in zip(g.nlargest(3, 'Clicks')['Keyword'], 
                                              g.nlargest(3, 'Clicks')['Clicks'])])
        ).to_dict()
    
        emo_agg = emo_df.groupby('emotion').agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR': 'mean',  # Approximate, adjust if needed
            'CVR': 'mean'
            }).reset_index()
    
        max_clicks_emo = emo_agg['Clicks'].max()
        palette = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
               COLORS['info'], COLORS['warning'], COLORS['danger']]
        cmap = {emo: palette[i % len(palette)] for i, emo in enumerate(emo_agg['emotion'])}
    
        for _, r in emo_agg.iterrows():
            size = 40 + (r['Clicks'] / max_clicks_emo) * 100
            top_kw = emo_top_keywords.get(r['emotion'], 'N/A')
            emo_ctr_cvr.add_trace(go.Scatter(
            x=[r['CTR']], y=[r['CVR']],
            mode='markers+text',
            text=[r['emotion']],
            textposition='middle center',
            textfont=dict(size=12, color='white', weight='bold'),
            marker=dict(size=size, color=cmap[r['emotion']], opacity=0.8,
                       line=dict(width=3, color='white')),
            hovertemplate=f"<b>{r['emotion']}</b><br>" +
                         f"CTR: {r['CTR']:.2f}%<br>" +
                         f"CVR: {r['CVR']:.2f}%<br>" +
                         f"<b>Clicks: {int(r['Clicks']):,}</b><br>" +
                         f"<b>Top Keywords:</b><br>{top_kw}<extra></extra>",
            showlegend=False
        ))
    
        emo_ctr_cvr.update_layout(
        xaxis_title="CTR (%)", yaxis_title="CVR (%)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.3)',
        height=500, font=dict(color='white'),
        xaxis=dict(color='white'), yaxis=dict(color='white')
    )
    # 7. EMOTION BUBBLE ROAS/CPA
    emo_roas_cpa = go.Figure()
    if emo_rows:
     for _, r in emo_agg.iterrows():
        size = 40 + (r['Clicks'] / max_clicks_emo) * 100
        top_kw = emo_top_keywords.get(r['emotion'], 'N/A')
        emo_roas_cpa.add_trace(go.Scatter(
            x=[r['ROAS']], y=[r['CPA']],
            mode='markers+text',
            text=[r['emotion']],
            textposition='middle center',
            textfont=dict(size=12, color='white', weight='bold'),
            marker=dict(size=size, color=cmap[r['emotion']], opacity=0.8,
                       line=dict(width=3, color='white')),
            hovertemplate=f"<b>{r['emotion']}</b><br>" +
                         f"ROAS: {r['ROAS']:.2f}x<br>" +
                         f"CPA: ${r['CPA']:.2f}<br>" +
                         f"<b>Clicks: {int(r['Clicks']):,}</b><br>" +
                         f"<b>Top Keywords:</b><br>{top_kw}<extra></extra>",
            showlegend=False
        ))
    
    emo_roas_cpa.update_layout(
        xaxis_title="ROAS", yaxis_title="CPA ($)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.3)',
        height=500, font=dict(color='white'),
        xaxis=dict(color='white'), yaxis=dict(color='white')
    )
    # 8. CHARACTER LENGTH - aggregated by character count (1 bubble per char length)
    # 8. CHARACTER LENGTH - aggregated by character count with top 3 keywords
    char_grp_list = []
    for char_count in d['Character_Count'].dropna().unique():
       char_data = d[d['Character_Count'] == char_count]
       top_keywords = char_data.nlargest(3, 'Clicks')['Keyword'].tolist()
       top_clicks = char_data.nlargest(3, 'Clicks')['Clicks'].tolist()
       keywords_text = '<br>'.join([f"  • {kw} ({int(c)} clicks)" for kw, c in zip(top_keywords, top_clicks)])
       char_grp_list.append({
        'Character_Count': char_count,
        'Clicks': char_data['Clicks'].sum(),
        'CTR': weighted_ctr(char_data),
        'CVR': weighted_cvr(char_data),
        'CPA': weighted_cpa(char_data),
        'ROAS': weighted_roas(char_data),
        'top_keywords': keywords_text
    })
    char_grp = pd.DataFrame(char_grp_list)
    char_fig = make_subplots(rows=2, cols=2,
                              subplot_titles=("CTR by Character Length", "CVR by Character Length",
                                            "ROAS by Character Length", "CPA by Character Length"),
                              vertical_spacing=0.12, horizontal_spacing=0.1)
    max_char_clicks = char_grp['Clicks'].max()
    char_grp['size'] = 10 + (char_grp['Clicks'] / max_char_clicks) * 40
    char_fig.add_trace(go.Scatter(x=char_grp['Character_Count'], y=char_grp['CTR'],
                                  mode='markers', marker=dict(size=char_grp['size'], color=COLORS['info'], opacity=0.7),
                                  customdata=char_grp['top_keywords'],
                                  hovertemplate='<b>Char Count: %{x}</b><br>CTR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
                                  name='CTR'), row=1, col=1)
    char_fig.add_trace(go.Scatter(x=char_grp['Character_Count'], y=char_grp['CVR'],
                                   mode='markers', marker=dict(size=char_grp['size'], color=COLORS['success'], opacity=0.7),
                                   customdata=char_grp['top_keywords'],
                                   hovertemplate='<b>Char Count: %{x}</b><br>CVR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
                                   name='CVR'), row=1, col=2)
    char_fig.add_trace(go.Scatter(x=char_grp['Character_Count'], y=char_grp['ROAS'],
                                   mode='markers', marker=dict(size=char_grp['size'], color=COLORS['primary'], opacity=0.7),
                                   customdata=char_grp['top_keywords'],
                                   hovertemplate='<b>Char Count: %{x}</b><br>ROAS: %{y:.2f}x<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
                                   name='ROAS'), row=2, col=1)
    char_fig.add_trace(go.Scatter(x=char_grp['Character_Count'], y=char_grp['CPA'],
                                   mode='markers', marker=dict(size=char_grp['size'], color=COLORS['warning'], opacity=0.7),
                                   customdata=char_grp['top_keywords'],
                                   hovertemplate='<b>Char Count: %{x}</b><br>CPA: $%{y:.2f}<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
                                   name='CPA'), row=2, col=2)
    char_fig.update_xaxes(title_text="Character Count", title_font=dict(color='white'), tickfont=dict(color='white'),
                          row=1, col=1)
    char_fig.update_xaxes(title_text="Character Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=2)
    char_fig.update_xaxes(title_text="Character Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=1)
    char_fig.update_xaxes(title_text="Character Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=2)
    char_fig.update_yaxes(title_text="CTR (%)", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=1)
    char_fig.update_yaxes(title_text="CVR (%)", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=2)
    char_fig.update_yaxes(title_text="ROAS", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=1)
    char_fig.update_yaxes(title_text="CPA ($)", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=2)
    char_fig.update_layout(height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.3)', showlegend=False,font=dict(color='white'), xaxis=dict(color='white'),yaxis=dict(color='white'))
    # 9. WORD COUNT - aggregated by word count (1 bubble per word count)
    # 9. WORD COUNT - aggregated by word count (1 bubble per word count)
    word_count_list = []
    for word_count in d['Word_Count'].dropna().unique():
       wc_data = d[d['Word_Count'] == word_count]
       top_keywords = wc_data.nlargest(3, 'Clicks')['Keyword'].tolist()
       top_clicks = wc_data.nlargest(3, 'Clicks')['Clicks'].tolist()
       keywords_text = '<br>'.join([f"  • {kw} ({int(c)} clicks)" for kw, c in zip(top_keywords, top_clicks)])
       word_count_list.append({
        'Word_Count': word_count,
        'Clicks': wc_data['Clicks'].sum(),
        'CTR': weighted_ctr(wc_data),
        'CVR': weighted_cvr(wc_data),
        'CPA': weighted_cpa(wc_data),
        'ROAS': weighted_roas(wc_data),
        'top_keywords': keywords_text})
    word_grp = pd.DataFrame(word_count_list)
    word_count_fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("CTR by Word Count", "CVR by Word Count",
                                              "ROAS by Word Count", "CPA by Word Count"),
                                vertical_spacing=0.12, horizontal_spacing=0.1)
    max_word_clicks = word_grp['Clicks'].max()
    word_grp['size'] = 10 + (word_grp['Clicks'] / max_word_clicks) * 40
    word_count_fig.add_trace(go.Scatter(x=word_grp['Word_Count'], y=word_grp['CTR'],mode='markers', marker=dict(size=word_grp['size'], color=COLORS['info'], opacity=0.7), customdata=word_grp['top_keywords'], hovertemplate='<b>Word Count: %{x}</b><br>CTR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>',name='CTR'), row=1, col=1)
    word_count_fig.add_trace(go.Scatter(x=word_grp['Word_Count'], y=word_grp['CVR'],mode='markers', marker=dict(size=word_grp['size'], color=COLORS['success'], opacity=0.7),customdata=word_grp['top_keywords'], hovertemplate='<b>Word Count: %{x}</b><br>CVR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>', name='CVR'), row=1, col=2)
    word_count_fig.add_trace(go.Scatter(x=word_grp['Word_Count'], y=word_grp['ROAS'],mode='markers', marker=dict(size=word_grp['size'], color=COLORS['primary'], opacity=0.7),customdata=word_grp['top_keywords'],hovertemplate='<b>Word Count: %{x}</b><br>ROAS: %{y:.2f}x<br><br>Top Keywords:<br>%{customdata}<extra></extra>',name='ROAS'), row=2, col=1)
    word_count_fig.add_trace(go.Scatter(x=word_grp['Word_Count'], y=word_grp['CPA'],mode='markers', marker=dict(size=word_grp['size'], color=COLORS['warning'], opacity=0.7),customdata=word_grp['top_keywords'],hovertemplate='<b>Word Count: %{x}</b><br>CPA: $%{y:.2f}<br><br>Top Keywords:<br>%{customdata}<extra></extra>',name='CPA'), row=2, col=2)
    word_count_fig.update_xaxes(title_text="Word Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=1)
    word_count_fig.update_xaxes(title_text="Word Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=2)
    word_count_fig.update_xaxes(title_text="Word Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=1)
    word_count_fig.update_xaxes(title_text="Word Count", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=2)
    word_count_fig.update_yaxes(title_text="CTR (%)", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=1)
    word_count_fig.update_yaxes(title_text="CVR (%)", title_font=dict(color='white'), tickfont=dict(color='white'), row=1, col=2)
    word_count_fig.update_yaxes(title_text="ROAS", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=1)
    word_count_fig.update_yaxes(title_text="CPA ($)", title_font=dict(color='white'), tickfont=dict(color='white'), row=2, col=2)
    word_count_fig.update_layout(
      height=700,
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(30,30,40,0.3)',
      showlegend=False,
      font=dict(color='white')
)
    
    # DEBUG SPECIFICITY
    
    
    # SPECIFICITY SCORE ANALYSIS - aggregated by specificity score
    # SPECIFICITY SCORE ANALYSIS with Top 3 Keywords
    if 'Specificity_Score' in d.columns and d['Specificity_Score'].notna().any():
    # Get aggregated data AND top 3 keywords per specificity score
        spec_data = []
        for score in d['Specificity_Score'].dropna().unique():
           score_data = d[d['Specificity_Score'] == score]
           top_keywords = score_data.nlargest(3, 'Clicks')['Keyword'].tolist()
           top_clicks = score_data.nlargest(3, 'Clicks')['Clicks'].tolist()
           keywords_text = '<br>'.join([f"  • {kw} ({int(c)} clicks)" for kw, c in zip(top_keywords, top_clicks)])
           spec_data.append({
            'Specificity_Score': score,
            'Clicks': score_data['Clicks'].sum(),
            'CTR': weighted_ctr(score_data),
            'CVR': weighted_cvr(score_data),
            'CPA': weighted_cpa(score_data),
            'ROAS': weighted_roas(score_data),
            'top_keywords': keywords_text
        })
        spec_grp = pd.DataFrame(spec_data)
        order_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 3}
        spec_grp['sort_order'] = spec_grp['Specificity_Score'].map(order_map)
        spec_grp = spec_grp.sort_values('sort_order')
        specificity_fig = make_subplots(rows=2, cols=2,
                                     subplot_titles=("CTR by Specificity", "CVR by Specificity",
                                                   "ROAS by Specificity", "CPA by Specificity"),
                                     vertical_spacing=0.12, horizontal_spacing=0.1)
        max_spec_clicks = spec_grp['Clicks'].max()
        spec_grp['size'] = 10 + (spec_grp['Clicks'] / max_spec_clicks) * 40
    # CTR
        specificity_fig.add_trace(go.Scatter(
          x=spec_grp['Specificity_Score'], y=spec_grp['CTR'],mode='markers', marker=dict(size=spec_grp['size'], color=COLORS['info'], opacity=0.7), customdata=spec_grp['top_keywords'], hovertemplate='<b>Specificity: %{x}</b><br>CTR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>', name='CTR'), row=1, col=1)
    # CVR
        specificity_fig.add_trace(go.Scatter(
          x=spec_grp['Specificity_Score'], y=spec_grp['CVR'], mode='markers', marker=dict(size=spec_grp['size'], color=COLORS['success'], opacity=0.7),customdata=spec_grp['top_keywords'], hovertemplate='<b>Specificity: %{x}</b><br>CVR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>',name='CVR'), row=1, col=2)
    # ROAS
        specificity_fig.add_trace(go.Scatter(
          x=spec_grp['Specificity_Score'], y=spec_grp['ROAS'],mode='markers', marker=dict(size=spec_grp['size'], color=COLORS['primary'], opacity=0.7), customdata=spec_grp['top_keywords'], hovertemplate='<b>Specificity: %{x}</b><br>ROAS: %{y:.2f}x<br><br>Top Keywords:<br>%{customdata}<extra></extra>', name='ROAS'), row=2, col=1)
    # CPA
        specificity_fig.add_trace(go.Scatter(x=spec_grp['Specificity_Score'], y=spec_grp['CPA'],mode='markers', marker=dict(size=spec_grp['size'], color=COLORS['warning'], opacity=0.7), customdata=spec_grp['top_keywords'], hovertemplate='<b>Specificity: %{x}</b><br>CPA: $%{y:.2f}<br><br>Top Keywords:<br>%{customdata}<extra></extra>',name='CPA'), row=2, col=2)
        specificity_fig.update_xaxes(title_text="Specificity Score", color='white', row=1, col=1, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        specificity_fig.update_xaxes(title_text="Specificity Score", color='white', row=1, col=2, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        specificity_fig.update_xaxes(title_text="Specificity Score", color='white', row=2, col=1, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        specificity_fig.update_xaxes(title_text="Specificity Score", color='white', row=2, col=2, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        specificity_fig.update_yaxes(title_text="CTR (%)", color='white', row=1, col=1)
        specificity_fig.update_yaxes(title_text="CVR (%)", color='white', row=1, col=2)
        specificity_fig.update_yaxes(title_text="ROAS", color='white', row=2, col=1)
        specificity_fig.update_yaxes(title_text="CPA ($)", color='white', row=2, col=2)
        specificity_fig.update_layout(
          height=700,
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(30,30,40,0.3)',
          showlegend=False,
          font=dict(color='white')
    )
    else:
        specificity_fig = go.Figure()
        
    # URGENCY LEVEL ANALYSIS with Top 3 Keywords - ORDERED Low, Medium, High
    if 'Urgency_Level' in d.columns and d['Urgency_Level'].notna().any():
        urgency_data = []
        for level in d['Urgency_Level'].dropna().unique():
            level_data = d[d['Urgency_Level'] == level]
            top_keywords = level_data.nlargest(3, 'Clicks')['Keyword'].tolist()
            top_clicks = level_data.nlargest(3, 'Clicks')['Clicks'].tolist()
            keywords_text = '<br>'.join([f"  • {kw} ({int(c)} clicks)" for kw, c in zip(top_keywords, top_clicks)])
            urgency_data.append({
                'Urgency_Level': level,
                'Clicks': level_data['Clicks'].sum(),
                'CTR': weighted_ctr(level_data),
                'CVR': weighted_cvr(level_data),
                'CPA': weighted_cpa(level_data),
                'ROAS': weighted_roas(level_data),
                'top_keywords': keywords_text
            })
        
        urgency_grp = pd.DataFrame(urgency_data)
        
        # Order: Low, Medium, High
        order_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 3}
        urgency_grp['sort_order'] = urgency_grp['Urgency_Level'].map(order_map)
        urgency_grp = urgency_grp.sort_values('sort_order')
        
        urgency_fig = make_subplots(rows=2, cols=2,
                                     subplot_titles=("CTR by Urgency", "CVR by Urgency",
                                                   "ROAS by Urgency", "CPA by Urgency"),
                                     vertical_spacing=0.12, horizontal_spacing=0.1)
        
        max_urgency_clicks = urgency_grp['Clicks'].max()
        urgency_grp['size'] = 10 + (urgency_grp['Clicks'] / max_urgency_clicks) * 40
        
        # CTR
        urgency_fig.add_trace(go.Scatter(
            x=urgency_grp['Urgency_Level'], y=urgency_grp['CTR'],
            mode='markers', marker=dict(size=urgency_grp['size'], color=COLORS['info'], opacity=0.7), 
            customdata=urgency_grp['top_keywords'], 
            hovertemplate='<b>Urgency: %{x}</b><br>CTR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
            name='CTR'), row=1, col=1)
        
        # CVR
        urgency_fig.add_trace(go.Scatter(
            x=urgency_grp['Urgency_Level'], y=urgency_grp['CVR'],
            mode='markers', marker=dict(size=urgency_grp['size'], color=COLORS['success'], opacity=0.7),
            customdata=urgency_grp['top_keywords'], 
            hovertemplate='<b>Urgency: %{x}</b><br>CVR: %{y:.2f}%<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
            name='CVR'), row=1, col=2)
        
        # ROAS
        urgency_fig.add_trace(go.Scatter(
            x=urgency_grp['Urgency_Level'], y=urgency_grp['ROAS'],
            mode='markers', marker=dict(size=urgency_grp['size'], color=COLORS['primary'], opacity=0.7), 
            customdata=urgency_grp['top_keywords'], 
            hovertemplate='<b>Urgency: %{x}</b><br>ROAS: %{y:.2f}x<br><br>Top Keywords:<br>%{customdata}<extra></extra>', 
            name='ROAS'), row=2, col=1)
        
        # CPA
        urgency_fig.add_trace(go.Scatter(
            x=urgency_grp['Urgency_Level'], y=urgency_grp['CPA'],
            mode='markers', marker=dict(size=urgency_grp['size'], color=COLORS['warning'], opacity=0.7), 
            customdata=urgency_grp['top_keywords'], 
            hovertemplate='<b>Urgency: %{x}</b><br>CPA: $%{y:.2f}<br><br>Top Keywords:<br>%{customdata}<extra></extra>',
            name='CPA'), row=2, col=2)
        
        urgency_fig.update_xaxes(title_text="Urgency Level", color='white', row=1, col=1, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        urgency_fig.update_xaxes(title_text="Urgency Level", color='white', row=1, col=2, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        urgency_fig.update_xaxes(title_text="Urgency Level", color='white', row=2, col=1, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        urgency_fig.update_xaxes(title_text="Urgency Level", color='white', row=2, col=2, categoryorder='array', categoryarray=['Low', 'Medium', 'High', 'Unknown'])
        urgency_fig.update_yaxes(title_text="CTR (%)", color='white', row=1, col=1)
        urgency_fig.update_yaxes(title_text="CVR (%)", color='white', row=1, col=2)
        urgency_fig.update_yaxes(title_text="ROAS", color='white', row=2, col=1)
        urgency_fig.update_yaxes(title_text="CPA ($)", color='white', row=2, col=2)
        urgency_fig.update_layout(
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,40,0.3)',
            showlegend=False,
            font=dict(color='white')
        )
    else:
        urgency_fig = go.Figure()    
    # 10. NUMBER PRESENT - bubble chart
    # 10. NUMBER PRESENT - bubble chart with top keywords
    # Number Present - aggregate data for Yes/No
    num_present_list = []
    for num_val in d['Is_Number_Present'].dropna().unique():
        num_subset = d[d['Is_Number_Present'] == num_val]
        if len(num_subset) > 0:
            top_keywords = num_subset.nlargest(3, 'Clicks')['Keyword'].tolist()
            top_clicks = num_subset.nlargest(3, 'Clicks')['Clicks'].tolist()
            keywords_text = '<br>'.join([f"• {kw} ({int(c)} clicks)" 
                                     for kw, c in zip(top_keywords, top_clicks)])
        
            num_present_list.append({
            'Is_Number_Present': num_val,
            'Clicks': num_subset['Clicks'].sum(),
            'CTR': weighted_ctr(num_subset),
            'CVR': weighted_cvr(num_subset),
            'CPA': weighted_cpa(num_subset),
            'ROAS': weighted_roas(num_subset),
            'top_keywords': keywords_text
        })

    num_grp = pd.DataFrame(num_present_list)
    num_fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("CTR", "CVR", "ROAS", "CPA"),
                        vertical_spacing=0.15, horizontal_spacing=0.15)

    max_num_clicks = num_grp['Clicks'].max()
    for _, r in num_grp.iterrows():
        size = 40 + (r['Clicks'] / max_num_clicks) * 60
    
        num_fig.add_trace(go.Scatter(
         x=[r['Is_Number_Present']], y=[r['CTR']],
         mode='markers+text',
         text=[f"{r['CTR']:.1f}%"],
         textposition='middle center',
         textfont=dict(size=11, color='white'),
         marker=dict(size=size, color=COLORS['info'], opacity=0.8),
         customdata=[[r['Clicks'], r['top_keywords']]],
         hovertemplate=f"<b>Number Present: %{{x}}</b><br>" +
                     f"CTR: %{{y:.2f}}%<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",
         showlegend=False), row=1, col=1)
    
        num_fig.add_trace(go.Scatter(
         x=[r['Is_Number_Present']], y=[r['CVR']],
         mode='markers+text',
         text=[f"{r['CVR']:.1f}%"],
         textposition='middle center',
         textfont=dict(size=11, color='white'),
         marker=dict(size=size, color=COLORS['success'], opacity=0.8),
         customdata=[[r['Clicks'], r['top_keywords']]],
         hovertemplate=f"<b>Number Present: %{{x}}</b><br>" +
                     f"CVR: %{{y:.2f}}%<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",
         showlegend=False), row=1, col=2)
    
        num_fig.add_trace(go.Scatter(
         x=[r['Is_Number_Present']], y=[r['ROAS']],
         mode='markers+text',
         text=[f"{r['ROAS']:.1f}x"],
         textposition='middle center',
         textfont=dict(size=11, color='white'),
         marker=dict(size=size, color=COLORS['primary'], opacity=0.8),
         customdata=[[r['Clicks'], r['top_keywords']]],
         hovertemplate=f"<b>Number Present: %{{x}}</b><br>" +
                     f"ROAS: %{{y:.2f}}x<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",
         showlegend=False), row=2, col=1)
    
        num_fig.add_trace(go.Scatter(
         x=[r['Is_Number_Present']], y=[r['CPA']],
         mode='markers+text',
         text=[f"${r['CPA']:.1f}"],
         textposition='middle center',
         textfont=dict(size=11, color='white'),
         marker=dict(size=size, color=COLORS['warning'], opacity=0.8),
         customdata=[[r['Clicks'], r['top_keywords']]],
         hovertemplate=f"<b>Number Present: %{{x}}</b><br>" +
                     f"CPA: $%{{y:.2f}}<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",
         showlegend=False), row=2, col=2)

        num_fig.update_xaxes(title_text="Number Present", row=1, col=1)
        num_fig.update_xaxes(title_text="Number Present", row=1, col=2)
        num_fig.update_xaxes(title_text="Number Present", row=2, col=1)
        num_fig.update_xaxes(title_text="Number Present", row=2, col=2)
        num_fig.update_layout(
            height=600, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(30,30,40,0.3)',
            font=dict(color='white'),
            xaxis=dict(color='white'),
            yaxis=dict(color='white'))
        
    # 11. NUMBER POSITION - bubble chart
    # 11. NUMBER POSITION - bubble chart with top keywords
    # Number Position - aggregate by position
    num_pos_list = []
    for pos in sorted(d[d['Position_of_Number'].notna()]['Position_of_Number'].unique()):
       pos_subset = d[d['Position_of_Number'] == pos]
       if len(pos_subset) > 0:
            top_keywords = pos_subset.nlargest(3, 'Clicks')['Keyword'].tolist()
            top_clicks = pos_subset.nlargest(3, 'Clicks')['Clicks'].tolist()
            keywords_text = '<br>'.join([f"• {kw} ({int(c)} clicks)" 
                                     for kw, c in zip(top_keywords, top_clicks)])
        
            num_pos_list.append({
            'Position_of_Number': int(pos),  # Convert to int for cleaner display
            'Clicks': pos_subset['Clicks'].sum(),
            'CTR': weighted_ctr(pos_subset),
            'CVR': weighted_cvr(pos_subset),
            'CPA': weighted_cpa(pos_subset),
            'ROAS': weighted_roas(pos_subset),
            'top_keywords': keywords_text
        })

    num_pos_grp = pd.DataFrame(num_pos_list)
    num_pos_fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("CTR by Number Position", "CVR by Number Position",
                                          "ROAS by Number Position", "CPA by Number Position"),
                            vertical_spacing=0.15, horizontal_spacing=0.15)
    if not num_pos_grp.empty:
        max_pos_clicks = num_pos_grp['Clicks'].max()
        num_pos_grp['size'] = 10 + (num_pos_grp['Clicks'] / max_pos_clicks) * 30
    
    # CTR
        num_pos_fig.add_trace(go.Scatter(
            x=num_pos_grp['Position_of_Number'], y=num_pos_grp['CTR'],
            mode='markers+text',
            text=num_pos_grp['CTR'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center',
            textfont=dict(size=10, color='white'),
            marker=dict(size=num_pos_grp['size'], color=COLORS['info'], opacity=0.8),
            customdata=list(zip(num_pos_grp['Clicks'], num_pos_grp['top_keywords'])),
            hovertemplate='<b>Position: %{x}</b><br>CTR: %{y:.2f}%<br><b>Clicks: %{customdata[0]:,}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>',
            showlegend=False), row=1, col=1)
    
        num_pos_fig.add_trace(go.Scatter(
            x=num_pos_grp['Position_of_Number'], y=num_pos_grp['CVR'],
            mode='markers+text',
            text=num_pos_grp['CVR'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center',
            textfont=dict(size=10, color='white'),
            marker=dict(size=num_pos_grp['size'], color=COLORS['success'], opacity=0.8),
            customdata=list(zip(num_pos_grp['Clicks'], num_pos_grp['top_keywords'])),
            hovertemplate='<b>Position: %{x}</b><br>CVR: %{y:.2f}%<br><b>Clicks: %{customdata[0]:,}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>',
            showlegend=False), row=1, col=2)
    
    # ROAS
        num_pos_fig.add_trace(go.Scatter(
            x=num_pos_grp['Position_of_Number'], y=num_pos_grp['ROAS'],
            mode='markers+text',
            text=num_pos_grp['ROAS'].apply(lambda x: f"{x:.1f}x"),
            textposition='top center',
            textfont=dict(size=10, color='white'),
            marker=dict(size=num_pos_grp['size'], color=COLORS['primary'], opacity=0.8),
            customdata=list(zip(num_pos_grp['Clicks'], num_pos_grp['top_keywords'])),
            hovertemplate='<b>Position: %{x}</b><br>ROAS: %{y:.2f}x<br><b>Clicks: %{customdata[0]:,}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>',
            showlegend=False), row=2, col=1)
    
    # CPA
        num_pos_fig.add_trace(go.Scatter(
            x=num_pos_grp['Position_of_Number'], y=num_pos_grp['CPA'],
            mode='markers+text',
            text=num_pos_grp['CPA'].apply(lambda x: f"${x:.1f}"),
            textposition='top center',
            textfont=dict(size=10, color='white'),
            marker=dict(size=num_pos_grp['size'], color=COLORS['warning'], opacity=0.8),
            customdata=list(zip(num_pos_grp['Clicks'], num_pos_grp['top_keywords'])),
            hovertemplate='<b>Position: %{x}</b><br>CPA: $%{y:.2f}<br><b>Clicks: %{customdata[0]:,}</b><br><br>Top Keywords:<br>%{customdata[1]}<extra></extra>',
            showlegend=False), row=2, col=2)

    num_pos_fig.update_xaxes(title_text="Position", row=1, col=1)
    num_pos_fig.update_xaxes(title_text="Position", row=1, col=2)
    num_pos_fig.update_xaxes(title_text="Position", row=2, col=1)
    num_pos_fig.update_xaxes(title_text="Position", row=2, col=2)
    num_pos_fig.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(30,30,40,0.3)',
    font=dict(color='white'),
    xaxis=dict(color='white'),
    yaxis=dict(color='white')
)
    
    # 12. QUESTION ANALYSIS - bubble chart
    # 12. QUESTION ANALYSIS - bubble chart with top keywords
    question_data = []
    for is_q in d['Is_Question'].dropna().unique():
        q_subset = d[d['Is_Question'] == is_q]
        top_keywords = q_subset.nlargest(3, 'Clicks')['Keyword'].tolist()
        top_clicks = q_subset.nlargest(3, 'Clicks')['Clicks'].tolist()
        keywords_text = '<br>'.join([f"• {kw} ({int(c)} clicks)" 
                                 for kw, c in zip(top_keywords, top_clicks)])
    
        question_data.append({
        'Is_Question': is_q,
        'Clicks': q_subset['Clicks'].sum(),
        'CTR': weighted_ctr(q_subset),
        'CVR': weighted_cvr(q_subset),
        'CPA': weighted_cpa(q_subset),
        'ROAS': weighted_roas(q_subset),
        'top_keywords': keywords_text
    })

    question_grp = pd.DataFrame(question_data)
    question_fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("CTR", "CVR", "ROAS", "CPA"),
                            vertical_spacing=0.15, horizontal_spacing=0.15)

    max_q_clicks = question_grp['Clicks'].max()
    for _, r in question_grp.iterrows():
        size = 60 + (r['Clicks'] / max_q_clicks) * 80
    
        question_fig.add_trace(go.Scatter(
          x=[r['Is_Question']], y=[r['CTR']],
          mode='markers+text',
          text=[f"{r['CTR']:.1f}%"],
          textposition='middle center',
          textfont=dict(size=11, color='white'),
          marker=dict(size=size, color=COLORS['info'], opacity=0.8),
          customdata=[[r['Clicks'], r['top_keywords']]],
          hovertemplate=f"<b>Is Question: %{{x}}</b><br>CTR: %{{y:.2f}}%<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",showlegend=False), row=1, col=1)
    
        question_fig.add_trace(go.Scatter(
          x=[r['Is_Question']], y=[r['CVR']],
          mode='markers+text',
          text=[f"{r['CVR']:.1f}%"],
          textposition='middle center',
          textfont=dict(size=11, color='white'),
          marker=dict(size=size, color=COLORS['success'], opacity=0.8),
          customdata=[[r['Clicks'], r['top_keywords']]],
          hovertemplate=f"<b>Is Question: %{{x}}</b><br>CVR: %{{y:.2f}}%<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",showlegend=False), row=1, col=2)
    
        question_fig.add_trace(go.Scatter(
          x=[r['Is_Question']], y=[r['ROAS']],
          mode='markers+text',
          text=[f"{r['ROAS']:.1f}x"],
          textposition='middle center',
          textfont=dict(size=11, color='white'),
          marker=dict(size=size, color=COLORS['primary'], opacity=0.8),
          customdata=[[r['Clicks'], r['top_keywords']]],
          hovertemplate=f"<b>Is Question: %{{x}}</b><br>ROAS: %{{y:.2f}}x<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",showlegend=False), row=2, col=1)
    
        question_fig.add_trace(go.Scatter(
          x=[r['Is_Question']], y=[r['CPA']],
          mode='markers+text',
          text=[f"${r['CPA']:.1f}"],
          textposition='middle center',
          textfont=dict(size=11, color='white'),
          marker=dict(size=size, color=COLORS['warning'], opacity=0.8),
          customdata=[[r['Clicks'], r['top_keywords']]],
          hovertemplate=f"<b>Is Question: %{{x}}</b><br>CPA: $%{{y:.2f}}<br>" +
                     f"<b>Clicks: %{{customdata[0]:,}}</b><br>" +
                     f"<b>Top Keywords:</b><br>%{{customdata[1]}}<extra></extra>",showlegend=False), row=2, col=2)

        question_fig.update_xaxes(title_text="Is Question", row=1, col=1)
        question_fig.update_xaxes(title_text="Is Question", row=1, col=2)
        question_fig.update_xaxes(title_text="Is Question", row=2, col=1)
        question_fig.update_xaxes(title_text="Is Question", row=2, col=2)
        question_fig.update_layout(
            height=600,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(30,30,40,0.3)',font=dict(color='white'),xaxis=dict(color='white'),yaxis=dict(color='white')
)
    # Table preview
    preview_cols = ['Keyword', 'Clicks', 'CTR', 'CVR', 'CPA', 'ROAS', 'Campaign_Type', 'Query_Type']
    preview_df = d[preview_cols].sort_values('Clicks', ascending=False).head(100)  # Increased to 100
    table_children = dash_table.DataTable(
    data=preview_df.to_dict('records'),
    columns=[{"name": i, "id": i} for i in preview_df.columns],
    page_size=30,
    page_current=0,
    style_table={'overflowX': 'auto'},
    style_cell={
        'textAlign': 'left',
        'backgroundColor': '#121419',
        'color': '#E5E7EB',
        'border': '1px solid #2d3748',
        'padding': '8px'
    },
    style_header={
        'backgroundColor': '#1a1a1a',
        'fontWeight': 'bold',
        'color': '#00D9FF',
        'border': '1px solid #2d3748'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#1a1f2e'
        }
    ]
)
    if hasattr(fig_cat, 'data') and len(fig_cat.data) > 0:
        print(f"First trace type: {type(fig_cat.data[0])}")
        print(f"First trace x data: {fig_cat.data[0].x if hasattr(fig_cat.data[0], 'x') else 'NO X'}")
    gc.collect()
    return (stat_row, treemap_ctr_cvr, treemap_cpa_roas,
        fig_cat, keyword_category_fig, emo_ctr_cvr, emo_roas_cpa, char_fig, specificity_fig,urgency_fig, word_count_fig,
        num_fig, num_pos_fig, question_fig, table_children)
# Download callback
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State('objective-dropdown','value'),
    State('advertiser-dropdown','value'),
    State('campaign-type-dropdown','value'),
    State('campaign-dropdown','value'),
    prevent_initial_call=True
)
def download_data(n, obj, adv, ctype, camp):
    if n is None:
        raise PreventUpdate
    d = work.copy()
    if obj: d = d[d['Campaign_Objective']==obj]
    if adv: d = d[d['Advertiser']==adv]
    if ctype: d = d[d['Campaign_Type']==ctype]
    if camp: d = d[d['Campaign']==camp]
    return dcc.send_data_frame(d.to_csv, "filtered_data.csv", index=False)

@app.callback(
    Output("download-keyword-category", "data"),
    Input("download-keyword-category-btn", "n_clicks"),
    State('objective-dropdown','value'),
    State('advertiser-dropdown','value'),
    State('campaign-type-dropdown','value'),
    State('campaign-dropdown','value'),
    prevent_initial_call=True
)



def download_keyword_category(n, obj, adv, ctype, camp):
    if n is None:
        raise PreventUpdate
    d = work.copy()
    if obj: d = d[d['Campaign_Objective']==obj]
    if adv: d = d[d['Advertiser']==adv]
    if ctype: d = d[d['Campaign_Type']==ctype]
    if camp: d = d[d['Campaign']==camp]
    
    if 'Keyword_Category' in d.columns and d['Keyword_Category'].notna().any():
        kw_cat = d.groupby('Keyword_Category').apply(lambda g: pd.Series({
            'Clicks': g['Clicks'].sum(),
            'CTR': weighted_ctr(g),
            'CVR': weighted_cvr(g),
            'CPA': weighted_cpa(g),
            'ROAS': weighted_roas(g)
        })).reset_index()
        return dcc.send_data_frame(kw_cat.to_csv, "keyword_category_analysis.csv", index=False)
    return None
# ==================== DOMAIN TAB CALLBACKS ====================
# Domain dropdown population
# Domain main update
# Domain main update

@app.callback(
    Output('domain-stats','children'),
    Output('domain_treemap_ctr_cvr','figure'),
    Output('domain_treemap_cpa_roas','figure'),
    Output('domain_category_overview','figure'),
    Output('domain_category_ctr_cvr','figure'),
    Output('domain_category_roas_cpa','figure'),
    Output('domain_table_preview','children'),
    Input('objective-dropdown','value'),
    Input('advertiser-dropdown','value'),
    Input('campaign-type-dropdown','value'),
    Input('campaign-dropdown','value'),
    Input('analysis-tabs', 'active_tab')
    #prevent_initial_call=True
)

def update_domain_dashboard(obj, adv, ctype, camp, active_tab):
    if active_tab != "domain-tab":  # ✅ Only run when domain tab is active
        raise PreventUpdate
    d = work_domain.copy()
    if obj: d = d[d['Campaign_Objective']==obj]
    if adv: d = d[d['Advertiser']==adv]
    if ctype: d = d[d['Campaign_Type']==ctype]
    if camp: d = d[d['Campaign']==camp]
    
    

    if d.shape[0] == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        return (html.Div("No data"), empty_fig, empty_fig, empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, html.Div("No data"))

    # Stats
    total_clicks = int(d['Clicks'].sum())
    total_impressions = int(d['Impressions'].sum())
    avg_ctr = weighted_ctr(d)
    avg_cvr = weighted_cvr(d)
    avg_cpa = weighted_cpa(d)
    avg_roas = weighted_roas(d)
    
    stat_row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Total Impressions", className='small-muted'), 
            html.Div(f"{total_impressions:,}", className='big-number')
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Total Clicks", className='small-muted'), 
            html.Div(f"{total_clicks:,}", className='big-number', style={'color':COLORS['secondary']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg CTR (%)", className='small-muted'), 
            html.Div(f"{avg_ctr:.2f}", className='big-number', style={'color':COLORS['info']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg CVR (%)", className='small-muted'), 
            html.Div(f"{avg_cvr:.2f}", className='big-number', style={'color':COLORS['success']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg CPA", className='small-muted'), 
            html.Div(f"${avg_cpa:.2f}", className='big-number', style={'color':COLORS['warning']})
        ])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg ROAS", className='small-muted'), 
            html.Div(f"{avg_roas:.2f}x", className='big-number', style={'color':COLORS['primary']})
        ])), md=2),
    ], className='mb-3')

    # Domain aggregation
    domain_agg = d.groupby('Domain').apply(lambda g: pd.Series({
        'Clicks': g['Clicks'].sum(),
        'Impressions': g['Impressions'].sum(),
        'CTR': weighted_ctr(g),
        'CVR': weighted_cvr(g),
        'CPA': weighted_cpa(g),
        'ROAS': weighted_roas(g)
    })).reset_index().sort_values('Clicks', ascending=False).head(50)

    # 1. Domain Treemap CTR/CVR
    treemap_ctr_cvr = go.Figure()
    if not domain_agg.empty:
        text_labels = domain_agg.apply(
            lambda r: f"<b>{r['Domain']}</b><br>CTR: {r['CTR']:.1f}% | CVR: {r['CVR']:.1f}%", axis=1
        )
        treemap_ctr_cvr = go.Figure(go.Treemap(
            labels=domain_agg['Domain'],
            parents=[''] * len(domain_agg),
            values=domain_agg['Clicks'],
            text=text_labels,
            textposition="middle center",
            marker=dict(
                colors=domain_agg['CVR'].apply(lambda x: cvr_color(x, 'CVR')),
                line=dict(width=2, color='#1a1a1a')
            ),
            textfont=dict(size=11, color='white')
        ))
        treemap_ctr_cvr.update_layout(
            margin=dict(l=5,r=5,t=5,b=5), 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=450,
            font=dict(color='white')
        )

    # 2. Domain Treemap CPA/ROAS
    treemap_cpa_roas = go.Figure()
    if not domain_agg.empty:
        text_labels = domain_agg.apply(
            lambda r: f"<b>{r['Domain']}</b><br>CPA: ${r['CPA']:.1f} | ROAS: {r['ROAS']:.1f}x", axis=1
        )
        treemap_cpa_roas = go.Figure(go.Treemap(
            labels=domain_agg['Domain'],
            parents=[''] * len(domain_agg),
            values=domain_agg['Clicks'],
            text=text_labels,
            textposition="middle center",
            marker=dict(
                colors=domain_agg['CPA'].apply(lambda x: cvr_color(x, 'CPA')),
                line=dict(width=2, color='#1a1a1a')
            ),
            textfont=dict(size=11, color='white')
        ))
        treemap_cpa_roas.update_layout(
            
            margin=dict(l=5,r=5,t=5,b=5), 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=450,
            font=dict(color='white')
        )

    
    # 5. Domain Category Overview
    cat_overview = go.Figure()
    if 'Domain_Category' in d.columns and d['Domain_Category'].notna().any():
        cat_grp = d.groupby('Domain_Category').apply(lambda g: pd.Series({
            'Clicks': g['Clicks'].sum(),
            'CTR': weighted_ctr(g),
            'CVR': weighted_cvr(g),
            'CPA': weighted_cpa(g),
            'ROAS': weighted_roas(g)
        })).reset_index().sort_values('Clicks', ascending=False).head(10)
        
        if not cat_grp.empty:
            fig_cat = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
    )
            
            fig_cat.add_trace(go.Bar(x=cat_grp['Domain_Category'], y=cat_grp['CTR'], 
                             name='CTR (%)', marker_color=COLORS['info']), secondary_y=False)
            fig_cat.add_trace(go.Bar(x=cat_grp['Domain_Category'], y=cat_grp['CVR'], 
                             name='CVR (%)', marker_color=COLORS['success']), secondary_y=False)
    
    # Plot CPA and ROAS on secondary y-axis
            fig_cat.add_trace(go.Bar(x=cat_grp['Domain_Category'], y=cat_grp['CPA'], 
                             name='CPA ($)', marker_color=COLORS['warning']), secondary_y=True)
            fig_cat.add_trace(go.Bar(x=cat_grp['Domain_Category'], y=cat_grp['ROAS'], 
                             name='ROAS (x)', marker_color=COLORS['primary']), secondary_y=True)
            fig_cat.update_xaxes(title_text="Domain_Category", tickangle=-20)
            fig_cat.update_yaxes(title_text="CTR/CVR (%)", secondary_y=False)
            fig_cat.update_yaxes(title_text="CPA ($) / ROAS (x)", secondary_y=True)
            fig_cat.update_layout(
                barmode='group', height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), xaxis=dict(color='white'), yaxis=dict(color='white')
    )
    # 6. Domain Category Bubble CTR/CVR
    cat_ctr_cvr = go.Figure()
    if 'Domain_Category' in d.columns and d['Domain_Category'].notna().any():
    # Get top 3 domains per category
        dom_cat_top = {}
        for cat in d['Domain_Category'].dropna().unique():
            cat_data = d[d['Domain_Category'] == cat]
            if len(cat_data) > 0 and 'Domain' in cat_data.columns:
                top_doms = cat_data.nlargest(3, 'Clicks')
                dom_cat_top[cat] = '<br>'.join([f"• {dom} ({int(c)} clicks)" 
                                            for dom, c in zip(top_doms['Domain'], top_doms['Clicks'])])
            else:
                dom_cat_top[cat] = 'N/A'
    
        cat_grp = d.groupby('Domain_Category').apply(lambda g: pd.Series({
            'Clicks': g['Clicks'].sum(),
            'CTR': weighted_ctr(g),
            'CVR': weighted_cvr(g),
            'CPA': weighted_cpa(g),
            'ROAS': weighted_roas(g)})).reset_index().sort_values('Clicks', ascending=False).head(10)
    
        if not cat_grp.empty:
            for col in ['Clicks', 'CTR', 'CVR', 'CPA', 'ROAS']:
                min_val, max_val = cat_grp[col].min(), cat_grp[col].max()
                cat_grp[f'{col}_norm'] = (cat_grp[col] - min_val) / (max_val - min_val + 1e-9) * 100
        
            cat_grp['top_domains'] = cat_grp['Domain_Category'].map(dom_cat_top)
        
            cat_overview = go.Figure()
            cat_overview.add_trace(go.Bar(
                x=cat_grp['Domain_Category'], 
                y=cat_grp['CTR_norm'], 
                name='CTR', 
                marker_color=COLORS['info'],
                customdata=list(zip(cat_grp['CTR'], cat_grp['Clicks'], cat_grp['top_domains'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual CTR: %{customdata[0]:.2f}%</b><br><b>Total Clicks: %{customdata[1]:,}</b><br><br>Top Domains:<br>%{customdata[2]}<extra></extra>'))
            cat_overview.add_trace(go.Bar(
                x=cat_grp['Domain_Category'], 
                y=cat_grp['CVR_norm'], 
                name='CVR', 
                marker_color=COLORS['success'],
                customdata=list(zip(cat_grp['CVR'], cat_grp['Clicks'], cat_grp['top_domains'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual CVR: %{customdata[0]:.2f}%</b><br><b>Total Clicks: %{customdata[1]:,}</b><br><br>Top Domains:<br>%{customdata[2]}<extra></extra>'))
            cat_overview.add_trace(go.Bar(
                x=cat_grp['Domain_Category'], 
                y=cat_grp['CPA_norm'], 
                name='CPA', 
                marker_color=COLORS['warning'],
                customdata=list(zip(cat_grp['CPA'], cat_grp['Clicks'], cat_grp['top_domains'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual CPA: $%{customdata[0]:.2f}</b><br><b>Total Clicks: %{customdata[1]:,}</b><br><br>Top Domains:<br>%{customdata[2]}<extra></extra>'))
            cat_overview.add_trace(go.Bar(
                x=cat_grp['Domain_Category'], 
                y=cat_grp['ROAS_norm'], 
                name='ROAS', 
                marker_color=COLORS['primary'],
                customdata=list(zip(cat_grp['ROAS'], cat_grp['Clicks'], cat_grp['top_domains'])),
                hovertemplate='<b>%{x}</b><br>Normalized: %{y:.1f}<br><b>Actual ROAS: %{customdata[0]:.2f}x</b><br><b>Total Clicks: %{customdata[1]:,}</b><br><br>Top Domains:<br>%{customdata[2]}<extra></extra>'))
    if 'Domain_Category' in d.columns and d['Domain_Category'].notna().any():
        cat_agg = d.groupby('Domain_Category').apply(lambda g: pd.Series({
            'Clicks': g['Clicks'].sum(),
            'CTR': weighted_ctr(g),
            'CVR': weighted_cvr(g),
            'CPA': weighted_cpa(g),
            'ROAS': weighted_roas(g)
        })).reset_index()
        
        max_cat_clicks = cat_agg['Clicks'].max()
        palette = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info'], COLORS['warning'], COLORS['danger']]
        
        for i, r in cat_agg.iterrows():
            size = 40 + (r['Clicks'] / max_cat_clicks) * 100
            cat_ctr_cvr.add_trace(go.Scatter(
                x=[r['CTR']], y=[r['CVR']],
                mode='markers+text',
                text=[r['Domain_Category']],
                textposition='middle center',
                textfont=dict(size=11, color='white', weight='bold'),
                marker=dict(size=size, color=palette[i % len(palette)], opacity=0.8, line=dict(width=3, color='white')),
                hovertemplate=f"<b>{r['Domain_Category']}</b><br>CTR: {r['CTR']:.2f}%<br>CVR: {r['CVR']:.2f}%<extra></extra>",
                showlegend=False
            ))
        cat_ctr_cvr.update_layout(
            xaxis_title="CTR (%)", yaxis_title="CVR (%)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.3)', 
            height=500,
            font=dict(color='white'),
            xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
        )

    # 7. Domain Category Bubble ROAS/CPA
    cat_roas_cpa = go.Figure()
    if 'Domain_Category' in d.columns and d['Domain_Category'].notna().any():
        for i, r in cat_agg.iterrows():
            size = 40 + (r['Clicks'] / max_cat_clicks) * 100
            cat_roas_cpa.add_trace(go.Scatter(
                x=[r['ROAS']], y=[r['CPA']],
                mode='markers+text',
                text=[r['Domain_Category']],
                textposition='middle center',
                textfont=dict(size=11, color='white', weight='bold'),
                marker=dict(size=size, color=palette[i % len(palette)], opacity=0.8, line=dict(width=3, color='white')),
                hovertemplate=f"<b>{r['Domain_Category']}</b><br>ROAS: {r['ROAS']:.2f}x<br>CPA: ${r['CPA']:.2f}<extra></extra>",
                showlegend=False
            ))
        cat_roas_cpa.update_layout(
            xaxis_title="ROAS", yaxis_title="CPA ($)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.3)', 
            height=500,
            font=dict(color='white'),
            xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
        )

    # Table preview
    preview_cols = ['Domain', 'Domain_Category', 'Clicks', 'CTR', 'CVR', 'CPA', 'ROAS']
    preview_df = d[preview_cols].sort_values('Clicks', ascending=False).head(100)  # Increased to 100
    table_children = dash_table.DataTable(
    data=preview_df.to_dict('records'),
    columns=[{"name": i, "id": i} for i in preview_df.columns],
    page_size=30,
    page_current=0,
    style_table={'overflowX': 'auto'},
    style_cell={
        'textAlign': 'left',
        'backgroundColor': '#121419',
        'color': '#E5E7EB',
        'border': '1px solid #2d3748',
        'padding': '8px'
    },
    style_header={
        'backgroundColor': '#1a1a1a',
        'fontWeight': 'bold',
        'color': '#00D9FF',
        'border': '1px solid #2d3748'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#1a1f2e'
        }
    ]
)

    return (stat_row, treemap_ctr_cvr, treemap_cpa_roas, cat_overview, cat_ctr_cvr, cat_roas_cpa, table_children)# Domain download callback
@app.callback(
    Output("download-domain-data", "data"),
    Input("download-domain-btn", "n_clicks"),
    State('objective-dropdown','value'),
    State('advertiser-dropdown','value'),
    State('campaign-type-dropdown','value'),
    State('campaign-dropdown','value'),
    prevent_initial_call=True
)
def download_domain_data(n, obj, adv, ctype, camp):
    if n is None:
        raise PreventUpdate
    d = work_domain.copy()
    if obj: d = d[d['Campaign_Objective']==obj]
    if adv: d = d[d['Advertiser']==adv]
    if ctype: d = d[d['Campaign_Type']==ctype]
    if camp: d = d[d['Campaign']==camp]
    return dcc.send_data_frame(d.to_csv, "filtered_domain_data.csv", index=False)
# Run

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))