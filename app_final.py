"""
╔══════════════════════════════════════════════════════════╗
║   FAKE NEWS DETECTION SYSTEM — BEAUTIFUL STREAMLIT UI   ║
║   Moaz Hassan | Roll No: 231168 | BSAI-VI-A             ║
╚══════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. pip install streamlit scikit-learn nltk lime requests transformers torch matplotlib plotly
  2. Put your models/ folder next to this file
  3. streamlit run app.py
"""

import os, re, warnings, requests, logging
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
warnings.filterwarnings("ignore")

# ── Suppress non-fatal Transformers / torchvision advisory logs ──────────────
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Tell Streamlit NOT to crawl every transformers sub-module on reload —
# this eliminates the flood of "No module named 'torchvision'" watcher spam.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.import_utils").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)

# ── Optional torchvision guard ───────────────────────────────────────────────
try:
    import torchvision  # noqa: F401 — imported only to confirm availability
except ImportError:
    # torchvision is not installed; BERT will still load but some fast
    # image-processing modules inside transformers will be unavailable.
    # Install with:  pip install torchvision
    pass


class TextPreprocessor:
    """Compatibility class for unpickling preprocessor objects trained in notebooks.

    Some training notebooks save this class under __main__.TextPreprocessor.
    Defining it here allows joblib to resolve the class during load.
    """

    def __init__(self, *args, **kwargs):
        pass

    def preprocess(self, text):
        if text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        stop_words = getattr(self, "stop_words", None)
        if stop_words:
            tokens = [tok for tok in text.split() if tok not in stop_words]
            text = " ".join(tokens)

        lemmatizer = getattr(self, "lemmatizer", None)
        if lemmatizer:
            try:
                text = " ".join(lemmatizer.lemmatize(tok) for tok in text.split())
            except Exception:
                pass

        return text

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield — AI News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  THEME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DARK  = "#080B12"
SURF  = "#0F1420"
SURF2 = "#141928"
RED   = "#FF3B5C"
GREEN = "#00C896"
BLUE  = "#3B82F6"
MUTED = "#6B7280"
TEXT  = "#E8ECF4"

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:#080B12; --surf:#0F1420; --surf2:#141928;
  --border:rgba(255,255,255,0.07);
  --red:#FF3B5C; --green:#00C896; --blue:#3B82F6;
  --text:#E8ECF4; --muted:#6B7280;
  --head:'Syne',sans-serif; --body:'DM Sans',sans-serif; --mono:'JetBrains Mono',monospace;
}

html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--body)!important;}
[data-testid="stHeader"]{background:transparent!important;}
section[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--border);}

h1,h2,h3,h4{font-family:var(--head)!important;color:var(--text)!important;}
p,span,label,div{font-family:var(--body)!important;}

/* Hero */
.hero{background:linear-gradient(135deg,#0F1420 0%,#141928 40%,#0F1420 100%);border:1px solid var(--border);border-radius:20px;padding:3rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(255,59,92,.12) 0%,transparent 70%);border-radius:50%;}
.hero::after{content:'';position:absolute;bottom:-40px;left:30%;width:200px;height:200px;background:radial-gradient(circle,rgba(59,130,246,.1) 0%,transparent 70%);border-radius:50%;}
.hero-badge{display:inline-block;background:rgba(255,59,92,.15);border:1px solid rgba(255,59,92,.3);color:#FF3B5C;font-family:var(--mono);font-size:.7rem;padding:3px 10px;border-radius:20px;margin-bottom:1rem;letter-spacing:.08em;}
.hero-title{font-family:var(--head)!important;font-size:2.8rem!important;font-weight:800!important;line-height:1.1!important;margin:0 0 .5rem!important;background:linear-gradient(135deg,#fff 40%,#FF3B5C);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-family:var(--body);font-size:1rem;color:var(--muted);margin:0!important;}

/* Cards */
.card{background:var(--surf);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.card-title{font-family:var(--head)!important;font-size:.85rem!important;font-weight:700!important;letter-spacing:.08em!important;text-transform:uppercase!important;color:var(--muted)!important;margin-bottom:1rem!important;}
.section-header{display:flex;align-items:center;gap:10px;padding:.75rem 1.25rem;background:var(--surf2);border:1px solid var(--border);border-radius:12px;margin:1.5rem 0 1rem;font-family:var(--head);font-size:1rem;font-weight:700;}
.num-badge{background:rgba(255,59,92,.2);color:#FF3B5C;border-radius:6px;padding:2px 8px;font-family:var(--mono);font-size:.75rem;font-weight:600;}

/* Inputs */
.stTextArea textarea,.stTextInput input{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:12px!important;color:var(--text)!important;font-family:var(--body)!important;font-size:.95rem!important;}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:rgba(255,59,92,.5)!important;box-shadow:0 0 0 3px rgba(255,59,92,.1)!important;}
label[data-testid="stWidgetLabel"]>div>p{font-family:var(--head)!important;font-weight:600!important;color:var(--text)!important;font-size:.9rem!important;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#FF3B5C,#c0212e)!important;color:white!important;border:none!important;border-radius:12px!important;font-family:var(--head)!important;font-weight:700!important;font-size:1rem!important;padding:.75rem 2rem!important;letter-spacing:.03em!important;transition:all .2s!important;box-shadow:0 4px 20px rgba(255,59,92,.3)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(255,59,92,.4)!important;}
.stButton>button[kind="secondary"]{background:var(--surf2)!important;border:1px solid var(--border)!important;box-shadow:none!important;}

/* Verdict */
.verdict-fake{background:linear-gradient(135deg,rgba(255,59,92,.15),rgba(255,59,92,.05));border:2px solid #FF3B5C;border-radius:20px;padding:2.5rem;text-align:center;margin:1rem 0;animation:pulseRed 2s ease-in-out infinite;}
.verdict-real{background:linear-gradient(135deg,rgba(0,200,150,.15),rgba(0,200,150,.05));border:2px solid #00C896;border-radius:20px;padding:2.5rem;text-align:center;margin:1rem 0;animation:pulseGreen 2s ease-in-out infinite;}
@keyframes pulseRed{0%,100%{box-shadow:0 0 20px rgba(255,59,92,.2);}50%{box-shadow:0 0 40px rgba(255,59,92,.5);}}
@keyframes pulseGreen{0%,100%{box-shadow:0 0 20px rgba(0,200,150,.2);}50%{box-shadow:0 0 40px rgba(0,200,150,.5);}}

/* Metric Card */
.metric-card{background:var(--surf2);border:1px solid var(--border);border-radius:14px;padding:1.25rem;text-align:center;}
.metric-label{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-family:var(--mono);}

/* Pills */
.pill{display:inline-block;border-radius:8px;padding:4px 12px;font-size:.8rem;font-family:var(--mono);margin:3px;}
.pill-red{background:rgba(255,59,92,.12);border:1px solid rgba(255,59,92,.25);color:#FDA4AF;}
.pill-green{background:rgba(0,200,150,.12);border:1px solid rgba(0,200,150,.25);color:#6EE7B7;}
.pill-blue{background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);color:#93C5FD;}
.pill-gray{background:rgba(107,114,128,.12);border:1px solid rgba(107,114,128,.25);color:#9CA3AF;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--surf2)!important;border-radius:12px!important;padding:4px!important;border:1px solid var(--border)!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;font-family:var(--head)!important;font-weight:600!important;}
.stTabs [aria-selected="true"]{background:var(--surf)!important;color:var(--text)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem!important;}

hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--surf2);border-radius:3px;}
.stAlert{border-radius:12px!important;}
.stCheckbox span{color:var(--text)!important;font-family:var(--body)!important;}

.footer{text-align:center;padding:2rem 0 1rem;color:var(--muted);font-size:.8rem;font-family:var(--mono);border-top:1px solid var(--border);margin-top:3rem;}

/* Expander */
details summary{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:10px!important;font-family:var(--head)!important;color:var(--text)!important;padding:.75rem 1rem!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TRUSTED / FAKE DOMAINS
# ─────────────────────────────────────────────────────────────────────────────
TRUSTED_DOMAINS = {
    'reuters.com','apnews.com','bbc.com','bbc.co.uk','theguardian.com',
    'nytimes.com','washingtonpost.com','bloomberg.com','npr.org','pbs.org',
    'economist.com','politico.com','time.com','cbsnews.com','nbcnews.com',
    'cnn.com','dawn.com','geo.tv','theatlantic.com','aljazeera.com',
    'foreignpolicy.com','sciencemag.org','nature.com','abc.net.au',
}
FAKE_DOMAINS = {
    'infowars.com','naturalnews.com','breitbart.com','zerohedge.com',
    'beforeitsnews.com','worldnewsdailyreport.com','empirenews.net',
    'huzlers.com','thelastlineofdefense.org',
}
TRUSTED_KWS = ["reuters","bbc","associated press","new york times","guardian",
               "washington post","bloomberg","npr","pbs","abc","cbs","nbc",
               "time","economist","politico","geo","dawn","aljazeera"]

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    m = {"loaded": False}
    try:
        # Search BASE_DIR first (files sit next to app_final.py) then models/
        model_search_dirs = [BASE_DIR, MODELS_DIR]

        def resolve_model_path(filename):
            for folder in model_search_dirs:
                path = os.path.join(folder, filename)
                if os.path.isfile(path):
                    return path
            raise FileNotFoundError(
                f"Could not find {filename} in any of: {', '.join(model_search_dirs)}"
            )

        m["lr"]           = joblib.load(resolve_model_path("lr_model.pkl"))
        m["nb"]           = joblib.load(resolve_model_path("nb_model.pkl"))
        m["preprocessor"] = joblib.load(resolve_model_path("preprocessor.pkl"))
        m["loaded"]       = True
        m["bert"]         = None
        m["bert_error"]   = None
        bert_candidates = [
            os.path.join(BASE_DIR, "bert_model"),   # ← files are here
            os.path.join(MODELS_DIR, "bert_model"),
        ]
        bert_dir = next((d for d in bert_candidates if os.path.isdir(d)), None)
        if bert_dir is not None:
            try:
                from transformers import pipeline as hf_pipeline
                pipe = hf_pipeline(
                    "text-classification",
                    model=bert_dir,
                    tokenizer=bert_dir,
                    framework="pt",
                    device=-1,
                    local_files_only=True,
                )
                # Detect which label the model uses for FAKE.
                # Handles both LABEL_0/LABEL_1 and explicit fake/real labels.
                _labels = list(pipe.model.config.id2label.values())
                _fake_label = next(
                    (l for l in _labels if "fake" in l.lower() or l == "LABEL_1"),
                    _labels[-1],          # fallback: last label = FAKE
                )
                m["bert"] = pipe
                m["bert_fake_label"] = _fake_label
            except Exception as e:
                m["bert_error"] = str(e)
    except Exception as e:
        m["error"] = str(e)
    return m

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def check_domain(url):
    if not url or not url.strip():
        return {"domain":"N/A","verdict":"NO_URL","credibility_score":50,
                "message":"No URL provided — verify manually."}
    domain = re.sub(r"https?://(www\.)?","",url).split("/")[0].lower()
    if any(t in domain for t in TRUSTED_DOMAINS):
        return {"domain":domain,"verdict":"TRUSTED","credibility_score":95,
                "message":f"{domain} is a trusted, credible source."}
    if any(f in domain for f in FAKE_DOMAINS):
        return {"domain":domain,"verdict":"KNOWN_FAKE","credibility_score":5,
                "message":f"{domain} is a known misinformation source!"}
    if any(p in domain for p in [".com.co",".com.de","breaking","viral",
                                  "shocking","patriot","freedom","liberty"]):
        return {"domain":domain,"verdict":"SUSPICIOUS","credibility_score":20,
                "message":f"{domain} has suspicious patterns."}
    return {"domain":domain,"verdict":"UNKNOWN","credibility_score":50,
            "message":f"{domain} is not in our database — verify manually."}


def google_news_crossref(query):
    try:
        short = query.strip()[:100].replace(" ","+")
        resp  = requests.get(
            f"https://news.google.com/rss/search?q={short}&hl=en-US&gl=US&ceid=US:en",
            timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if resp.status_code != 200:
            return {"error":f"HTTP {resp.status_code}","sources":[],"trusted_count":0,"verdict":"ERROR"}
        items, sources = re.findall(r"<item>(.*?)</item>",resp.text,re.DOTALL), []
        for item in items[:5]:
            t=re.search(r"<title>(.*?)</title>",item)
            s=re.search(r"<source[^>]*>(.*?)</source>",item)
            l=re.search(r"<link>(.*?)</link>",item)
            if t:
                sources.append({"title":re.sub(r"<[^>]+>","",t.group(1)).strip(),
                                 "source":s.group(1).strip() if s else "Unknown",
                                 "link":l.group(1).strip() if l else "#"})
        tc = sum(1 for src in sources if any(k in src["source"].lower() for k in TRUSTED_KWS))
        if not sources:   v="NOT_FOUND"
        elif tc>=2:       v="WIDELY_COVERED"
        elif tc==1:       v="SOME_COVERAGE"
        else:             v="NO_TRUSTED_SOURCE"
        return {"sources":sources,"trusted_count":tc,"total_found":len(sources),"verdict":v}
    except Exception as e:
        return {"error":str(e),"sources":[],"trusted_count":0,"verdict":"ERROR"}


def google_factcheck(query, api_key):
    if not api_key: return [{"error":"No API key provided."}]
    try:
        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query":query[:120],"key":api_key,"pageSize":5}, timeout=8)
        if resp.status_code==403: return [{"error":"API key rejected."}]
        if resp.status_code!=200: return [{"error":f"API error {resp.status_code}"}]
        claims = resp.json().get("claims",[])
        if not claims: return []
        return [{"text":c.get("text",""),"claimant":c.get("claimant","Unknown"),
                 "date":c.get("claimDate","")[:10] if c.get("claimDate") else "N/A",
                 "rating":c.get("claimReview",[{}])[0].get("textualRating","N/A"),
                 "publisher":c.get("claimReview",[{}])[0].get("publisher",{}).get("name","N/A"),
                 "url":c.get("claimReview",[{}])[0].get("url","#")} for c in claims]
    except Exception as e:
        return [{"error":str(e)}]

# ─────────────────────────────────────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def gauge_chart(value, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value*100,1),
        number={"suffix":"%","font":{"size":30,"color":TEXT,"family":"Syne"}},
        title={"text":label,"font":{"size":12,"color":MUTED,"family":"DM Sans"}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":9,"color":MUTED},"tickcolor":MUTED},
            "bar":{"color":color,"thickness":0.75},
            "bgcolor":SURF2,"borderwidth":0,
            "steps":[{"range":[0,50],"color":"rgba(255,255,255,.03)"},
                     {"range":[50,100],"color":"rgba(255,255,255,.06)"}],
        },
    ))
    fig.update_layout(height=210,margin=dict(t=45,b=5,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font={"family":"DM Sans"})
    return fig

def comparison_bar(names, fake_p, real_p):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Fake Probability",x=names,y=[p*100 for p in fake_p],
                         marker_color=RED,marker_line_width=0,
                         text=[f"{p:.1%}" for p in fake_p],textposition="outside",
                         textfont={"color":TEXT,"size":11,"family":"JetBrains Mono"}))
    fig.add_trace(go.Bar(name="Real Probability",x=names,y=[p*100 for p in real_p],
                         marker_color=GREEN,marker_line_width=0,
                         text=[f"{p:.1%}" for p in real_p],textposition="outside",
                         textfont={"color":TEXT,"size":11,"family":"JetBrains Mono"}))
    fig.update_layout(
        barmode="group",height=320,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=SURF,
        legend=dict(font=dict(color=TEXT,family="DM Sans"),bgcolor="rgba(0,0,0,0)",
                    orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
        margin=dict(t=40,b=20,l=20,r=20),
        xaxis=dict(tickfont=dict(color=TEXT,family="Syne",size=13),gridcolor="#1E2535"),
        yaxis=dict(tickfont=dict(color=MUTED,size=10),gridcolor="#1E2535",
                   range=[0,120],ticksuffix="%"),
        font={"family":"DM Sans","color":TEXT},
        title=dict(text="Model Probability Comparison",
                   font=dict(color=TEXT,family="Syne",size=14),x=0),
    )
    return fig

def domain_gauge(score, verdict):
    c = GREEN if score>=70 else ("#F59E0B" if score>=40 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference":50,"increasing":{"color":GREEN},"decreasing":{"color":RED},
               "font":{"size":13}},
        number={"suffix":"/100","font":{"size":38,"color":c,"family":"Syne"}},
        title={"text":f"Domain Credibility<br><span style='font-size:11px;color:{MUTED}'>{verdict}</span>",
               "font":{"size":13,"color":TEXT,"family":"Syne"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"color":MUTED,"size":9},"tickcolor":MUTED},
               "bar":{"color":c,"thickness":0.8},"bgcolor":SURF2,"borderwidth":0,
               "steps":[{"range":[0,30],"color":"rgba(255,59,92,.08)"},
                        {"range":[30,70],"color":"rgba(245,158,11,.06)"},
                        {"range":[70,100],"color":"rgba(0,200,150,.08)"}]},
    ))
    fig.update_layout(height=270,margin=dict(t=65,b=10,l=30,r=30),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig

def vote_donut(fake_v, real_v):
    winner = "FAKE" if fake_v > real_v else "REAL"
    wcolor = RED if winner=="FAKE" else GREEN
    fig = go.Figure(go.Pie(
        labels=["FAKE signals","REAL signals"],values=[fake_v,real_v],hole=0.65,
        marker=dict(colors=[RED,GREEN],line=dict(color=DARK,width=3)),
        textinfo="label+percent",textfont=dict(color=TEXT,family="Syne",size=12),
    ))
    fig.add_annotation(text=f"<b>{winner}</b>",x=0.5,y=0.5,
                       font=dict(size=24,color=wcolor,family="Syne"),showarrow=False)
    fig.update_layout(
        height=300,showlegend=True,
        legend=dict(font=dict(color=TEXT,family="DM Sans"),bgcolor="rgba(0,0,0,0)",
                    orientation="h",yanchor="top",y=-0.05),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20,b=20,l=20,r=20),
        title=dict(text="Signal Vote Summary",font=dict(color=TEXT,family="Syne",size=13),x=0),
    )
    return fig

def lime_chart(word_weights, verdict):
    words  = [w[0] for w in word_weights]
    scores = [w[1] for w in word_weights]
    colors = [RED if s>0 else GREEN for s in scores]
    fig = go.Figure(go.Bar(
        x=scores,y=words,orientation="h",marker_color=colors,marker_line_width=0,
        text=[f"{s:+.4f}" for s in scores],textposition="outside",
        textfont=dict(color=TEXT,size=10,family="JetBrains Mono"),
    ))
    fig.add_vline(x=0,line_color=TEXT,line_width=1,opacity=0.4)
    fig.update_layout(
        height=440,
        title=dict(text=f"LIME Word Contributions — Verdict: <b>{verdict}</b>",
                   font=dict(color=TEXT,size=14,family="Syne"),x=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=SURF,
        margin=dict(t=50,b=20,l=20,r=90),
        xaxis=dict(
            title=dict(text="← pushes REAL  |  pushes FAKE →",
                       font=dict(color=MUTED, size=11)),
            tickfont=dict(color=MUTED,size=9),
            gridcolor="#1E2535",
        ),
        yaxis=dict(tickfont=dict(color=TEXT,size=11,family="JetBrains Mono"),autorange="reversed"),
        font=dict(family="DM Sans",color=TEXT),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🛡️ AI-POWERED VERIFICATION SYSTEM</div>
  <p class="hero-title">FakeShield</p>
  <p style="font-family:'Syne',sans-serif;font-size:1.15rem;color:#9CA3AF;margin:0 0 1rem 0;">
    Fake News Detection System
  </p>
  <p class="hero-sub">
    NLP Lab Project &nbsp;·&nbsp; <b style="color:#E8ECF4;">Moaz Hassan</b> &nbsp;·&nbsp;
    Roll No: 231168 &nbsp;·&nbsp; BSAI-VI-A &nbsp;·&nbsp;
    Submitted to: <b style="color:#E8ECF4;">Ma'am Mehvish Fatima</b>
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
models = load_models()

if not models.get("loaded"):
    st.error(f"""
**❌ Models not found!**

Make sure you have a `models/` folder next to `app.py` containing:
`lr_model.pkl` · `nb_model.pkl` · `preprocessor.pkl`

First run `save_models_colab_cell.py` in Colab, then download from Google Drive.

Error: `{models.get('error','Unknown')}`
""")
    st.stop()

if models.get("bert_error"):
    st.warning(f"BERT load warning: {models['bert_error']}")

# Status bar
c1, c2, c3 = st.columns(3)
for col, lbl, ok in [
    (c1, "Logistic Regression", True),
    (c2, "Naive Bayes", True),
    (c3, "BERT", models["bert"] is not None),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{lbl}</div>
      <div style="font-family:'Syne';font-size:1.1rem;font-weight:700;
                  color:{'#00C896' if ok else '#6B7280'};margin-top:4px;">
        {'✓ Loaded' if ok else '— Not found'}
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍  Analyze Article", "📊  About & How It Works"])

# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
# ══════════════════════════════════════════════════════════════════════════════
    col_l, col_r = st.columns([3,2], gap="large")

    with col_l:
        st.markdown('<p class="card-title">📰 Article Content</p>', unsafe_allow_html=True)
        article_text = st.text_area("Article", placeholder="Paste any news article here…",
                                    height=220, label_visibility="collapsed")
    with col_r:
        st.markdown('<p class="card-title">⚙️ Options</p>', unsafe_allow_html=True)
        article_url    = st.text_input("🔗 Article URL (optional)", placeholder="https://…")
        api_key        = st.text_input("🔑 Google Fact Check API Key (optional)", type="password")
        show_lime      = st.checkbox("Show LIME word explanation", value=True)
        show_crossref  = st.checkbox("Google News cross-reference", value=True)

    analyze_btn = st.button("🔍  Analyze Article", type="primary", use_container_width=True)

    # Examples
    st.markdown('<p class="card-title" style="margin-top:1.2rem;">⚡ Quick Examples</p>',
                unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)
    EXAMPLES = {
        "fake":{"text":"SHOCKING: Secret government documents reveal NASA has been hiding proof of alien contact for decades! Deep state operatives want you to stay silent! Share before it gets deleted!","url":"https://www.infowars.com/posts/nasa-aliens"},
        "real":{"text":"The Federal Reserve held interest rates steady on Wednesday, with policymakers signaling they want more evidence that inflation is cooling before cutting borrowing costs. The decision was unanimous among voting members.","url":"https://www.reuters.com/markets/us/fed-holds-rates"},
        "geo":{"text":"China has warned the situation in the Middle East was at a critical juncture after President Trump extended a ceasefire to allow Iran more time to negotiate, AFP reports. The foreign ministry spokesman urged all parties to prevent a resumption of hostilities.","url":"https://www.dawn.com/live/iran-israel-war"},
    }
    if ex1.button("🚨 Fake Example",   use_container_width=True): st.session_state.update({"ex_t":EXAMPLES["fake"]["text"],"ex_u":EXAMPLES["fake"]["url"]}); st.rerun()
    if ex2.button("✅ Real Example",   use_container_width=True): st.session_state.update({"ex_t":EXAMPLES["real"]["text"],"ex_u":EXAMPLES["real"]["url"]}); st.rerun()
    if ex3.button("🌐 Geo-political",  use_container_width=True): st.session_state.update({"ex_t":EXAMPLES["geo"]["text"], "ex_u":EXAMPLES["geo"]["url"]});  st.rerun()

    if "ex_t" in st.session_state:
        article_text = st.session_state.pop("ex_t")
        article_url  = st.session_state.pop("ex_u","")

    # ─────────────────────────────────────────────────────────────────────────
    if analyze_btn and article_text.strip():
        preprocessor = models["preprocessor"]
        clean_text   = preprocessor.preprocess(article_text)

        if not clean_text.strip():
            st.warning("⚠️ Text too short after preprocessing. Try a longer article."); st.stop()

        st.markdown("---")

        # ── 01 ML Predictions ─────────────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="num-badge">01</span> ML Model Predictions</div>',
                    unsafe_allow_html=True)

        model_map   = {"Logistic Regression":models["lr"],"Naive Bayes":models["nb"]}
        ml_verdicts = {}
        ml_names, ml_fake_p, ml_real_p = [], [], []

        for name, mdl in model_map.items():
            label   = mdl.predict([clean_text])[0]
            proba   = mdl.predict_proba([clean_text])[0]
            verdict = "FAKE" if label==1 else "REAL"
            conf    = max(proba)
            ml_verdicts[name] = {"verdict":verdict,"confidence":conf,
                                  "fake_p":proba[1],"real_p":proba[0]}
            ml_names.append(name); ml_fake_p.append(proba[1]); ml_real_p.append(proba[0])

        bert_verdict = bert_conf = None
        if models["bert"]:
            b_out = models["bert"](article_text[:512])[0]
            _fake_lbl = models.get("bert_fake_label", "LABEL_1")
            bert_verdict = "FAKE" if b_out["label"] == _fake_lbl else "REAL"
            bert_conf    = b_out["score"]
            bfp = bert_conf if bert_verdict=="FAKE" else 1-bert_conf
            ml_names.append("BERT"); ml_fake_p.append(bfp); ml_real_p.append(1-bfp)

        # Gauge row
        gcols = st.columns(len(ml_names))
        for i,(name,_) in enumerate(model_map.items()):
            v=ml_verdicts[name]
            with gcols[i]:
                st.plotly_chart(gauge_chart(v["fake_p"],f"{name}<br>Fake Probability",
                                            RED if v["verdict"]=="FAKE" else GREEN),
                                use_container_width=True)
        if bert_verdict is not None:
            with gcols[-1]:
                bfp2=bert_conf if bert_verdict=="FAKE" else 1-bert_conf
                st.plotly_chart(gauge_chart(bfp2,"BERT<br>Fake Probability",
                                            RED if bert_verdict=="FAKE" else GREEN),
                                use_container_width=True)

        st.plotly_chart(comparison_bar(ml_names,ml_fake_p,ml_real_p), use_container_width=True)

        # ── 02 Domain ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="num-badge">02</span> Domain Credibility Check</div>',
                    unsafe_allow_html=True)
        domain_result = check_domain(article_url)
        dc1, dc2 = st.columns([1,2])
        with dc1:
            st.plotly_chart(domain_gauge(domain_result["credibility_score"],
                                         domain_result["verdict"]), use_container_width=True)
        with dc2:
            score = domain_result["credibility_score"]
            msg_fn = st.success if score>=70 else (st.error if score<40 else st.warning)
            msg_fn(f"**{domain_result['domain']}** — {domain_result['message']}")
            pclr  = "pill-green" if score>=70 else ("pill-red" if score<40 else "pill-gray")
            vclr  = ("pill-green" if domain_result["verdict"]=="TRUSTED" else
                     "pill-red"   if domain_result["verdict"] in ["KNOWN_FAKE","SUSPICIOUS"] else "pill-gray")
            st.markdown(f"""<div style="margin-top:1rem;">
              <span class="pill {pclr}">Score: {score}/100</span>
              <span class="pill {vclr}">{domain_result['verdict']}</span>
            </div>""", unsafe_allow_html=True)

        # ── 03 Google News ─────────────────────────────────────────────────────
        crossref = {"verdict":"ERROR","sources":[]}
        if show_crossref:
            st.markdown('<div class="section-header"><span class="num-badge">03</span> Google News Cross-Reference</div>',
                        unsafe_allow_html=True)
            with st.spinner("Scanning Google News…"):
                crossref = google_news_crossref(article_text[:100])

            if "error" in crossref:
                st.warning(f"⚠️ {crossref['error']}")
            else:
                vmap = {
                    "WIDELY_COVERED":    (GREEN,"✅ WIDELY COVERED",f"Found in {crossref['trusted_count']} trusted outlets"),
                    "SOME_COVERAGE":     ("#F59E0B","🟡 SOME COVERAGE",f"Found in {crossref['trusted_count']} trusted outlet(s)"),
                    "NO_TRUSTED_SOURCE": (RED,"🚨 NO TRUSTED SOURCE","Found online but NOT in any credible outlet — suspicious!"),
                    "NOT_FOUND":         (MUTED,"⚠️ NOT FOUND","Not found in Google News"),
                }
                vc,vl,vd = vmap.get(crossref["verdict"],(MUTED,"UNKNOWN",""))
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);border:1px solid {vc}40;
                            border-left:4px solid {vc};border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
                  <div style="font-family:'Syne';font-weight:700;color:{vc};font-size:1rem;">{vl}</div>
                  <div style="color:#9CA3AF;font-size:.88rem;margin-top:4px;">{vd}</div>
                </div>""", unsafe_allow_html=True)

                for src in crossref["sources"]:
                    trusted = any(k in src["source"].lower() for k in TRUSTED_KWS)
                    pc = "pill-green" if trusted else "pill-blue"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:.55rem 0;border-bottom:1px solid #1E2535;">
                      <span class="pill {pc}">{src['source']}</span>
                      <a href="{src['link']}" target="_blank"
                         style="color:#93C5FD;text-decoration:none;font-size:.87rem;">
                        {src['title'][:85]}…
                      </a>
                    </div>""", unsafe_allow_html=True)

        # ── 04 Fact Check ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="num-badge">04</span> Google Fact Check API</div>',
                    unsafe_allow_html=True)
        if api_key:
            with st.spinner("Querying fact-check database…"):
                fc_results = google_factcheck(article_text, api_key)
            if not fc_results:
                st.info("ℹ️ No fact-checks found for this article.")
            elif "error" in fc_results[0]:
                st.error(f"⚠️ {fc_results[0]['error']}")
            else:
                for fc in fc_results:
                    rl     = fc["rating"].lower()
                    icon   = "🔴" if any(w in rl for w in ["false","fake","mislead"]) else \
                             "🟢" if any(w in rl for w in ["true","correct","accurate"]) else "🟡"
                    rc     = RED if icon=="🔴" else (GREEN if icon=="🟢" else "#F59E0B")
                    st.markdown(f"""
                    <div style="background:var(--surf2);border:1px solid #1E2535;border-radius:12px;
                                padding:1rem 1.25rem;margin-bottom:.75rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'Syne';font-weight:700;color:{rc};">{icon} {fc['rating']}</span>
                        <span style="font-family:'JetBrains Mono';font-size:.75rem;color:{MUTED};">
                          {fc['publisher']} · {fc['date']}
                        </span>
                      </div>
                      <div style="color:#D1D5DB;font-size:.87rem;margin-top:.4rem;">{fc['text'][:120]}…</div>
                      <a href="{fc['url']}" target="_blank" style="font-size:.8rem;color:#60A5FA;">
                        View full fact-check →
                      </a>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);
                        border-radius:12px;padding:1rem 1.5rem;color:#93C5FD;">
              ℹ️ Paste your free <b>Google Fact Check API key</b> above to enable this feature.
              &nbsp;<a href="https://developers.google.com/fact-check/tools/api/reference/rest"
              target="_blank" style="color:#60A5FA;">Get a free key →</a>
            </div>""", unsafe_allow_html=True)

        # ── 05 LIME ────────────────────────────────────────────────────────────
        if show_lime:
            st.markdown('<div class="section-header"><span class="num-badge">05</span> LIME Explanation — Why did the model decide this?</div>',
                        unsafe_allow_html=True)
            with st.spinner("Computing word contributions…"):
                try:
                    from lime.lime_text import LimeTextExplainer
                    explainer = LimeTextExplainer(class_names=["Real","Fake"],random_state=42)
                    lime_mdl  = models["lr"]

                    def pf(texts):
                        return lime_mdl.predict_proba([preprocessor.preprocess(t) for t in texts])

                    exp          = explainer.explain_instance(article_text,pf,num_features=12,num_samples=300)
                    word_weights = exp.as_list()
                    lime_v       = "FAKE" if lime_mdl.predict([clean_text])[0]==1 else "REAL"

                    st.plotly_chart(lime_chart(word_weights,lime_v), use_container_width=True)

                    fake_ws = [(w,s) for w,s in word_weights if s>0][:5]
                    real_ws = [(w,s) for w,s in word_weights if s<0][:5]
                    max_abs  = max(abs(x[1]) for x in word_weights) or 1

                    wc1,wc2 = st.columns(2)
                    with wc1:
                        st.markdown('<p class="card-title">🔴 Pushes toward FAKE</p>', unsafe_allow_html=True)
                        for w,s in fake_ws:
                            bar=int(abs(s)/max_abs*100)
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
                              <code style="background:#1E2535;color:#FDA4AF;padding:2px 8px;
                                           border-radius:6px;font-size:.82rem;min-width:110px;">{w}</code>
                              <div style="flex:1;background:#1E2535;border-radius:4px;height:8px;">
                                <div style="background:{RED};width:{bar}%;height:8px;border-radius:4px;"></div>
                              </div>
                              <code style="color:{RED};font-size:.75rem;">+{s:.4f}</code>
                            </div>""", unsafe_allow_html=True)
                    with wc2:
                        st.markdown('<p class="card-title">🟢 Pushes toward REAL</p>', unsafe_allow_html=True)
                        for w,s in real_ws:
                            bar=int(abs(s)/max_abs*100)
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
                              <code style="background:#1E2535;color:#6EE7B7;padding:2px 8px;
                                           border-radius:6px;font-size:.82rem;min-width:110px;">{w}</code>
                              <div style="flex:1;background:#1E2535;border-radius:4px;height:8px;">
                                <div style="background:{GREEN};width:{bar}%;height:8px;border-radius:4px;"></div>
                              </div>
                              <code style="color:{GREEN};font-size:.75rem;">{s:.4f}</code>
                            </div>""", unsafe_allow_html=True)
                except ImportError:
                    st.warning("Install lime: `pip install lime`")
                except Exception as e:
                    st.warning(f"LIME error: {e}")

        # ── 06 Final Verdict ───────────────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="num-badge">06</span> Overall Verdict</div>',
                    unsafe_allow_html=True)

        fake_votes = sum(1 for v in ml_verdicts.values() if v["verdict"]=="FAKE")
        real_votes = sum(1 for v in ml_verdicts.values() if v["verdict"]=="REAL")
        if bert_verdict=="FAKE":  fake_votes+=1
        elif bert_verdict:        real_votes+=1
        if domain_result["credibility_score"]<30: fake_votes+=1
        if domain_result["credibility_score"]>70: real_votes+=1
        if "verdict" in crossref:
            if crossref["verdict"]=="WIDELY_COVERED":    real_votes+=1
            if crossref["verdict"]=="NO_TRUSTED_SOURCE": fake_votes+=1

        final  = "FAKE" if fake_votes>real_votes else "REAL"
        total  = fake_votes+real_votes

        vc1,vc2 = st.columns([3,2])
        with vc1:
            if final=="FAKE":
                st.markdown(f"""
                <div class="verdict-fake">
                  <div style="font-size:3.5rem;margin-bottom:.5rem;">🚨</div>
                  <div style="font-family:'Syne';font-size:3rem;font-weight:800;color:{RED};">FAKE NEWS</div>
                  <div style="color:#9CA3AF;margin-top:.5rem;">{fake_votes} out of {total} signals indicate FAKE</div>
                  <div style="margin-top:1rem;">
                    <span class="pill pill-red">🔴 {fake_votes} FAKE signals</span>
                    <span class="pill pill-green">🟢 {real_votes} REAL signals</span>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-real">
                  <div style="font-size:3.5rem;margin-bottom:.5rem;">✅</div>
                  <div style="font-family:'Syne';font-size:3rem;font-weight:800;color:{GREEN};">REAL NEWS</div>
                  <div style="color:#9CA3AF;margin-top:.5rem;">{real_votes} out of {total} signals indicate REAL</div>
                  <div style="margin-top:1rem;">
                    <span class="pill pill-green">🟢 {real_votes} REAL signals</span>
                    <span class="pill pill-red">🔴 {fake_votes} FAKE signals</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        with vc2:
            st.plotly_chart(vote_donut(fake_votes,real_votes), use_container_width=True)

        # Signal table
        with st.expander("📋 Detailed signal breakdown"):
            rows = [(n,v["verdict"],f"{v['confidence']:.1%}") for n,v in ml_verdicts.items()]
            if bert_verdict: rows.append(("BERT",bert_verdict,f"{bert_conf:.1%}"))
            rows.append(("Domain",domain_result["verdict"],f"{domain_result['credibility_score']}/100"))
            if "verdict" in crossref and crossref["verdict"]!="ERROR":
                rows.append(("Google News",crossref["verdict"],f"{crossref.get('total_found',0)} results"))

            st.markdown("""<table style="width:100%;border-collapse:collapse;font-family:'DM Sans';font-size:.88rem;">
              <thead><tr style="border-bottom:1px solid #1E2535;">
                <th style="padding:10px;text-align:left;color:#6B7280;">Signal</th>
                <th style="padding:10px;text-align:center;color:#6B7280;">Result</th>
                <th style="padding:10px;text-align:right;color:#6B7280;">Detail</th>
              </tr></thead><tbody>""", unsafe_allow_html=True)
            for sig,vrd,det in rows:
                vc=(RED if any(x in vrd for x in ["FAKE","SUSPICIOUS","KNOWN","NO_TRUSTED"]) else
                    GREEN if any(x in vrd for x in ["REAL","TRUSTED","WIDELY"]) else MUTED)
                st.markdown(f"""
                <tr style="border-bottom:1px solid #0F1420;">
                  <td style="padding:10px;color:{TEXT};">{sig}</td>
                  <td style="padding:10px;text-align:center;">
                    <span style="background:{vc}20;color:{vc};padding:2px 10px;border-radius:6px;
                                 font-family:'JetBrains Mono';font-size:.78rem;">{vrd}</span>
                  </td>
                  <td style="padding:10px;text-align:right;color:{MUTED};font-family:'JetBrains Mono';font-size:.82rem;">{det}</td>
                </tr>""", unsafe_allow_html=True)
            st.markdown("</tbody></table>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️ Please paste an article to analyze.")

# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
# ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🧠 How FakeShield Works</div>', unsafe_allow_html=True)

    for row in [
        [("🧹","Text Preprocessing","Reuters bias fix · stopword removal · POS tagging · WordNet lemmatization (NLTK)"),
         ("🤖","ML Models","TF-IDF vectorization + Logistic Regression & Naive Bayes trained on Kaggle + FakeNewsNet"),
         ("⚡","BERT Fine-tuned","HuggingFace bert-base-uncased fine-tuned on GPU for binary fake/real classification")],
        [("🌐","Domain Check","Credibility scoring against 40+ trusted & known-fake domain databases"),
         ("📰","Google News","Live RSS cross-reference — checks if story is covered by credible outlets"),
         ("💡","LIME Explanation","Highlights exact words that pushed the model toward FAKE or REAL")]
    ]:
        cols = st.columns(3)
        for col,(icon,title,desc) in zip(cols,row):
            col.markdown(f"""
            <div class="card" style="text-align:center;height:160px;display:flex;flex-direction:column;align-items:center;justify-content:center;">
              <div style="font-size:2.2rem;margin-bottom:.6rem;">{icon}</div>
              <div style="font-family:'Syne';font-weight:700;font-size:.95rem;margin-bottom:.4rem;">{title}</div>
              <div style="color:{MUTED};font-size:.82rem;line-height:1.4;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:2rem;">⚙️ Setup Instructions</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
      <div style="font-family:'JetBrains Mono';font-size:.85rem;line-height:2.3;color:{TEXT};">
        <span style="color:{MUTED};"># Step 1 — Install dependencies</span><br>
        <span style="color:{GREEN};">pip install</span> streamlit scikit-learn nltk lime requests transformers torch plotly<br><br>
        <span style="color:{MUTED};"># Step 2 — Folder structure</span><br>
        app.py<br>
        models/<br>
        &nbsp;&nbsp;├── lr_model.pkl<br>
        &nbsp;&nbsp;├── nb_model.pkl<br>
        &nbsp;&nbsp;├── preprocessor.pkl<br>
        &nbsp;&nbsp;└── bert_model/ &nbsp;<span style="color:{MUTED};">(optional — only if BERT was trained)</span><br><br>
        <span style="color:{MUTED};"># Step 3 — Run</span><br>
        <span style="color:{GREEN};">streamlit run</span> app.py
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(0,200,150,.08);border:1px solid rgba(0,200,150,.2);
                border-radius:12px;padding:1rem 1.5rem;margin-top:1rem;color:#6EE7B7;">
      💡 <b>Get models:</b> Run <code>save_models_colab_cell.py</code> in your Colab notebook
      after training, then download the <code>FakeNewsDetector/</code> folder from Google Drive
      and rename it to <code>models/</code>.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  FakeShield &nbsp;·&nbsp; NLP Lab Project &nbsp;·&nbsp;
  Moaz Hassan (231168) &nbsp;·&nbsp; BSAI-VI-A &nbsp;·&nbsp;
  Submitted to Ma'am Mehvish Fatima
</div>
""", unsafe_allow_html=True)
