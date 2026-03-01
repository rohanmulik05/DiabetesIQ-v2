from http.server import BaseHTTPRequestHandler
import json, os, sys, urllib.parse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# ── Fallback constants ────────────────────────────────────────
FALLBACK_MEDIANS = {
    'Glucose':       {0:107.0,1:140.0},
    'BloodPressure': {0:70.0, 1:74.0},
    'SkinThickness': {0:27.0, 1:32.0},
    'Insulin':       {0:102.5,1:169.5},
    'BMI':           {0:30.1, 1:34.3},
}
FALLBACK_INSULIN_CAP = 196.0
FALLBACK_ROBUST = {
    'Pregnancies':              (3.0,  4.0),
    'Glucose':                  (117.0,37.0),
    'BloodPressure':            (72.0, 18.0),
    'SkinThickness':            (29.0, 14.0),
    'Insulin':                  (125.0,93.75),
    'BMI':                      (32.0, 9.9),
    'DiabetesPedigreeFunction': (0.37, 0.38),
    'Age':                      (29.0, 14.0),
}

MODEL_DIR = os.path.join(ROOT, "models")
_cache = {}

def safe_load(f):
    p = os.path.join(MODEL_DIR, f)
    if not os.path.exists(p): return None
    try: return joblib.load(p)
    except: return None

def get_models():
    if _cache: return _cache
    _cache['cm']  = safe_load("clinical_model.pkl")
    _cache['cs']  = safe_load("clinical_preprocessor.pkl")
    _cache['cmd'] = safe_load("clinical_medians.pkl")
    _cache['cic'] = safe_load("clinical_insulin_cap.pkl")
    _cache['lm']  = safe_load("lifestyle_model.pkl")
    lp = safe_load("lifestyle_pipeline.pkl")
    _cache['lp']  = lp if lp else _build_pipeline()
    return _cache

LIFESTYLE_COLS = ['Gender','Polyuria','Polydipsia','sudden weight loss','weakness',
    'Polyphagia','Genital thrush','visual blurring','Itching','Irritability',
    'delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','Age']
BINARY_COLS = [c for c in LIFESTYLE_COLS if c != 'Age']

def _build_pipeline():
    rows = []
    for g in ['Male','Female']:
        for yn in ['Yes','No']:
            rows.append({'Gender':g,**{c:yn for c in BINARY_COLS if c!='Gender'},'Age':30.0})
    for age in [1.0,100.0]:
        rows.append({'Gender':'Male',**{c:'Yes' for c in BINARY_COLS if c!='Gender'},'Age':age})
    df = pd.DataFrame(rows, columns=LIFESTYLE_COLS)
    pipe = Pipeline([
        ('pre',ColumnTransformer([('cat',OneHotEncoder(drop='first'),BINARY_COLS)],remainder='passthrough')),
        ('scl',MinMaxScaler())
    ])
    pipe.fit(df)
    return pipe

NUMERIC_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']

def preprocess_clinical(raw, m):
    row = {
        'Pregnancies':float(raw['pregnancies']),
        'Glucose':float(raw['glucose']),
        'BloodPressure':float(raw['blood_pressure']),
        'SkinThickness':float(raw['skin_thickness']),
        'Insulin':float(raw['insulin']),
        'BMI':float(raw['bmi']),
        'DiabetesPedigreeFunction':float(raw['dpf']),
        'Age':float(raw['age']),
    }
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        if row[col]==0: row[col]=np.nan
    med = m['cmd'] or FALLBACK_MEDIANS
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        if np.isnan(row[col]) and col in med:
            row[col]=(med[col][0]+med[col][1])/2.0
    cap = float(m['cic']) if m['cic'] is not None else FALLBACK_INSULIN_CAP
    if row['Insulin']>cap: row['Insulin']=cap
    bmi=row['BMI']
    if   bmi<18.5:  nb='Underweight'
    elif bmi<=24.9: nb='Normal'
    elif bmi<=29.9: nb='Overweight'
    elif bmi<=34.9: nb='Obesity 1'
    elif bmi<=39.9: nb='Obesity 2'
    else:           nb='Obesity 3'
    ni='Normal' if 16<=row['Insulin']<=166 else 'Abnormal'
    g=row['Glucose']
    if   g<=70:  ng='Low'
    elif g<=99:  ng='Normal'
    elif g<=126: ng='Overweight'
    else:        ng='Secret'
    ohe=[int(nb=='Obesity 1'),int(nb=='Obesity 2'),int(nb=='Obesity 3'),
         int(nb=='Overweight'),int(nb=='Underweight'),int(ni=='Normal'),
         int(ng=='Low'),int(ng=='Normal'),int(ng=='Overweight'),int(ng=='Secret')]
    if m['cs']:
        scaled=m['cs'].transform([[row[c] for c in NUMERIC_COLS]])[0]
    else:
        scaled=np.array([(row[c]-FALLBACK_ROBUST[c][0])/FALLBACK_ROBUST[c][1] for c in NUMERIC_COLS])
    return np.concatenate([scaled,ohe]).reshape(1,-1)

BIN={0:'No',1:'Yes'}; GEN={0:'Female',1:'Male'}

def preprocess_lifestyle(raw, m):
    row={
        'Gender':GEN[int(raw['gender'])],
        'Polyuria':BIN[int(raw['polyuria'])],
        'Polydipsia':BIN[int(raw['polydipsia'])],
        'sudden weight loss':BIN[int(raw['weight_loss'])],
        'weakness':BIN[int(raw['weakness'])],
        'Polyphagia':BIN[int(raw['polyphagia'])],
        'Genital thrush':BIN[int(raw['genital_thrush'])],
        'visual blurring':BIN[int(raw['visual_blurring'])],
        'Itching':BIN[int(raw['itching'])],
        'Irritability':BIN[int(raw['irritability'])],
        'delayed healing':BIN[int(raw['delayed_healing'])],
        'partial paresis':BIN[int(raw['partial_paresis'])],
        'muscle stiffness':BIN[int(raw['muscle_stiffness'])],
        'Alopecia':BIN[int(raw['alopecia'])],
        'Obesity':BIN[int(raw['obesity'])],
        'Age':float(raw['age']),
    }
    return m['lp'].transform(pd.DataFrame([row],columns=LIFESTYLE_COLS))

def run_predict(model, X):
    pred=int(model.predict(X)[0])
    if hasattr(model,'predict_proba'):
        prob=float(model.predict_proba(X)[0][1])
    elif hasattr(model,'decision_function'):
        prob=float(1/(1+np.exp(-model.decision_function(X)[0])))
    else:
        prob=float(pred)
    return prob, pred

def read_html():
    p = os.path.join(ROOT,'templates','index.html')
    with open(p,'r',encoding='utf-8') as f: return f.read()

# ── Vercel requires BaseHTTPRequestHandler ───────────────────
class handler(BaseHTTPRequestHandler):

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == '/status':
            m = get_models()
            self.send_json(200,{
                "clinical_model_loaded":    m['cm'] is not None,
                "lifestyle_model_loaded":   m['lm'] is not None,
                "clinical_scaler_loaded":   m['cs'] is not None,
                "lifestyle_pipeline_loaded":m['lp'] is not None,
                "clinical_medians_loaded":  m['cmd'] is not None,
            })
        else:
            try:
                self.send_html(read_html())
            except Exception as e:
                self.send_json(500,{"error":str(e)})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get('Content-Length',0))
        try:
            raw = json.loads(self.rfile.read(length))
        except Exception as e:
            self.send_json(400,{"error":f"Invalid JSON: {e}"}); return

        m = get_models()

        try:
            if path == '/predict/clinical':
                if not m['cm']:
                    self.send_json(503,{"error":"clinical_model.pkl not found"}); return
                X = preprocess_clinical(raw, m)
                prob, pred = run_predict(m['cm'], X)
                self.send_json(200,{"probability":prob,"prediction":pred})

            elif path == '/predict/lifestyle':
                if not m['lm']:
                    self.send_json(503,{"error":"lifestyle_model.pkl not found"}); return
                X = preprocess_lifestyle(raw, m)
                prob, pred = run_predict(m['lm'], X)
                self.send_json(200,{"probability":prob,"prediction":pred})

            elif path == '/predict/combined':
                if not m['cm'] or not m['lm']:
                    self.send_json(503,{"error":"Both models required"}); return
                cp, cpred = run_predict(m['cm'], preprocess_clinical(raw, m))
                lp, lpred = run_predict(m['lm'], preprocess_lifestyle(raw, m))
                ep = (cp+lp)/2.0
                self.send_json(200,{
                    "clinical": {"probability":cp,"prediction":cpred},
                    "lifestyle":{"probability":lp,"prediction":lpred},
                    "combined": {"probability":ep,"prediction":int(ep>=0.5)},
                })
            else:
                self.send_json(404,{"error":"Not found"})

        except Exception as e:
            self.send_json(400,{"error":str(e)})

    def log_message(self, fmt, *args):
        pass  # suppress default request logs
