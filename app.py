# ============================================================================
# SIMULASI MONTE CARLO — ESTIMASI WAKTU PEMBANGUNAN GEDUNG FITE 5 LANTAI
# [11S1221] Pemodelan dan Simulasi (MODSIM) — Praktikum 5
# Studi Kasus: Streamlit App
# ============================================================================
# Cara menjalankan:
#   pip install streamlit plotly numpy pandas scipy
#   streamlit run app.py
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Monte Carlo — Gedung FITE 5 Lantai",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.25);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
}

.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.kpi-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.kpi-unit {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}
.kpi-accent { color: #38bdf8; }
.kpi-accent-green { color: #34d399; }
.kpi-accent-orange { color: #fb923c; }
.kpi-accent-purple { color: #a78bfa; }

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #3b82f6;
    padding-left: 0.7rem;
    margin: 1.5rem 0 1rem 0;
}

.info-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.info-card-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.6rem;
}

.prob-chip {
    display: inline-block;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.95rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px 3px;
}
.prob-danger  { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }
.prob-warning { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
.prob-ok      { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

.rec-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,78,59,0.08));
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
    font-size: 0.85rem;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

/* Main background */
.stApp {
    background: #0b1120;
}
.block-container {
    max-width: 1200px;
}

div[data-testid="stMetric"] {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.7rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. MODEL SISTEM
# ============================================================================
class ProjectStage:
    def __init__(self, name, optimistic, most_likely, pessimistic, risk_factors=None):
        self.name        = name
        self.optimistic  = optimistic
        self.most_likely = most_likely
        self.pessimistic = pessimistic
        self.risk_factors = risk_factors or {}
        self.pert_mean = (optimistic + 4 * most_likely + pessimistic) / 6
        self.pert_std  = (pessimistic - optimistic) / 6

    def simulate(self, n):
        span = self.pessimistic - self.optimistic
        if span <= 0:
            base = np.full(n, self.optimistic)
        else:
            mu, sigma = self.pert_mean, self.pert_std
            alpha = max(((mu - self.optimistic) / span) *
                        ((mu - self.optimistic) * (self.pessimistic - mu) /
                         sigma**2 - 1), 0.5)
            beta  = max(alpha * (self.pessimistic - mu) /
                        (mu - self.optimistic), 0.5)
            base  = self.optimistic + np.random.beta(alpha, beta, n) * span

        risk_impact = np.zeros(n)
        for _, info in self.risk_factors.items():
            occurs = np.random.random(n) < info['probability']
            impact = np.random.uniform(info['impact_min'], info['impact_max'], n)
            risk_impact += occurs * impact

        return np.maximum(base + risk_impact, self.optimistic * 0.8)


class MonteCarloSimulator:
    def __init__(self, stages_cfg, n_sim, seed=42):
        np.random.seed(seed)
        self.n_sim = n_sim
        self.stages = {}
        for name, cfg in stages_cfg.items():
            bp = cfg['base_params']
            self.stages[name] = ProjectStage(
                name         = name,
                optimistic   = bp['optimistic'],
                most_likely  = bp['most_likely'],
                pessimistic  = bp['pessimistic'],
                risk_factors = cfg.get('risk_factors', {})
            )

    def run(self):
        data = {name: stage.simulate(self.n_sim) for name, stage in self.stages.items()}
        df = pd.DataFrame(data)
        df['Total'] = df[list(self.stages.keys())].sum(axis=1)
        return df

    def critical_path(self, df):
        total = df['Total']
        thr   = np.percentile(total, 75)
        out   = {}
        for name in self.stages:
            corr, _ = stats.spearmanr(df[name], total)
            mask = df[name] > df[name].median()
            prob = float(np.mean(total[mask] > thr)) if mask.sum() > 0 else 0.0
            out[name] = {'probability': prob, 'correlation': corr,
                         'mean': df[name].mean(), 'contribution': df[name].mean() / total.mean() * 100}
        return pd.DataFrame(out).T

    def risk_contribution(self, df):
        risk_types = ['cuaca_buruk', 'keterlambatan_material', 'perubahan_desain', 'produktivitas_pekerja']
        contrib = {}
        for rt in risk_types:
            probs, impacts, count = [], [], 0
            for s in self.stages.values():
                if rt in s.risk_factors:
                    info = s.risk_factors[rt]
                    probs.append(info['probability'])
                    impacts.append((info['impact_min'] + info['impact_max']) / 2)
                    count += 1
            if count:
                contrib[rt] = {
                    'avg_probability': np.mean(probs),
                    'avg_impact':      np.mean(impacts),
                    'stages_affected': count,
                    'risk_index':      np.mean(probs) * np.mean(impacts) * 100
                }
        return pd.DataFrame(contrib).T


# ============================================================================
# 3. KONFIGURASI TAHAPAN DEFAULT
# ============================================================================
DEFAULT_STAGES = {
    "Perencanaan_dan_Desain": {
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
        "risk_factors": {
            "perubahan_desain": {"probability": 0.45, "impact_min": 0.3, "impact_max": 1.2},
            "cuaca_buruk":      {"probability": 0.10, "impact_min": 0.1, "impact_max": 0.3},
        }
    },
    "Persiapan_Lahan_dan_Pondasi": {
        "base_params": {"optimistic": 1.0, "most_likely": 1.5, "pessimistic": 3.0},
        "risk_factors": {
            "cuaca_buruk":          {"probability": 0.50, "impact_min": 0.2, "impact_max": 0.8},
            "produktivitas_pekerja":{"probability": 0.30, "impact_min": 0.1, "impact_max": 0.5},
        }
    },
    "Struktur_Bangunan_5_Lantai": {
        "base_params": {"optimistic": 4.0, "most_likely": 6.0, "pessimistic": 9.0},
        "risk_factors": {
            "cuaca_buruk":           {"probability": 0.55, "impact_min": 0.5, "impact_max": 1.5},
            "keterlambatan_material":{"probability": 0.40, "impact_min": 0.3, "impact_max": 1.0},
            "produktivitas_pekerja": {"probability": 0.35, "impact_min": 0.2, "impact_max": 0.8},
        }
    },
    "Instalasi_MEP": {
        "base_params": {"optimistic": 2.0, "most_likely": 3.0, "pessimistic": 5.0},
        "risk_factors": {
            "keterlambatan_material":{"probability": 0.45, "impact_min": 0.3, "impact_max": 1.2},
            "produktivitas_pekerja": {"probability": 0.30, "impact_min": 0.1, "impact_max": 0.6},
        }
    },
    "Instalasi_Lab_Komputer_Elektro": {
        "base_params": {"optimistic": 1.0, "most_likely": 1.5, "pessimistic": 2.5},
        "risk_factors": {
            "keterlambatan_material":{"probability": 0.40, "impact_min": 0.2, "impact_max": 0.8},
            "perubahan_desain":      {"probability": 0.25, "impact_min": 0.1, "impact_max": 0.5},
        }
    },
    "Instalasi_Lab_VR_AR_Game": {
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.5},
        "risk_factors": {
            "keterlambatan_material":{"probability": 0.65, "impact_min": 0.5, "impact_max": 2.0},
            "perubahan_desain":      {"probability": 0.40, "impact_min": 0.3, "impact_max": 1.0},
        }
    },
    "Instalasi_Lab_Mobile": {
        "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 1.8},
        "risk_factors": {
            "keterlambatan_material":{"probability": 0.30, "impact_min": 0.1, "impact_max": 0.4},
        }
    },
    "Finishing_Interior_Furnitur": {
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
        "risk_factors": {
            "cuaca_buruk":           {"probability": 0.25, "impact_min": 0.1, "impact_max": 0.4},
            "keterlambatan_material":{"probability": 0.35, "impact_min": 0.2, "impact_max": 0.8},
            "produktivitas_pekerja": {"probability": 0.30, "impact_min": 0.1, "impact_max": 0.5},
        }
    },
    "Pengujian_dan_Commissioning": {
        "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 2.0},
        "risk_factors": {
            "perubahan_desain":{"probability": 0.35, "impact_min": 0.2, "impact_max": 0.8},
        }
    },
    "Administrasi_Serah_Terima": {
        "base_params": {"optimistic": 0.5, "most_likely": 0.8, "pessimistic": 1.5},
        "risk_factors": {
            "perubahan_desain":{"probability": 0.20, "impact_min": 0.1, "impact_max": 0.4},
        }
    },
}

RISK_LABELS = {
    'cuaca_buruk':            'Cuaca Buruk',
    'keterlambatan_material': 'Keterlambatan Material',
    'perubahan_desain':       'Perubahan Desain',
    'produktivitas_pekerja':  'Produktivitas Pekerja',
}

PLOTLY_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    font=dict(family='Plus Jakarta Sans', color='#cbd5e1', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.08)'),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ============================================================================
# 4. SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🏗️ Monte Carlo FITE")
    st.markdown("---")
    st.markdown("### ⚙️ Parameter Simulasi")

    n_sim = st.slider("Jumlah Iterasi", 5_000, 50_000, 20_000, 1_000,
                      help="Lebih banyak = lebih akurat, tapi lebih lambat")
    seed  = st.number_input("Random Seed", 0, 9999, 42, 1)

    st.markdown("---")
    st.markdown("### 📋 Edit Durasi Tahapan")

    stages_cfg = {}
    for stage_name, cfg in DEFAULT_STAGES.items():
        label = stage_name.replace('_', ' ')
        with st.expander(f"📌 {label}", expanded=False):
            bp = cfg['base_params']
            opt = st.number_input("Optimistik (bln)", 0.1, 24.0, float(bp['optimistic']), 0.1, key=f"o_{stage_name}")
            ml  = st.number_input("Most Likely (bln)", 0.1, 24.0, float(bp['most_likely']),  0.1, key=f"m_{stage_name}")
            pes = st.number_input("Pesimistik (bln)",  0.1, 36.0, float(bp['pessimistic']), 0.1, key=f"p_{stage_name}")
            stages_cfg[stage_name] = {
                'base_params':  {'optimistic': opt, 'most_likely': ml, 'pessimistic': pes},
                'risk_factors': cfg.get('risk_factors', {})
            }

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)

    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; margin-top:1rem; line-height:1.6;">
    <b>MODSIM Praktikum 5</b><br>
    Monte Carlo Simulation<br>
    Studi Kasus: Gedung FITE 5 Lantai
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 5. HERO HEADER
# ============================================================================
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">📊 MODSIM · Praktikum 5 · Monte Carlo Simulation</div>
  <div class="hero-title">🏗️ Estimasi Waktu Pembangunan<br>Gedung FITE 5 Lantai</div>
  <div class="hero-sub">Simulasi probabilistik berbasis Monte Carlo untuk memodelkan ketidakpastian proyek konstruksi — analisis risiko, critical path, dan rekomendasi resource.</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# 6. STATE & SIMULASI
# ============================================================================
if 'df' not in st.session_state:
    st.session_state.df  = None
    st.session_state.sim = None

if run_btn:
    with st.spinner("⏳ Menjalankan simulasi Monte Carlo..."):
        sim = MonteCarloSimulator(stages_cfg, n_sim, seed)
        df  = sim.run()
        st.session_state.df  = df
        st.session_state.sim = sim
    st.success(f"✅ Selesai! {n_sim:,} skenario berhasil disimulasikan.")

# ─── Belum dijalankan ────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div style="text-align:center; padding:3rem; background:rgba(30,41,59,0.4);
         border:1px solid rgba(255,255,255,0.05); border-radius:16px; margin-top:2rem;">
        <div style="font-size:3rem;">🚀</div>
        <h3 style="color:#e2e8f0; margin:0.5rem 0;">Siap Memulai Simulasi?</h3>
        <p style="color:#64748b;">Atur parameter di sidebar kiri, lalu klik <b style="color:#38bdf8">Jalankan Simulasi</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Konfigurasi Tahapan Proyek</div>', unsafe_allow_html=True)
    rows = []
    for name, cfg in DEFAULT_STAGES.items():
        bp = cfg['base_params']
        rows.append({'Tahapan': name.replace('_',' '),
                     'Optimistik (bln)': bp['optimistic'],
                     'Most Likely (bln)': bp['most_likely'],
                     'Pesimistik (bln)': bp['pessimistic']})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.stop()


# ─── Ada hasil simulasi ──────────────────────────────────────────────────────
df  = st.session_state.df
sim = st.session_state.sim
total = df['Total']
stage_names = list(sim.stages.keys())

mean_t  = total.mean()
med_t   = total.median()
std_t   = total.std()
ci_80   = np.percentile(total, [10, 90])
ci_95   = np.percentile(total, [2.5, 97.5])
p16     = np.mean(total <= 16) * 100
p20     = np.mean(total <= 20) * 100
p24     = np.mean(total <= 24) * 100


# ============================================================================
# 7. KPI ROW
# ============================================================================
st.markdown('<div class="section-title">📈 Statistik Utama Proyek</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Rata-rata Durasi</div>
      <div class="kpi-value kpi-accent">{mean_t:.1f}</div>
      <div class="kpi-unit">bulan</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Median</div>
      <div class="kpi-value kpi-accent-green">{med_t:.1f}</div>
      <div class="kpi-unit">bulan</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Std Deviasi</div>
      <div class="kpi-value kpi-accent-orange">{std_t:.1f}</div>
      <div class="kpi-unit">bulan</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">80% CI</div>
      <div class="kpi-value kpi-accent-purple" style="font-size:1.2rem">{ci_80[0]:.1f}–{ci_80[1]:.1f}</div>
      <div class="kpi-unit">bulan</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">95% CI</div>
      <div class="kpi-value" style="font-size:1.2rem; color:#f1f5f9">{ci_95[0]:.1f}–{ci_95[1]:.1f}</div>
      <div class="kpi-unit">bulan</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# 8. TABS UTAMA
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribusi & Deadline",
    "🔴 Critical Path",
    "⚠️ Analisis Risiko",
    "🔧 Analisis Resource",
    "📋 Laporan Lengkap"
])


# ─── TAB 1: Distribusi & Deadline ────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">📊 Distribusi Durasi Total Proyek</div>', unsafe_allow_html=True)

        kde_x  = np.linspace(total.min(), total.max(), 400)
        kde_fn = stats.gaussian_kde(total)

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=total, nbinsx=60, name='Simulasi', histnorm='probability density',
            marker_color='rgba(56,189,248,0.5)', marker_line_color='rgba(56,189,248,0.9)',
            marker_line_width=0.4
        ))
        fig1.add_trace(go.Scatter(
            x=kde_x, y=kde_fn(kde_x), mode='lines', name='KDE',
            line=dict(color='#38bdf8', width=2.5)
        ))
        # CI spans
        for span, col, lbl in [
            ([ci_95[0], ci_95[1]], 'rgba(251,146,60,0.08)', '95% CI'),
            ([ci_80[0], ci_80[1]], 'rgba(250,204,21,0.12)',  '80% CI'),
        ]:
            fig1.add_vrect(x0=span[0], x1=span[1], fillcolor=col, line_width=0,
                           annotation_text=lbl, annotation_font_size=10)
        fig1.add_vline(x=mean_t, line_dash='dash', line_color='#f87171',
                       annotation_text=f'Mean {mean_t:.1f}', annotation_font_size=10)
        fig1.add_vline(x=med_t,  line_dash='dot',  line_color='#34d399',
                       annotation_text=f'Median {med_t:.1f}', annotation_font_size=10)

        for dl, col, lbl in [(16,'#f87171','16 bln'),(20,'#fb923c','20 bln'),(24,'#34d399','24 bln')]:
            prob = np.mean(total <= dl)*100
            fig1.add_vline(x=dl, line_color=col, line_width=1.2, line_dash='dashdot',
                           annotation_text=f'{lbl} ({prob:.0f}%)', annotation_font_size=9)

        fig1.update_layout(
            **PLOTLY_DARK,
            title=dict(text='Histogram + KDE Durasi Total', font=dict(size=13), x=0),
            xaxis_title='Durasi (Bulan)', yaxis_title='Densitas Probabilitas',
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            height=380
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">🎯 Kurva Probabilitas Penyelesaian</div>', unsafe_allow_html=True)

        dls   = np.arange(8, 42, 0.5)
        probs = [np.mean(total <= d) for d in dls]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dls, y=probs, mode='lines', name='P(Selesai)',
            line=dict(color='#a78bfa', width=3),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.1)'
        ))
        for ref, col, lbl in [(0.5,'#f87171','50%'),(0.80,'#34d399','80%'),(0.95,'#38bdf8','95%')]:
            fig2.add_hline(y=ref, line_dash='dash', line_color=col,
                           annotation_text=lbl, annotation_position='right')
        for dl, col in [(16,'#f87171'),(20,'#fb923c'),(24,'#34d399')]:
            p = np.mean(total <= dl)
            fig2.add_trace(go.Scatter(
                x=[dl], y=[p], mode='markers+text',
                marker=dict(size=11, color=col, line=dict(color='white', width=1.5)),
                text=[f'{p:.0%}'], textposition='top center',
                textfont=dict(size=10, color=col), showlegend=False
            ))
        fig2.update_layout(
            **PLOTLY_DARK,
            title=dict(text='Kurva CDF Penyelesaian Proyek', font=dict(size=13), x=0),
            xaxis_title='Deadline (Bulan)', yaxis_title='Probabilitas',
            yaxis=dict(tickformat='.0%', range=[-0.03, 1.08],
                       gridcolor='rgba(255,255,255,0.05)'),
            height=380
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Deadline chips
    st.markdown('<div class="section-title">📅 Probabilitas per Deadline</div>', unsafe_allow_html=True)
    dl_cols = st.columns(7)
    for i, dl in enumerate([12,14,16,18,20,22,24]):
        p = np.mean(total <= dl)*100
        chip_cls = 'prob-ok' if p >= 70 else ('prob-warning' if p >= 35 else 'prob-danger')
        with dl_cols[i]:
            st.markdown(f"""
            <div style="text-align:center;">
              <div style="font-size:0.7rem;color:#64748b;font-weight:600;margin-bottom:4px">{dl} BLN</div>
              <div class="prob-chip {chip_cls}">{p:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Boxplot per tahapan
    st.markdown('<div class="section-title">📦 Distribusi Durasi per Tahapan</div>', unsafe_allow_html=True)
    fig_box = go.Figure()
    palette = px.colors.qualitative.Bold
    for i, s in enumerate(stage_names):
        fig_box.add_trace(go.Box(
            y=df[s], name=s.replace('_','\n'),
            marker_color=palette[i % len(palette)],
            boxmean='sd', boxpoints='outliers',
            marker=dict(size=3, opacity=0.4),
            line=dict(width=1.5)
        ))
    fig_box.update_layout(
        **PLOTLY_DARK,
        title=dict(text='Box-Whisker Plot Durasi Tiap Tahapan', font=dict(size=13), x=0),
        yaxis_title='Durasi (Bulan)', showlegend=False, height=420,
        xaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ─── TAB 2: Critical Path ─────────────────────────────────────────────────────
with tab2:
    cp = sim.critical_path(df).sort_values('probability', ascending=True)

    st.markdown('<div class="section-title">🔴 Analisis Jalur Kritis (Critical Path)</div>', unsafe_allow_html=True)

    bar_colors = ['#ef4444' if p > 0.55 else ('#fb923c' if p > 0.38 else '#34d399')
                  for p in cp['probability']]

    fig_cp = go.Figure()
    fig_cp.add_trace(go.Bar(
        y=[n.replace('_','\n') for n in cp.index],
        x=cp['probability'],
        orientation='h',
        marker_color=bar_colors,
        text=[f'{p:.1%}' for p in cp['probability']],
        textposition='outside',
        textfont=dict(size=11, color='#e2e8f0')
    ))
    fig_cp.add_vline(x=0.55, line_dash='dash', line_color='#ef4444',
                     annotation_text='Sangat Kritis 55%', annotation_font_size=10)
    fig_cp.add_vline(x=0.38, line_dash='dash', line_color='#fb923c',
                     annotation_text='Kritis 38%', annotation_font_size=10)
    fig_cp.update_layout(
        **PLOTLY_DARK,
        title=dict(text='Probabilitas Menjadi Critical Path', font=dict(size=13), x=0),
        xaxis_title='Probabilitas', xaxis=dict(tickformat='.0%', range=[0,1.15],
                                                gridcolor='rgba(255,255,255,0.05)'),
        height=480,
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig_cp, use_container_width=True)

    # Legend + top 3
    l1, l2, l3 = st.columns(3)
    top3 = cp.sort_values('probability', ascending=False).head(3)
    for i, (col, (name, row)) in enumerate(zip([l1,l2,l3], top3.iterrows())):
        cls = 'prob-danger' if row['probability']>0.55 else ('prob-warning' if row['probability']>0.38 else 'prob-ok')
        with col:
            st.markdown(f"""<div class="info-card">
              <div class="info-card-title">#{i+1} Paling Kritis</div>
              <b style="color:#e2e8f0">{name.replace('_',' ')}</b><br>
              <span class="prob-chip {cls}">{row['probability']:.1%}</span><br>
              <small style="color:#64748b">Kontribusi: {row['contribution']:.1f}% | Korelasi: {row['correlation']:.3f}</small>
            </div>""", unsafe_allow_html=True)

    # Tabel detail
    st.markdown('<div class="section-title">📋 Tabel Detail Critical Path</div>', unsafe_allow_html=True)
    cp_show = cp.sort_values('probability', ascending=False).copy()
    cp_show['Status'] = cp_show['probability'].apply(
        lambda p: '🔴 Sangat Kritis' if p>0.55 else ('🟠 Kritis' if p>0.38 else '🟢 Normal'))
    cp_show = cp_show[['probability','correlation','mean','contribution','Status']]
    cp_show.columns = ['P. Kritis','Korelasi','Rata-rata (bln)','Kontribusi (%)','Status']
    cp_show = cp_show.round(3)
    cp_show.index = [n.replace('_',' ') for n in cp_show.index]
    st.dataframe(cp_show, use_container_width=True)


# ─── TAB 3: Analisis Risiko ──────────────────────────────────────────────────
with tab3:
    rc = sim.risk_contribution(df)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown('<div class="section-title">⚠️ Indeks Kontribusi Risiko</div>', unsafe_allow_html=True)
        rc_sorted = rc.sort_values('risk_index', ascending=False)
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            x=[RISK_LABELS.get(n, n) for n in rc_sorted.index],
            y=rc_sorted['risk_index'],
            marker_color=['#ef4444','#fb923c','#facc15','#34d399'][:len(rc_sorted)],
            text=[f'{v:.1f}' for v in rc_sorted['risk_index']],
            textposition='outside', textfont=dict(size=12, color='#e2e8f0')
        ))
        fig_risk.update_layout(
            **PLOTLY_DARK,
            title=dict(text='Risk Index per Faktor', font=dict(size=13), x=0),
            yaxis_title='Risk Index', height=350
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">🗺️ Matriks Korelasi Tahapan</div>', unsafe_allow_html=True)
        corr_m = df[stage_names].corr()
        fig_hm = go.Figure(go.Heatmap(
            z=corr_m.values,
            x=[n.replace('_','\n') for n in corr_m.columns],
            y=[n.replace('_','\n') for n in corr_m.index],
            colorscale='RdBu', zmid=0,
            text=np.round(corr_m.values,2), texttemplate='%{text}',
            textfont=dict(size=8), hoverongaps=False,
            colorbar=dict(tickfont=dict(color='#cbd5e1'))
        ))
        fig_hm.update_layout(
            **PLOTLY_DARK,
            title=dict(text='Korelasi Antar Tahapan', font=dict(size=13), x=0),
            height=350,
            xaxis=dict(tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=7))
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # Detail risiko
    st.markdown('<div class="section-title">📋 Detail Faktor Risiko</div>', unsafe_allow_html=True)
    rc_show = rc.copy()
    rc_show.index = [RISK_LABELS.get(n, n) for n in rc_show.index]
    rc_show.columns = ['Rata-rata Probabilitas','Rata-rata Dampak (bln)','Tahapan Terdampak','Risk Index']
    rc_show = rc_show.round(3)
    st.dataframe(rc_show, use_container_width=True)


# ─── TAB 4: Analisis Resource ────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔧 Analisis Penambahan Resource</div>', unsafe_allow_html=True)

    NILAI_WAKTU = 150_000_000
    RESOURCE_SCENARIOS = [
        {'stage':'Struktur_Bangunan_5_Lantai',   'resource':'Alat Berat Tambahan',  'qty':2,  'eff':0.20, 'dur':6,   'biaya_satuan':25_000_000},
        {'stage':'Struktur_Bangunan_5_Lantai',   'resource':'Pekerja Khusus',        'qty':10, 'eff':0.15, 'dur':6,   'biaya_satuan':8_000_000},
        {'stage':'Instalasi_Lab_VR_AR_Game',     'resource':'Insinyur Spesialis',    'qty':3,  'eff':0.25, 'dur':3,   'biaya_satuan':20_000_000},
        {'stage':'Instalasi_MEP',                'resource':'Insinyur MEP',          'qty':2,  'eff':0.22, 'dur':3,   'biaya_satuan':20_000_000},
        {'stage':'Finishing_Interior_Furnitur',  'resource':'Pekerja Finishing',     'qty':8,  'eff':0.18, 'dur':2.5, 'biaya_satuan':8_000_000},
    ]

    sc_results = []
    for sc in RESOURCE_SCENARIOS:
        rn = df.copy()
        factor = min(sc['eff'] * sc['qty'], 0.55)
        rn[sc['stage']] = rn[sc['stage']] * (1 - factor)
        rn['Total']     = rn[stage_names].sum(axis=1)
        saved   = total.mean() - rn['Total'].mean()
        cost    = sc['biaya_satuan'] * sc['qty'] * sc['dur']
        benefit = saved * NILAI_WAKTU
        roi     = (benefit - cost) / cost * 100 if cost > 0 else 0
        sc_results.append({**sc, 'saved': saved, 'cost': cost, 'roi': roi, 'rn': rn})

    # Charts row
    c_r1, c_r2 = st.columns(2)

    with c_r1:
        labels   = [f"S{i+1}: {s['resource']}" for i,s in enumerate(RESOURCE_SCENARIOS)]
        saved_v  = [s['saved'] for s in sc_results]
        fig_sv = go.Figure(go.Bar(
            y=labels, x=saved_v, orientation='h',
            marker_color='rgba(52,211,153,0.7)',
            marker_line_color='#34d399', marker_line_width=1.2,
            text=[f'{v:.2f} bln' for v in saved_v], textposition='outside',
            textfont=dict(color='#e2e8f0', size=10)
        ))
        fig_sv.update_layout(**PLOTLY_DARK,
            title=dict(text='Penghematan Durasi per Skenario', font=dict(size=13), x=0),
            xaxis_title='Pengurangan (Bulan)', height=320, yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_sv, use_container_width=True)

    with c_r2:
        roi_v   = [s['roi'] for s in sc_results]
        r_colors = ['#34d399' if r>0 else '#ef4444' for r in roi_v]
        fig_roi = go.Figure(go.Bar(
            x=[f'S{i+1}' for i in range(len(sc_results))], y=roi_v,
            marker_color=r_colors, marker_line_color='rgba(255,255,255,0.2)',
            text=[f'{r:.0f}%' for r in roi_v], textposition='outside',
            textfont=dict(color='#e2e8f0', size=11)
        ))
        fig_roi.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_width=1)
        fig_roi.update_layout(**PLOTLY_DARK,
            title=dict(text='Return on Investment (ROI)', font=dict(size=13), x=0),
            yaxis_title='ROI (%)', height=320)
        st.plotly_chart(fig_roi, use_container_width=True)

    # Cost-Benefit
    cost_v    = [s['cost']/1e6    for s in sc_results]
    benefit_v = [s['saved']*NILAI_WAKTU/1e6 for s in sc_results]
    fig_cb = go.Figure()
    fig_cb.add_trace(go.Bar(name='Biaya (Juta Rp)',    x=[f'S{i+1}' for i in range(5)], y=cost_v,
                            marker_color='rgba(251,146,60,0.7)'))
    fig_cb.add_trace(go.Bar(name='Net Benefit (Juta Rp)', x=[f'S{i+1}' for i in range(5)], y=benefit_v,
                            marker_color='rgba(56,189,248,0.7)'))
    fig_cb.update_layout(**PLOTLY_DARK,
        title=dict(text='Analisis Cost–Benefit per Skenario', font=dict(size=13), x=0),
        barmode='group', yaxis_title='Juta Rupiah', height=320,
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_cb, use_container_width=True)

    # Probabilitas deadline setelah resource
    st.markdown('<div class="section-title">📈 Peningkatan Probabilitas Deadline</div>', unsafe_allow_html=True)
    dls_cmp = [16, 20, 24]
    fig_dl = go.Figure()
    base_p = [np.mean(total<=d)*100 for d in dls_cmp]
    fig_dl.add_trace(go.Bar(name='Baseline', x=[f'{d} bln' for d in dls_cmp], y=base_p,
                            marker_color='rgba(100,116,139,0.7)'))
    colors5 = ['#38bdf8','#34d399','#fb923c','#a78bfa','#f87171']
    for i, sc in enumerate(sc_results):
        d_col = sc['rn']['Total']
        sc_p  = [np.mean(d_col<=d)*100 for d in dls_cmp]
        fig_dl.add_trace(go.Bar(name=f"S{i+1}: {sc['resource']}",
                                x=[f'{d} bln' for d in dls_cmp], y=sc_p,
                                marker_color=colors5[i]))
    fig_dl.update_layout(**PLOTLY_DARK,
        title=dict(text='Perbandingan Probabilitas Penyelesaian: Baseline vs Skenario', font=dict(size=13), x=0),
        barmode='group', yaxis_title='Probabilitas (%)', yaxis=dict(range=[0,110]),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)), height=380)
    st.plotly_chart(fig_dl, use_container_width=True)

    # Ringkasan tabel
    st.markdown('<div class="section-title">📋 Ringkasan Skenario Resource</div>', unsafe_allow_html=True)
    tbl = []
    for i, sc in enumerate(sc_results):
        tbl.append({
            'No': f'S{i+1}',
            'Tahapan': sc['stage'].replace('_',' '),
            'Resource': sc['resource'],
            'Jumlah': sc['qty'],
            'Dur. (bln)': sc['dur'],
            'Hemat (bln)': round(sc['saved'],2),
            'Biaya (Jt)': round(sc['cost']/1e6,1),
            'ROI (%)': round(sc['roi'],1),
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


# ─── TAB 5: Laporan Lengkap ──────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">📋 Laporan Statistik Lengkap</div>', unsafe_allow_html=True)

    # Jawaban 1
    with st.expander("✅ Jawaban 1 — Berapa Total Waktu yang Dibutuhkan?", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="info-card">
            <div class="info-card-title">Estimasi Durasi Total Proyek</div>
            <b>Rata-rata :</b> {mean_t:.2f} bulan<br>
            <b>Median    :</b> {med_t:.2f} bulan<br>
            <b>Std Dev   :</b> {std_t:.2f} bulan<br>
            <b>Minimum   :</b> {total.min():.2f} bulan<br>
            <b>Maximum   :</b> {total.max():.2f} bulan<br>
            <b>Skewness  :</b> {stats.skew(total):.3f}<br>
            <b>Kurtosis  :</b> {stats.kurtosis(total):.3f}
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="info-card">
            <div class="info-card-title">Confidence Intervals</div>
            <b>80% CI :</b> [{ci_80[0]:.2f}, {ci_80[1]:.2f}] bulan<br>
            <b>90% CI :</b> [{np.percentile(total,5):.2f}, {np.percentile(total,95):.2f}] bulan<br>
            <b>95% CI :</b> [{ci_95[0]:.2f}, {ci_95[1]:.2f}] bulan<br><br>
            <b>Kesimpulan:</b><br>
            Proyek diperkirakan selesai dalam <b>{mean_t:.1f} ± {std_t:.1f} bulan</b>.
            Rentang 80% skenario: <b>{ci_80[0]:.1f}–{ci_80[1]:.1f} bulan</b>.
            </div>""", unsafe_allow_html=True)

        rows = []
        for s in stage_names:
            col = df[s]
            rows.append({'Tahapan': s.replace('_',' '),
                         'Mean': round(col.mean(),2), 'Std': round(col.std(),2),
                         'Min': round(col.min(),2),   'Max': round(col.max(),2)})
        rows.append({'Tahapan':'TOTAL',
                     'Mean':round(mean_t,2),'Std':round(std_t,2),
                     'Min':round(total.min(),2),'Max':round(total.max(),2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Jawaban 2
    with st.expander("✅ Jawaban 2 — Risiko Keterlambatan", expanded=False):
        baseline = sum(s.pert_mean for s in sim.stages.values())
        added    = mean_t - baseline
        st.markdown(f"""
        <div class="info-card">
        <div class="info-card-title">Dampak Risiko terhadap Durasi</div>
        <b>Baseline (tanpa risiko) :</b> {baseline:.2f} bulan<br>
        <b>Rata-rata simulasi       :</b> {mean_t:.2f} bulan<br>
        <b>Penambahan akibat risiko :</b> +{added:.2f} bulan<br>
        <b>Value at Risk 90% (P90)  :</b> {np.percentile(total,90):.2f} bulan
        </div>""", unsafe_allow_html=True)
        rc2 = sim.risk_contribution(df)
        rc2.index = [RISK_LABELS.get(n,n) for n in rc2.index]
        rc2.columns = ['Avg Prob','Avg Dampak (bln)','Thn Terdampak','Risk Index']
        st.dataframe(rc2.round(3), use_container_width=True)

    # Jawaban 3
    with st.expander("✅ Jawaban 3 — Tahapan Kritis (Critical Path)", expanded=False):
        cp3 = sim.critical_path(df).sort_values('probability', ascending=False)
        cp3['Status'] = cp3['probability'].apply(
            lambda p: '🔴 Sangat Kritis' if p>0.55 else ('🟠 Kritis' if p>0.38 else '🟢 Normal'))
        cp3.index = [n.replace('_',' ') for n in cp3.index]
        cp3.columns = ['P. Kritis','Korelasi','Mean (bln)','Kontribusi (%)','Status']
        st.dataframe(cp3.round(3), use_container_width=True)
        top3_names = cp3.index[:3].tolist()
        st.markdown(f"""
        <div class="rec-box">
        <b>🏆 3 Tahapan Paling Kritis:</b><br>
        1. {top3_names[0]} — {cp3.iloc[0]['P. Kritis']:.1%}<br>
        2. {top3_names[1]} — {cp3.iloc[1]['P. Kritis']:.1%}<br>
        3. {top3_names[2]} — {cp3.iloc[2]['P. Kritis']:.1%}
        </div>""", unsafe_allow_html=True)

    # Jawaban 4
    with st.expander("✅ Jawaban 4 — Probabilitas Penyelesaian sesuai Deadline", expanded=False):
        rows4 = []
        for dl in [14,16,18,20,22,24]:
            p_on  = np.mean(total <= dl)
            p_lt  = 1 - p_on
            late  = total[total>dl].mean() - dl if p_lt > 0 else 0
            flag  = '✅' if p_on>=0.80 else ('⚠️' if p_on>=0.40 else '❌')
            rows4.append({'DL (bln)': dl, 'P(Tepat)': f'{p_on:.1%}',
                          'P(Terlambat)': f'{p_lt:.1%}',
                          'Rata-rata Telat': f'{late:.2f} bln', 'Status': flag})
        st.dataframe(pd.DataFrame(rows4), use_container_width=True, hide_index=True)
        st.markdown(f"""
        <div class="info-card">
        <div class="info-card-title">Deadline untuk Tingkat Kepercayaan</div>
        50% confidence → {np.percentile(total,50):.1f} bulan<br>
        80% confidence → {np.percentile(total,80):.1f} bulan<br>
        90% confidence → {np.percentile(total,90):.1f} bulan<br>
        95% confidence → {np.percentile(total,95):.1f} bulan<br><br>
        <b>Deadline 16 bln:</b> {p16:.1f}% → {'⚠️ SANGAT BERISIKO' if p16<30 else '🟠 BERISIKO'}<br>
        <b>Deadline 20 bln:</b> {p20:.1f}% → {'✅ REALISTIS' if p20>=50 else '⚠️ BERISIKO'}<br>
        <b>Deadline 24 bln:</b> {p24:.1f}% → {'✅ SANGAT AMAN' if p24>=80 else '🟠 REALISTIS'}
        </div>""", unsafe_allow_html=True)

    # Jawaban 5
    with st.expander("✅ Jawaban 5 — Pengaruh Penambahan Resource", expanded=False):
        rows5 = []
        for i, sc in enumerate(sc_results):
            rows5.append({'No': f'S{i+1}', 'Tahapan': sc['stage'].replace('_',' '),
                          'Resource': sc['resource'], 'Qty': sc['qty'],
                          'Hemat (bln)': round(sc['saved'],2),
                          'Biaya (Jt Rp)': round(sc['cost']/1e6,1),
                          'ROI (%)': round(sc['roi'],1),
                          'Status': '✅' if sc['roi']>0 else '❌'})
        st.dataframe(pd.DataFrame(rows5), use_container_width=True, hide_index=True)

        best_roi   = max(sc_results, key=lambda x: x['roi'])
        most_saved = max(sc_results, key=lambda x: x['saved'])
        sb = np.percentile(total,80) - mean_t
        cr = np.percentile(total,95) - mean_t

        st.markdown(f"""
        <div class="rec-box">
        <b>🏆 ROI Terbaik:</b> {best_roi['resource']} di {best_roi['stage'].replace('_',' ')} → ROI {best_roi['roi']:.1f}%<br>
        <b>⏱️ Hemat Terbesar:</b> {most_saved['resource']} → {most_saved['saved']:.2f} bulan<br><br>
        <b>📌 Rekomendasi Manajemen Risiko:</b><br>
        • Safety Buffer (80% conf): {sb:.2f} bulan<br>
        • Contingency Reserve (95%): {cr:.2f} bulan<br>
        • Jadwal yang direkomendasikan: {mean_t:.1f} + {sb:.1f} = <b>{mean_t+sb:.1f} bulan</b>
        </div>""", unsafe_allow_html=True)

    # Simulasi custom deadline
    st.markdown('<div class="section-title">🎯 Cek Deadline Target Kamu</div>', unsafe_allow_html=True)
    target = st.slider("Masukkan deadline target (bulan):", 10, 36, 20, 1)
    p_target = np.mean(total <= target)
    days_risk = max(0, np.percentile(total,95) - target)
    cls_t = 'prob-ok' if p_target>=0.7 else ('prob-warning' if p_target>=0.35 else 'prob-danger')
    st.markdown(f"""
    <div class="info-card">
    Deadline <b>{target} bulan</b>:
    <span class="prob-chip {cls_t}">{p_target:.1%} peluang selesai tepat waktu</span>
    <br><small style="color:#64748b">Potensi keterlambatan (P95): {days_risk:.2f} bulan</small>
    </div>""", unsafe_allow_html=True)


# ============================================================================
# 9. FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#475569; font-size:0.8rem; padding:0.5rem 0 1rem;">
  <b style="color:#64748b">Monte Carlo Simulation — Gedung FITE 5 Lantai</b> &nbsp;|&nbsp;
  {n_sim:,} iterasi &nbsp;|&nbsp; Seed: {seed} &nbsp;|&nbsp;
  MODSIM Praktikum 5 · [11S1221]<br>
  <span style="font-size:0.75rem; color:#334155">
  ⚠️ Hasil simulasi bersifat estimasi probabilistik, bukan prediksi deterministik.
  </span>
</div>
""", unsafe_allow_html=True)