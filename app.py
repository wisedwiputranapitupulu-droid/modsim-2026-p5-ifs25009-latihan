import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling (diadaptasi dari template modul bagian 1.3)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stage-card {
        background-color: #F8FAFC;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 3px solid #10B981;
    }
    .risk-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .deadline-safe {
        background-color: #D1FAE5;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 0.3rem 0;
    }
    .deadline-risk {
        background-color: #FEE2E2;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# (Diadaptasi dari template modul - diubah ke konteks konstruksi gedung FITE)
# ============================================================================
class ProjectStage:
    """Kelas untuk memodelkan tahapan pembangunan gedung FITE."""

    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic  = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []

    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        """Sampling durasi dengan distribusi triangular + faktor risiko konstruksi."""
        # Distribusi triangular untuk estimasi tiga titik (dalam bulan)
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )

        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                # Risiko diskrit: cuaca buruk, keterlambatan material, perubahan desain
                probability = risk_params['probability']
                impact      = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration
                )

            elif risk_params['type'] == 'continuous':
                # Risiko kontinu: produktivitas pekerja, kondisi lapangan
                mean = risk_params['mean']
                std  = risk_params['std']
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)

        return base_duration * risk_multiplier


class MonteCarloProjectSimulation:
    """Kelas untuk menjalankan simulasi Monte Carlo pembangunan gedung FITE."""

    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config   = stages_config
        self.num_simulations = num_simulations
        self.stages          = {}
        self.simulation_results = None
        self.initialize_stages()

    def initialize_stages(self):
        """Inisialisasi objek tahapan dari konfigurasi."""
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = ProjectStage(
                name=stage_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )

    def run_simulation(self):
        """Menjalankan simulasi Monte Carlo lengkap dengan dependensi antar tahapan."""
        results = pd.DataFrame(index=range(self.num_simulations))

        # Sampling durasi per tahapan
        for stage_name, stage in self.stages.items():
            results[stage_name] = stage.sample_duration(self.num_simulations)

        # Hitung waktu mulai & selesai berdasarkan dependensi (jalur konstruksi)
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times   = pd.DataFrame(index=range(self.num_simulations))

        for stage_name in self.stages.keys():
            deps = self.stages[stage_name].dependencies
            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)
            end_times[stage_name] = start_times[stage_name] + results[stage_name]

        # Total durasi proyek
        results['Total_Duration'] = end_times.max(axis=1)

        for stage_name in self.stages.keys():
            results[f'{stage_name}_Finish'] = end_times[stage_name]
            results[f'{stage_name}_Start']  = start_times[stage_name]

        self.simulation_results = results
        return results

    def calculate_critical_path_probability(self):
        """Menghitung probabilitas setiap tahapan berada di critical path."""
        if self.simulation_results is None:
            raise ValueError("Run simulation first")

        critical_path_probs = {}
        total_duration = self.simulation_results['Total_Duration']

        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{stage_name}_Finish']
            correlation  = self.simulation_results[stage_name].corr(total_duration)
            is_critical  = (stage_finish + 0.01) >= total_duration
            prob_critical = np.mean(is_critical)

            critical_path_probs[stage_name] = {
                'probability':  prob_critical,
                'correlation':  correlation,
                'avg_duration': self.simulation_results[stage_name].mean()
            }

        return pd.DataFrame(critical_path_probs).T

    def analyze_risk_contribution(self):
        """Analisis kontribusi risiko tiap tahapan terhadap variabilitas total."""
        if self.simulation_results is None:
            raise ValueError("Run simulation first")

        total_var     = self.simulation_results['Total_Duration'].var()
        contributions = {}

        for stage_name in self.stages.keys():
            stage_var   = self.simulation_results[stage_name].var()
            stage_covar = self.simulation_results[stage_name].cov(
                self.simulation_results['Total_Duration']
            )
            contribution = (stage_covar / total_var) * 100

            contributions[stage_name] = {
                'variance':             stage_var,
                'contribution_percent': contribution,
                'std_dev':              np.sqrt(stage_var)
            }

        return pd.DataFrame(contributions).T


# ============================================================================
# 3. KONFIGURASI DEFAULT TAHAPAN PEMBANGUNAN GEDUNG FITE
# (Menggantikan tahapan proyek software di template modul)
# Durasi dalam satuan BULAN
# ============================================================================
DEFAULT_CONFIG = {
    "Perencanaan_Desain": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "perubahan_desain_lab": {
                "type": "discrete",
                "probability": 0.35,
                "impact": 0.30   # +30% jika desain lab VR/AR, game berubah
            },
            "kejelasan_spesifikasi": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.15
            }
        }
    },
    "Persiapan_Lahan": {
        "base_params": {"optimistic": 1, "most_likely": 1.5, "pessimistic": 2.5},
        "risk_factors": {
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.20
            },
            "kondisi_tanah_bermasalah": {
                "type": "discrete",
                "probability": 0.15,
                "impact": 0.40
            }
        },
        "dependencies": ["Perencanaan_Desain"]
    },
    "Struktur_Pondasi": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.30,
                "impact": 0.25
            },
            "keterlambatan_material_besi_beton": {
                "type": "discrete",
                "probability": 0.20,
                "impact": 0.35
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.20
            }
        },
        "dependencies": ["Persiapan_Lahan"]
    },
    "Struktur_Beton_5_Lantai": {
        "base_params": {"optimistic": 3, "most_likely": 5, "pessimistic": 8},
        "risk_factors": {
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.35,
                "impact": 0.20
            },
            "keterlambatan_material_teknis": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.30   # Formwork, scaffolding, readymix
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.25
            }
        },
        "dependencies": ["Struktur_Pondasi"]
    },
    "Pemasangan_Dinding_Fasad": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.30,
                "impact": 0.20
            },
            "keterlambatan_material_fasad": {
                "type": "discrete",
                "probability": 0.20,
                "impact": 0.25
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.20
            }
        },
        "dependencies": ["Struktur_Beton_5_Lantai"]
    },
    "Instalasi_MEP": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "keterlambatan_material_teknis_khusus": {
                "type": "discrete",
                "probability": 0.30,
                "impact": 0.35   # Listrik lab, plumbing, HVAC
            },
            "perubahan_desain_lab": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.20
            },
            "produktivitas_teknisi": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.20
            }
        },
        "dependencies": ["Pemasangan_Dinding_Fasad"]
    },
    "Finishing_Interior": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "perubahan_desain_interior_lab": {
                "type": "discrete",
                "probability": 0.30,
                "impact": 0.25
            },
            "keterlambatan_material_interior": {
                "type": "discrete",
                "probability": 0.20,
                "impact": 0.20
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.20
            }
        },
        "dependencies": ["Instalasi_MEP"]
    },
    "Instalasi_Peralatan_Lab": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "keterlambatan_peralatan_impor": {
                "type": "discrete",
                "probability": 0.40,
                "impact": 0.50   # Peralatan VR/AR, lab game sering indent/impor
            },
            "kalibrasi_uji_coba_lab": {
                "type": "discrete",
                "probability": 0.30,
                "impact": 0.25
            }
        },
        "dependencies": ["Finishing_Interior"]
    },
    "Uji_Kelayakan_Serah_Terima": {
        "base_params": {"optimistic": 0.5, "most_likely": 1, "pessimistic": 2},
        "risk_factors": {
            "temuan_inspeksi_bangunan": {
                "type": "discrete",
                "probability": 0.20,
                "impact": 0.60
            },
            "proses_administrasi_izin": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.30
            }
        },
        "dependencies": ["Instalasi_Peralatan_Lab"]
    }
}


# ============================================================================
# 4. FUNGSI VISUALISASI PLOTLY
# (Diadaptasi dari template modul - satuan diubah ke bulan, deadline disesuaikan)
# ============================================================================
def create_distribution_plot(results):
    """Membuat plot distribusi durasi total proyek."""
    total_duration  = results['Total_Duration']
    mean_duration   = total_duration.mean()
    median_duration = np.median(total_duration)

    fig = go.Figure()

    # Histogram distribusi
    fig.add_trace(go.Histogram(
        x=total_duration,
        nbinsx=60,
        name='Distribusi Durasi',
        marker_color='steelblue',
        opacity=0.75,
        histnorm='probability density'
    ))

    # Garis mean dan median
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_duration:.1f} bulan",
                  annotation_position="top right")
    fig.add_vline(x=median_duration, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_duration:.1f} bulan",
                  annotation_position="top left")

    # Confidence intervals
    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])

    fig.add_vrect(x0=ci_80[0], x1=ci_80[1], fillcolor="yellow", opacity=0.2,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci_95[0], x1=ci_95[1], fillcolor="orange", opacity=0.1,
                  annotation_text="95% CI", line_width=0)

    # Garis deadline studi kasus (16, 20, 24 bulan)
    for dl, col, pos in zip(
        [16, 20, 24],
        ['purple', 'darkorange', 'darkgreen'],
        ['top left', 'top right', 'top right']
    ):
        fig.add_vline(x=dl, line_dash="dot", line_color=col, line_width=2,
                      annotation_text=f"Deadline {dl} bln",
                      annotation_position=pos)

    fig.update_layout(
        title='Distribusi Durasi Total Pembangunan Gedung FITE 5 Lantai',
        xaxis_title='Durasi Total Proyek (Bulan)',
        yaxis_title='Densitas Probabilitas',
        showlegend=True,
        height=500
    )

    return fig, {
        'mean':   mean_duration,
        'median': median_duration,
        'std':    total_duration.std(),
        'min':    total_duration.min(),
        'max':    total_duration.max(),
        'ci_80':  ci_80,
        'ci_95':  ci_95
    }


def create_completion_probability_plot(results):
    """Membuat kurva probabilitas penyelesaian proyek."""
    deadlines        = np.arange(10, 55, 0.5)
    completion_probs = [np.mean(results['Total_Duration'] <= dl) for dl in deadlines]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=deadlines,
        y=completion_probs,
        mode='lines',
        name='Probabilitas Selesai',
        line=dict(color='darkblue', width=3),
        fill='tozeroy',
        fillcolor='rgba(173, 216, 230, 0.3)'
    ))

    # Garis referensi probabilitas
    for level, col, label in zip(
        [0.5, 0.8, 0.95],
        ['red', 'green', 'blue'],
        ['50%', '80%', '95%']
    ):
        fig.add_hline(y=level, line_dash="dash", line_color=col,
                      annotation_text=label, annotation_position="right")

    # Tandai 3 deadline studi kasus
    for dl, col in zip([16, 20, 24], ['purple', 'darkorange', 'darkgreen']):
        prob = np.mean(results['Total_Duration'] <= dl)
        fig.add_trace(go.Scatter(
            x=[dl], y=[prob],
            mode='markers+text',
            marker=dict(size=14, color=col, symbol='diamond'),
            text=[f'{dl} bln<br>{prob:.1%}'],
            textposition="top center",
            name=f'Deadline {dl} bln'
        ))

    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek Pembangunan Gedung FITE',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[10, 55],
        height=500
    )

    return fig


def create_critical_path_plot(critical_analysis):
    """Membuat plot analisis critical path."""
    critical_analysis = critical_analysis.sort_values('probability', ascending=True)

    fig = go.Figure()

    colors = ['#DC2626' if prob > 0.7 else ('#F59E0B' if prob > 0.3 else '#86EFAC')
              for prob in critical_analysis['probability']]

    fig.add_trace(go.Bar(
        y=[stage.replace('_', ' ') for stage in critical_analysis.index],
        x=critical_analysis['probability'],
        orientation='h',
        marker_color=colors,
        text=[f'{prob:.1%}' for prob in critical_analysis['probability']],
        textposition='auto'
    ))

    fig.add_vline(x=0.5, line_dash="dot", line_color="gray",
                  annotation_text="50%")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange",
                  annotation_text="Batas Kritis 70%")

    fig.update_layout(
        title='Analisis Critical Path per Tahapan Konstruksi',
        xaxis_title='Probabilitas Menjadi Critical Path',
        xaxis_range=[0, 1.1],
        height=500
    )

    return fig


def create_stage_boxplot(results, stages):
    """Membuat boxplot distribusi durasi per tahapan."""
    stage_names = list(stages.keys())

    fig = go.Figure()

    for i, stage in enumerate(stage_names):
        fig.add_trace(go.Box(
            y=results[stage],
            name=stage.replace('_', '<br>'),
            boxmean='sd',
            marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title='Distribusi Durasi per Tahapan (Bulan)',
        yaxis_title='Durasi (Bulan)',
        height=500,
        showlegend=False
    )

    return fig


def create_risk_contribution_plot(risk_contrib):
    """Membuat plot kontribusi risiko per tahapan."""
    risk_contrib = risk_contrib.sort_values('contribution_percent', ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[name.replace('_', '<br>') for name in risk_contrib.index],
        y=risk_contrib['contribution_percent'],
        marker_color=px.colors.qualitative.Set3,
        text=[f'{contrib:.1f}%' for contrib in risk_contrib['contribution_percent']],
        textposition='auto'
    ))

    fig.update_layout(
        title='Kontribusi Risiko per Tahapan terhadap Variabilitas Total',
        yaxis_title='Kontribusi terhadap Variabilitas (%)',
        height=420
    )

    return fig


def create_correlation_heatmap(results, stages):
    """Membuat heatmap korelasi antar tahapan."""
    correlation_matrix = results[list(stages.keys())].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=[name.replace('_', '<br>') for name in correlation_matrix.columns],
        y=[name.replace('_', '<br>') for name in correlation_matrix.index],
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 9},
        hoverongaps=False
    ))

    fig.update_layout(
        title='Matriks Korelasi Antar Tahapan Konstruksi',
        height=500
    )

    return fig


# ============================================================================
# 5. FUNGSI UTAMA STREAMLIT
# (Struktur tetap mengikuti template modul bagian 1.3, konten disesuaikan studi kasus)
# ============================================================================
def main():
    # Header aplikasi
    st.markdown(
        '<h1 class="main-header">🏗️ Simulasi Monte Carlo<br>Estimasi Waktu Pembangunan Gedung FITE 5 Lantai</h1>',
        unsafe_allow_html=True
    )

    # Deskripsi studi kasus
    st.markdown("""
    <div class="info-box">
    <b>📌 Studi Kasus:</b> Proyek pembangunan gedung <b>Fakultas Informatika & Teknik Elektro (FITE) 5 lantai</b>
    dengan fasilitas: ruang kelas, laboratorium komputer, lab elektro, lab mobile, lab VR/AR, lab game,
    ruang dosen, toilet, dan ruang serbaguna.<br><br>
    Simulasi Monte Carlo digunakan untuk memodelkan <b>ketidakpastian durasi konstruksi</b> akibat faktor:
    cuaca buruk, keterlambatan material teknis khusus, perubahan desain laboratorium, dan produktivitas pekerja.
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SIDEBAR: Konfigurasi Simulasi
    # -----------------------------------------------------------------------
    st.sidebar.markdown('<h2>⚙️ Konfigurasi Simulasi</h2>', unsafe_allow_html=True)

    num_simulations = st.sidebar.slider(
        'Jumlah Iterasi Simulasi:',
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
        help='Semakin banyak iterasi, semakin akurat hasilnya tetapi lebih lama waktu prosesnya'
    )

    st.sidebar.markdown('<h3>📋 Konfigurasi Tahapan Konstruksi</h3>', unsafe_allow_html=True)

    # Buat salinan config yang bisa diubah user
    user_config = {}
    for stage_name, config in DEFAULT_CONFIG.items():
        bp = config['base_params']
        with st.sidebar.expander(f"⚙️ {stage_name.replace('_', ' ')}", expanded=False):
            optimistic = st.number_input(
                "Optimistic (bulan)",
                min_value=0.5, max_value=20.0,
                value=float(bp['optimistic']), step=0.5,
                key=f"opt_{stage_name}"
            )
            most_likely = st.number_input(
                "Most Likely (bulan)",
                min_value=0.5, max_value=30.0,
                value=float(bp['most_likely']), step=0.5,
                key=f"ml_{stage_name}"
            )
            pessimistic = st.number_input(
                "Pessimistic (bulan)",
                min_value=0.5, max_value=40.0,
                value=float(bp['pessimistic']), step=0.5,
                key=f"pes_{stage_name}"
            )

        stage_cfg = dict(config)
        stage_cfg['base_params'] = {
            'optimistic':  optimistic,
            'most_likely': most_likely,
            'pessimistic': pessimistic
        }
        user_config[stage_name] = stage_cfg

    # Tombol jalankan simulasi
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #666;">
    <b>Keterangan:</b><br>
    • Optimistic  : Estimasi terbaik<br>
    • Most Likely : Estimasi realistis<br>
    • Pessimistic : Estimasi terburuk<br>
    • CI          : Confidence Interval<br><br>
    <b>Faktor Risiko Konstruksi:</b><br>
    • Cuaca buruk → +20–25% durasi<br>
    • Keterlambatan material → +30–35%<br>
    • Perubahan desain lab → +25–30%<br>
    • Produktivitas pekerja → variasi kontinu<br>
    • Peralatan impor (VR/AR) → +50%
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Session state
    # -----------------------------------------------------------------------
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None

    # Jalankan simulasi
    if run_simulation:
        with st.spinner('⏳ Menjalankan simulasi Monte Carlo... Harap tunggu...'):
            simulator = MonteCarloProjectSimulation(
                stages_config=user_config,
                num_simulations=num_simulations
            )
            results = simulator.run_simulation()
            st.session_state.simulation_results = results
            st.session_state.simulator = simulator
        st.success(f'✅ Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.')

    # -----------------------------------------------------------------------
    # TAMPILKAN HASIL
    # -----------------------------------------------------------------------
    if st.session_state.simulation_results is not None:
        results   = st.session_state.simulation_results
        simulator = st.session_state.simulator

        total_duration  = results['Total_Duration']
        mean_duration   = total_duration.mean()
        median_duration = np.median(total_duration)
        ci_80 = np.percentile(total_duration, [10, 90])
        ci_95 = np.percentile(total_duration, [2.5, 97.5])

        # ==================================================================
        # BAGIAN 1: STATISTIK UTAMA
        # ==================================================================
        st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h3>{mean_duration:.1f} bln</h3><p>Rata-rata Durasi</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <h3>{median_duration:.1f} bln</h3><p>Median Durasi</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <h3>{ci_80[0]:.1f}–{ci_80[1]:.1f}</h3><p>80% Confidence Interval (bln)</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <h3>{ci_95[0]:.1f}–{ci_95[1]:.1f}</h3><p>95% Confidence Interval (bln)</p>
            </div>""", unsafe_allow_html=True)

        # Jawaban langsung 5 pertanyaan studi kasus
        st.markdown("---")
        st.markdown("### 🎯 Jawaban Permasalahan Studi Kasus")

        p16 = np.mean(total_duration <= 16)
        p20 = np.mean(total_duration <= 20)
        p24 = np.mean(total_duration <= 24)

        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            warna = "deadline-safe" if p16 >= 0.5 else "deadline-risk"
            st.markdown(f"""<div class="{warna}">
                <b>Deadline 16 Bulan</b><br>
                Probabilitas selesai: <b>{p16:.1%}</b><br>
                {'✅ Aman' if p16 >= 0.5 else '🔴 Sangat Berisiko'}
            </div>""", unsafe_allow_html=True)
        with qa2:
            warna = "deadline-safe" if p20 >= 0.5 else "deadline-risk"
            st.markdown(f"""<div class="{warna}">
                <b>Deadline 20 Bulan</b><br>
                Probabilitas selesai: <b>{p20:.1%}</b><br>
                {'✅ Aman' if p20 >= 0.5 else '🔴 Sangat Berisiko'}
            </div>""", unsafe_allow_html=True)
        with qa3:
            warna = "deadline-safe" if p24 >= 0.5 else "deadline-risk"
            st.markdown(f"""<div class="{warna}">
                <b>Deadline 24 Bulan</b><br>
                Probabilitas selesai: <b>{p24:.1%}</b><br>
                {'✅ Aman' if p24 >= 0.5 else '🔴 Sangat Berisiko'}
            </div>""", unsafe_allow_html=True)

        # ==================================================================
        # BAGIAN 2: VISUALISASI UTAMA
        # ==================================================================
        st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Distribusi Durasi",
            "🎯 Probabilitas Penyelesaian",
            "🔍 Analisis Tahapan",
            "📊 Analisis Risiko"
        ])

        with tab1:
            fig_dist, stats = create_distribution_plot(results)
            st.plotly_chart(fig_dist, use_container_width=True)

            with st.expander("📋 Detail Statistik Distribusi"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Statistik Deskriptif:**")
                    st.write(f"- Rata-rata  : {stats['mean']:.2f} bulan")
                    st.write(f"- Median     : {stats['median']:.2f} bulan")
                    st.write(f"- Standar Deviasi: {stats['std']:.2f} bulan")
                    st.write(f"- Minimum    : {stats['min']:.2f} bulan")
                    st.write(f"- Maksimum   : {stats['max']:.2f} bulan")
                with col2:
                    st.write("**Confidence Intervals:**")
                    st.write(f"- 80% CI : [{stats['ci_80'][0]:.2f}, {stats['ci_80'][1]:.2f}] bulan")
                    st.write(f"- 95% CI : [{stats['ci_95'][0]:.2f}, {stats['ci_95'][1]:.2f}] bulan")

        with tab2:
            fig_prob = create_completion_probability_plot(results)
            st.plotly_chart(fig_prob, use_container_width=True)

            with st.expander("📅 Analisis Probabilitas Deadline"):
                deadlines_list = list(range(14, 46, 2))
                cols = st.columns(len(deadlines_list))
                for i, deadline in enumerate(deadlines_list):
                    prob_on_time = np.mean(total_duration <= deadline)
                    prob_late    = 1 - prob_on_time
                    with cols[i]:
                        st.metric(
                            label=f"{deadline} bln",
                            value=f"{prob_on_time:.0%}",
                            delta=f"{prob_late:.0%} terlambat" if prob_late > 0 else "Tepat waktu",
                            delta_color="inverse"
                        )

            # Custom deadline checker
            st.markdown("**Cek Deadline Custom:**")
            target_deadline = st.number_input(
                "Masukkan deadline target (bulan):",
                min_value=10, max_value=60, value=30, step=1
            )
            prob_target   = np.mean(total_duration <= target_deadline)
            days_at_risk  = max(0, np.percentile(total_duration, 95) - target_deadline)
            st.metric(
                label=f"Probabilitas selesai dalam {target_deadline} bulan",
                value=f"{prob_target:.1%}",
                delta=f"Potensi keterlambatan: {days_at_risk:.1f} bulan" if days_at_risk > 0 else "Tepat waktu",
                delta_color="inverse"
            )

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                critical_analysis = simulator.calculate_critical_path_probability()
                fig_critical = create_critical_path_plot(critical_analysis)
                st.plotly_chart(fig_critical, use_container_width=True)
            with col2:
                fig_boxplot = create_stage_boxplot(results, simulator.stages)
                st.plotly_chart(fig_boxplot, use_container_width=True)

            with st.expander("🔍 Detail Analisis Critical Path"):
                critical_df = critical_analysis.sort_values('probability', ascending=False).copy()
                critical_df['status'] = critical_df['probability'].apply(
                    lambda p: '🔴 Kritis' if p > 0.7 else ('🟡 Perlu Perhatian' if p > 0.3 else '🟢 Aman')
                )
                st.dataframe(critical_df, use_container_width=True)

        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                risk_contrib = simulator.analyze_risk_contribution()
                fig_risk = create_risk_contribution_plot(risk_contrib)
                st.plotly_chart(fig_risk, use_container_width=True)
            with col2:
                fig_corr = create_correlation_heatmap(results, simulator.stages)
                st.plotly_chart(fig_corr, use_container_width=True)

            with st.expander("📋 Detail Analisis Kontribusi Risiko"):
                st.dataframe(risk_contrib, use_container_width=True)

        # ==================================================================
        # BAGIAN 3: ANALISIS STATISTIK LENGKAP
        # ==================================================================
        st.markdown('<h2 class="sub-header">📋 Analisis Statistik Lengkap</h2>', unsafe_allow_html=True)

        with st.expander("📊 Tabel Data Simulasi (100 baris pertama)", expanded=False):
            st.dataframe(results.head(100), use_container_width=True)

        # Statistik per tahapan
        st.markdown("**Statistik Durasi per Tahapan Konstruksi (Bulan):**")
        stage_stats = pd.DataFrame()
        for stage_name in simulator.stages.keys():
            stage_data = results[stage_name]
            stage_stats[stage_name] = [
                stage_data.mean(),
                stage_data.std(),
                np.percentile(stage_data, 25),
                np.percentile(stage_data, 50),
                np.percentile(stage_data, 75)
            ]
        stage_stats.index = ['Mean', 'Std Dev', 'Q1', 'Median', 'Q3']
        st.dataframe(stage_stats.T.round(2), use_container_width=True)

        # ==================================================================
        # BAGIAN 4: ANALISIS DEADLINE & REKOMENDASI
        # ==================================================================
        st.markdown('<h2 class="sub-header">🎯 Analisis Deadline & Rekomendasi</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            target = st.number_input(
                "Masukkan deadline target Anda (bulan):",
                min_value=10, max_value=60, value=24, step=1,
                key="deadline_rekomendasi"
            )
            prob_t  = np.mean(total_duration <= target)
            at_risk = max(0, np.percentile(total_duration, 95) - target)
            st.metric(
                label=f"Probabilitas selesai dalam {target} bulan",
                value=f"{prob_t:.1%}",
                delta=f"Potensi keterlambatan: {at_risk:.1f} bulan" if at_risk > 0 else "Tepat waktu",
                delta_color="inverse"
            )

        with col2:
            safety_buffer       = np.percentile(total_duration, 80) - mean_duration
            contingency_reserve = np.percentile(total_duration, 95) - mean_duration

            st.markdown(f"""
            <div class="info-box">
                <h4>🏗️ Rekomendasi Manajemen Risiko:</h4>
                • <b>Safety Buffer</b> (80% confidence): <b>+{safety_buffer:.1f} bulan</b><br>
                • <b>Contingency Reserve</b> (95%): <b>+{contingency_reserve:.1f} bulan</b><br><br>
                • <b>Estimasi realistis:</b> {mean_duration:.1f} + {safety_buffer:.1f}
                  = <b>{mean_duration + safety_buffer:.1f} bulan</b><br>
                • <b>Estimasi konservatif:</b> {mean_duration:.1f} + {contingency_reserve:.1f}
                  = <b>{mean_duration + contingency_reserve:.1f} bulan</b>
            </div>
            """, unsafe_allow_html=True)

        # ==================================================================
        # BAGIAN 5: ANALISIS PENAMBAHAN RESOURCE
        # (Menjawab pertanyaan ke-5 studi kasus)
        # ==================================================================
        st.markdown('<h2 class="sub-header">⚡ Analisis Penambahan Resource</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Simulasi dampak penambahan resource (pekerja khusus, alat berat, insinyur) terhadap
        percepatan penyelesaian proyek pembangunan gedung FITE.
        </div>
        """, unsafe_allow_html=True)

        RESOURCE_TYPES = {
            'pekerja_khusus':   {'cost_per_month': 8_000_000,  'productivity_gain': 0.25},
            'alat_berat':       {'cost_per_month': 25_000_000, 'productivity_gain': 0.35},
            'insinyur':         {'cost_per_month': 15_000_000, 'productivity_gain': 0.20},
            'mandor_senior':    {'cost_per_month': 12_000_000, 'productivity_gain': 0.22},
            'konsultan_teknis': {'cost_per_month': 20_000_000, 'productivity_gain': 0.15},
        }

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sel_stage = st.selectbox(
                "Tahapan yang dioptimasi:",
                list(simulator.stages.keys()),
                format_func=lambda x: x.replace('_', ' ')
            )
        with rc2:
            sel_resource = st.selectbox(
                "Jenis Resource:",
                list(RESOURCE_TYPES.keys()),
                format_func=lambda x: x.replace('_', ' ')
            )
        with rc3:
            sel_qty = st.slider("Jumlah Resource:", 1, 5, 2)
        sel_dur = st.slider("Durasi penambahan (bulan):", 1, 12, 3)

        if st.button("🔍 Hitung Dampak Penambahan Resource"):
            rp = RESOURCE_TYPES[sel_resource]
            improvement_factor = 1 - (rp['productivity_gain'] * min(sel_qty / 3, 1))

            sc_results = results.copy()
            sc_results[sel_stage] = sc_results[sel_stage] * improvement_factor

            sc_totals = []
            for idx in range(len(results)):
                stage_times = {}
                for curr_stage in simulator.stages.keys():
                    deps  = simulator.stages[curr_stage].dependencies
                    start = 0 if not deps else max(stage_times.get(d, 0) for d in deps)
                    dur   = (sc_results.loc[idx, curr_stage]
                             if curr_stage == sel_stage
                             else results.loc[idx, curr_stage])
                    stage_times[curr_stage] = start + dur
                sc_totals.append(max(stage_times.values()))

            opt_mean  = np.mean(sc_totals)
            reduction = mean_duration - opt_mean
            pct_imp   = (reduction / mean_duration) * 100
            total_cost = rp['cost_per_month'] * sel_qty * sel_dur
            saving     = reduction * 150_000_000   # Rp 150 juta/bulan biaya proyek
            roi        = ((saving - total_cost) / total_cost) * 100

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Durasi Baseline",   f"{mean_duration:.1f} bln")
            r2.metric("Durasi Setelah",    f"{opt_mean:.1f} bln",
                      delta=f"-{reduction:.1f} bln ({pct_imp:.1f}%)")
            r3.metric("Biaya Tambahan",    f"Rp {total_cost/1e6:.0f} jt")
            r4.metric("ROI",               f"{roi:.0f}%",
                      delta="Menguntungkan" if roi > 0 else "Merugi",
                      delta_color="normal" if roi > 0 else "inverse")

            # Perbandingan probabilitas deadline
            prob_data = []
            for dl in [16, 20, 24, 28, 32]:
                pb = np.mean(total_duration <= dl)
                po = np.mean(np.array(sc_totals) <= dl)
                prob_data.append({
                    'Deadline (bln)': dl,
                    'Baseline': f'{pb:.1%}',
                    'Setelah Optimasi': f'{po:.1%}',
                    'Peningkatan': f'+{(po - pb) * 100:.1f}%'
                })
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

        # ==================================================================
        # BAGIAN 6: INFORMASI TEKNIS
        # ==================================================================
        with st.expander("ℹ️ Informasi Teknis Simulasi", expanded=False):
            st.write("**Parameter Simulasi:**")
            st.write(f"- Jumlah iterasi : {num_simulations:,}")
            st.write(f"- Jumlah tahapan : {len(simulator.stages)}")
            st.write(f"- Metode         : Distribusi Triangular + Faktor Risiko Monte Carlo")

            st.write("\n**Konfigurasi Tahapan Konstruksi:**")
            for stage_name, config in user_config.items():
                base = config['base_params']
                st.markdown(f"""
                <div class="stage-card">
                <b>{stage_name.replace('_', ' ')}</b><br>
                • Optimistic : {base['optimistic']} bulan<br>
                • Most Likely: {base['most_likely']} bulan<br>
                • Pessimistic: {base['pessimistic']} bulan
                </div>
                """, unsafe_allow_html=True)

    else:
        # Tampilkan instruksi jika simulasi belum dijalankan
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background-color: #f8f9fa; border-radius: 10px;">
            <h3>🚀 Siap untuk memulai simulasi?</h3>
            <p>Atur parameter di sidebar kiri, lalu klik tombol <b>"Run Simulation"</b> untuk memulai analisis.</p>
            <p>📊 Hasil simulasi akan ditampilkan di sini setelah proses selesai.</p>
        </div>
        """, unsafe_allow_html=True)

        # Preview konfigurasi tahapan
        st.markdown('<h2 class="sub-header">📋 Preview Tahapan Konstruksi Gedung FITE</h2>',
                    unsafe_allow_html=True)
        for stage_name, config in DEFAULT_CONFIG.items():
            base = config['base_params']
            st.markdown(f"""
            <div class="stage-card">
            <b>{stage_name.replace('_', ' ')}</b> |
            Optimistic: {base['optimistic']} bln |
            Most Likely: {base['most_likely']} bln |
            Pessimistic: {base['pessimistic']} bln
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><b>[11S1221] Pemodelan dan Simulasi | Praktikum 5: Monte Carlo Simulation</b></p>
    <p>Studi Kasus: Estimasi Waktu Pembangunan Gedung FITE 5 Lantai</p>
    <p>⚠️ Hasil simulasi ini merupakan estimasi probabilistik dan bukan prediksi pasti.</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 6. JALANKAN APLIKASI
# ============================================================================
if __name__ == "__main__":
    main()