# ============================================================================
# SIMULASI MONTE CARLO - ESTIMASI WAKTU PEMBANGUNAN GEDUNG FITE 5 LANTAI
# Modul Praktikum 5: Monte Carlo Simulation
# ============================================================================

# ============================================================================
# 1. KONFIGURASI
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Konfigurasi
NUM_SIMULATIONS = 20000  # Iterasi simulasi Monte Carlo
np.random.seed(42)

print("=" * 70)
print("SIMULASI MONTE CARLO - PEMBANGUNAN GEDUNG FITE 5 LANTAI")
print("=" * 70)
print(f"Jumlah iterasi simulasi : {NUM_SIMULATIONS:,}")
print(f"Seed acak               : 42")

# ============================================================================
# 2. PEMODELAN SISTEM
# ============================================================================
class TahapanProyek:
    """
    Kelas untuk memodelkan tahapan pembangunan gedung FITE.
    Setiap tahapan memiliki durasi acak mengikuti distribusi triangular
    ditambah faktor risiko konstruksi.
    """
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']    # Bulan
        self.most_likely = base_params['most_likely']  # Bulan
        self.pessimistic = base_params['pessimistic']  # Bulan
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []

    def sample_duration(self, n_simulations):
        """
        Sampling durasi dengan distribusi triangular + faktor risiko konstruksi.
        """
        # Distribusi triangular (optimistic, most_likely, pessimistic)
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )

        # Aplikasi faktor risiko
        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                # Risiko diskrit: cuaca buruk, keterlambatan material, perubahan desain
                probability = risk_params['probability']
                impact = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration
                )
            elif risk_params['type'] == 'continuous':
                # Risiko kontinu: produktivitas pekerja
                mean = risk_params['mean']
                std = risk_params['std']
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)

        return base_duration


# ============================================================================
# 3. DEFINISI TAHAPAN PROYEK PEMBANGUNAN GEDUNG FITE
# ============================================================================
# Durasi dalam satuan BULAN
# Faktor risiko disesuaikan konteks konstruksi gedung

project_stages_config = {
    "Perencanaan_Desain": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "perubahan_desain_lab": {
                "type": "discrete",
                "probability": 0.35,
                "impact": 0.30   # +30% jika ada perubahan desain lab (VR/AR, game, dll)
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
                "impact": 0.20   # +20% akibat cuaca buruk
            },
            "kondisi_tanah": {
                "type": "discrete",
                "probability": 0.15,
                "impact": 0.40   # +40% jika kondisi tanah bermasalah
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
            "keterlambatan_material_besi": {
                "type": "discrete",
                "probability": 0.20,
                "impact": 0.35   # +35% jika material besi/beton terlambat
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.20
            }
        },
        "dependencies": ["Persiapan_Lahan"]
    },
    "Struktur_Beton_Lantai": {
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
                "impact": 0.30   # Material khusus: formwork, scaffolding, dll
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
        "dependencies": ["Struktur_Beton_Lantai"]
    },
    "Instalasi_MEP": {
        # Mechanical, Electrical, Plumbing
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "keterlambatan_material_teknis_khusus": {
                "type": "discrete",
                "probability": 0.30,
                "impact": 0.35   # Peralatan lab VR/AR, listrik khusus
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
        # Ruang kelas, laboratorium, ruang dosen, toilet, serbaguna
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "perubahan_desain_interior": {
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
        # Lab komputer, elektro, mobile, VR/AR, game
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "keterlambatan_peralatan_khusus": {
                "type": "discrete",
                "probability": 0.40,
                "impact": 0.50   # Peralatan lab VR/AR & game sering indent/impor
            },
            "kalibrasi_dan_uji_coba": {
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
            "temuan_inspeksi": {
                "type": "discrete",
                "probability": 0.20,
                "impact": 0.60   # Jika ada temuan mayor, perbaikan bisa lama
            },
            "proses_administrasi": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.30
            }
        },
        "dependencies": ["Instalasi_Peralatan_Lab"]
    }
}

print(f"\nJumlah tahapan proyek    : {len(project_stages_config)}")
print("Tahapan:")
for i, stage in enumerate(project_stages_config.keys(), 1):
    print(f"  {i}. {stage.replace('_', ' ')}")


# ============================================================================
# 4. SIMULATOR MONTE CARLO
# ============================================================================
class SimulasiMonteCarloGedung:
    """Kelas untuk menjalankan simulasi Monte Carlo pembangunan gedung FITE."""

    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self._inisialisasi_tahapan()

    def _inisialisasi_tahapan(self):
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = TahapanProyek(
                name=stage_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )

    def run_simulation(self):
        """Menjalankan simulasi Monte Carlo lengkap."""
        results = pd.DataFrame(index=range(self.num_simulations))

        # Sampling durasi per tahapan
        for stage_name, stage in self.stages.items():
            results[stage_name] = stage.sample_duration(self.num_simulations)

        # Hitung waktu mulai dan selesai berdasarkan dependensi
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))

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
            results[f'{stage_name}_Start'] = start_times[stage_name]

        self.simulation_results = results
        return results

    def hitung_critical_path(self):
        """Menghitung probabilitas setiap tahapan menjadi critical path."""
        if self.simulation_results is None:
            raise ValueError("Jalankan simulasi terlebih dahulu.")

        critical_path_probs = {}
        total_duration = self.simulation_results['Total_Duration']

        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{stage_name}_Finish']
            correlation = self.simulation_results[stage_name].corr(total_duration)
            is_critical = (stage_finish + 0.01) >= total_duration
            prob_critical = np.mean(is_critical)

            critical_path_probs[stage_name] = {
                'probability': prob_critical,
                'correlation': correlation,
                'avg_duration': self.simulation_results[stage_name].mean()
            }

        return pd.DataFrame(critical_path_probs).T

    def analisis_kontribusi_risiko(self):
        """Analisis kontribusi setiap tahapan terhadap variabilitas total."""
        if self.simulation_results is None:
            raise ValueError("Jalankan simulasi terlebih dahulu.")

        total_var = self.simulation_results['Total_Duration'].var()
        contributions = {}

        for stage_name in self.stages.keys():
            stage_var = self.simulation_results[stage_name].var()
            stage_covar = self.simulation_results[stage_name].cov(
                self.simulation_results['Total_Duration']
            )
            contribution = (stage_covar / total_var) * 100
            contributions[stage_name] = {
                'variance': stage_var,
                'contribution_percent': contribution,
                'std_dev': np.sqrt(stage_var)
            }

        return pd.DataFrame(contributions).T


# ============================================================================
# 5. JALANKAN SIMULASI
# ============================================================================
print("\n" + "=" * 70)
print("MENJALANKAN SIMULASI...")
print("=" * 70)

simulator = SimulasiMonteCarloGedung(
    stages_config=project_stages_config,
    num_simulations=NUM_SIMULATIONS
)

results = simulator.run_simulation()
print(f"Simulasi selesai! {NUM_SIMULATIONS:,} iterasi berhasil dijalankan.\n")
print(results[['Total_Duration'] + list(simulator.stages.keys())].describe().round(2))


# ============================================================================
# 6. ANALISIS STATISTIK
# ============================================================================
print("\n" + "=" * 70)
print("LAPORAN STATISTIK PROYEK PEMBANGUNAN GEDUNG FITE")
print("=" * 70)

total_duration = results['Total_Duration']

print(f"\nSTATISTIK DURASI TOTAL PROYEK (dalam bulan):")
print(f"• Rata-rata       : {total_duration.mean():.1f} bulan")
print(f"• Median          : {np.median(total_duration):.1f} bulan")
print(f"• Standar Deviasi : {total_duration.std():.1f} bulan")
print(f"• Minimum         : {total_duration.min():.1f} bulan")
print(f"• Maksimum        : {total_duration.max():.1f} bulan")

print(f"\nCONFIDENCE INTERVALS:")
print(f"• 80% CI : [{np.percentile(total_duration, 10):.1f}, {np.percentile(total_duration, 90):.1f}] bulan")
print(f"• 90% CI : [{np.percentile(total_duration, 5):.1f}, {np.percentile(total_duration, 95):.1f}] bulan")
print(f"• 95% CI : [{np.percentile(total_duration, 2.5):.1f}, {np.percentile(total_duration, 97.5):.1f}] bulan")

# Pertanyaan utama studi kasus:
# 4. Probabilitas penyelesaian sesuai deadline: 16, 20, 24 bulan
print(f"\nPROBABILITAS PENYELESAIAN SESUAI DEADLINE:")
deadline_scenarios = [16, 20, 24]
for dl in deadline_scenarios:
    prob_on_time = np.mean(total_duration <= dl)
    prob_late = 1 - prob_on_time
    days_at_risk = max(0, np.percentile(total_duration, 95) - dl)
    print(f"\n  Deadline {dl} bulan:")
    print(f"  • Probabilitas selesai tepat waktu : {prob_on_time:.1%}")
    print(f"  • Probabilitas terlambat           : {prob_late:.1%}")
    print(f"  • Potensi keterlambatan (95% CI)   : {days_at_risk:.1f} bulan")

# 3. Critical path
print(f"\nTAHAPAN KRITIS (Critical Path Analysis):")
critical_probs = simulator.hitung_critical_path()
for stage, data in critical_probs.sort_values('probability', ascending=False).iterrows():
    label = "🔴 KRITIS" if data['probability'] > 0.7 else ("🟡 PERLU PERHATIAN" if data['probability'] > 0.3 else "🟢 Aman")
    print(f"  • {stage.replace('_',' '):<35}: {data['probability']:.1%}  {label}")

# Rekomendasi buffer
print(f"\nREKOMENDASI MANAJEMEN RISIKO:")
safety_buffer = np.percentile(total_duration, 80) - total_duration.mean()
contingency = np.percentile(total_duration, 95) - total_duration.mean()
print(f"• Safety Buffer (80% confidence)   : +{safety_buffer:.1f} bulan")
print(f"• Contingency Reserve (95%)        : +{contingency:.1f} bulan")
print(f"• Rekomendasi jadwal               : {total_duration.mean():.1f} + {safety_buffer:.1f} = {total_duration.mean() + safety_buffer:.1f} bulan")


# ============================================================================
# 7. VISUALISASI
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SIMULASI MONTE CARLO\nEstimasi Waktu Pembangunan Gedung FITE 5 Lantai',
             fontsize=15, fontweight='bold', y=0.99)

mean_duration = total_duration.mean()
median_duration = np.median(total_duration)
ci_80 = np.percentile(total_duration, [10, 90])
ci_95 = np.percentile(total_duration, [2.5, 97.5])

# --- PLOT 1: Distribusi Durasi Total ---
ax1 = axes[0, 0]
ax1.hist(total_duration, bins=60, edgecolor='black', alpha=0.7,
         density=True, color='steelblue', label='Distribusi Durasi')
ax1.axvline(mean_duration, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_duration:.1f} bln')
ax1.axvline(median_duration, color='green', linestyle='--', linewidth=2,
            label=f'Median: {median_duration:.1f} bln')
ax1.axvspan(ci_80[0], ci_80[1], alpha=0.2, color='yellow', label='80% CI')
ax1.axvspan(ci_95[0], ci_95[1], alpha=0.1, color='orange', label='95% CI')

# Deadline markers
for dl, color in zip([16, 20, 24], ['purple', 'darkorange', 'darkgreen']):
    ax1.axvline(dl, color=color, linestyle=':', linewidth=1.5, label=f'Deadline {dl} bln')

stats_text = (f"Mean: {mean_duration:.1f} bln\n"
              f"Median: {median_duration:.1f} bln\n"
              f"Std Dev: {total_duration.std():.1f} bln\n"
              f"Min: {total_duration.min():.1f} bln\n"
              f"Max: {total_duration.max():.1f} bln\n"
              f"80% CI: [{ci_80[0]:.1f}, {ci_80[1]:.1f}]\n"
              f"95% CI: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=8.5,
         verticalalignment='top', bbox=props, fontfamily='monospace')

ax1.set_xlabel('Durasi Total Proyek (Bulan)', fontweight='bold')
ax1.set_ylabel('Densitas Probabilitas', fontweight='bold')
ax1.set_title('Distribusi Durasi Total Proyek', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- PLOT 2: Kurva Probabilitas Penyelesaian ---
ax2 = axes[0, 1]
deadlines_range = np.arange(8, 35, 0.5)
completion_probs = [np.mean(total_duration <= dl) for dl in deadlines_range]

ax2.plot(deadlines_range, completion_probs, linewidth=2.5, color='darkblue')
ax2.fill_between(deadlines_range, completion_probs, alpha=0.3, color='lightblue')

ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='50%')
ax2.axhline(0.8, color='green', linestyle='--', linewidth=1.5, label='80%')
ax2.axhline(0.95, color='blue', linestyle='--', linewidth=1.5, label='95%')

# Titik deadline studi kasus
for dl, col in zip([16, 20, 24], ['purple', 'darkorange', 'darkgreen']):
    prob = np.mean(total_duration <= dl)
    ax2.scatter(dl, prob, s=120, color=col, zorder=5, edgecolor='black')
    ax2.annotate(f'{dl} bln\n{prob:.1%}', (dl, prob),
                 textcoords="offset points", xytext=(8, 8),
                 fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                 arrowprops=dict(arrowstyle="->"))

ax2.set_xlabel('Deadline (Bulan)', fontweight='bold')
ax2.set_ylabel('Probabilitas Selesai Tepat Waktu', fontweight='bold')
ax2.set_title('Kurva Probabilitas Penyelesaian Proyek', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

# --- PLOT 3: Critical Path ---
ax3 = axes[1, 0]
critical_analysis = simulator.hitung_critical_path().sort_values('probability', ascending=True)
colors_bar = ['red' if p > 0.7 else ('orange' if p > 0.3 else 'lightcoral')
              for p in critical_analysis['probability']]

bars = ax3.barh(range(len(critical_analysis)), critical_analysis['probability'],
                color=colors_bar, edgecolor='darkred', linewidth=1.2, height=0.7)

for i, (bar, prob) in enumerate(zip(bars, critical_analysis['probability'])):
    ax3.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
             f'{prob:.1%}', va='center', fontweight='bold', fontsize=9)

ax3.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax3.axvline(x=0.7, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='Batas Kritis (70%)')
ax3.set_yticks(range(len(critical_analysis)))
ax3.set_yticklabels([s.replace('_', '\n') for s in critical_analysis.index], fontsize=8.5, fontweight='bold')
ax3.set_xlabel('Probabilitas Menjadi Critical Path', fontweight='bold')
ax3.set_title('Analisis Critical Path per Tahapan', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_xlim(0, 1.0)
ax3.grid(True, alpha=0.3, axis='x')

# --- PLOT 4: Boxplot Durasi per Tahapan ---
ax4 = axes[1, 1]
stage_names = list(simulator.stages.keys())
stage_data = [results[s] for s in stage_names]
box = ax4.boxplot(stage_data, vert=True, patch_artist=True,
                  labels=[s.replace('_', '\n') for s in stage_names],
                  widths=0.65,
                  medianprops=dict(color='red', linewidth=2),
                  boxprops=dict(linewidth=1.5),
                  whiskerprops=dict(linewidth=1.5))

palette = plt.cm.Set3(np.linspace(0, 1, len(stage_names)))
for patch, color in zip(box['boxes'], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for i, stage in enumerate(stage_names, 1):
    ax4.scatter(i, results[stage].mean(), color='darkblue', s=80, zorder=5, edgecolor='black')

ax4.set_ylabel('Durasi (Bulan)', fontweight='bold')
ax4.set_title('Distribusi Durasi per Tahapan', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', labelsize=7.5)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.97])

footer = (f"Simulasi Monte Carlo: {NUM_SIMULATIONS:,} iterasi  |  "
          f"Tahapan: {len(stage_names)}  |  "
          f"Rata-rata total: {mean_duration:.1f} bulan")
plt.figtext(0.5, 0.005, footer, ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))

plt.savefig('/mnt/user-data/outputs/hasil_simulasi_gedung_fite.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nVisualisasi disimpan ke: hasil_simulasi_gedung_fite.png")


# ============================================================================
# 8. ANALISIS SENSITIVITAS: PENAMBAHAN RESOURCE
# ============================================================================
print("\n" + "=" * 70)
print("ANALISIS SENSITIVITAS: PENGARUH PENAMBAHAN RESOURCE")
print("=" * 70)

class AnalisaResource:
    """Analisis dampak penambahan resource terhadap proyek konstruksi."""

    def __init__(self, simulation_results, project_stages):
        self.results = simulation_results
        self.stages = project_stages
        self.resource_costs = {
            'pekerja_khusus':   {'cost_per_month': 8_000_000,  'productivity_gain': 0.25},
            'alat_berat':       {'cost_per_month': 25_000_000, 'productivity_gain': 0.35},
            'insinyur':         {'cost_per_month': 15_000_000, 'productivity_gain': 0.20},
            'mandor_senior':    {'cost_per_month': 12_000_000, 'productivity_gain': 0.22},
            'konsultan_teknis': {'cost_per_month': 20_000_000, 'productivity_gain': 0.15},
        }

    def hitung_dampak(self, stage_name, resource_type, quantity, duration_months):
        resource_params = self.resource_costs[resource_type]
        base_improvement = resource_params['productivity_gain']
        actual_improvement = 1 - (base_improvement * min(quantity / 3, 1))

        # Skenario dengan resource tambahan
        scenario_results = self.results.copy()
        scenario_results[stage_name] = scenario_results[stage_name] * actual_improvement

        # Hitung total durasi dengan dependensi
        scenario_totals = []
        for idx in range(len(self.results)):
            stage_times = {}
            for curr_stage in self.stages.keys():
                deps = self.stages[curr_stage].dependencies
                start_time = 0 if not deps else max(stage_times.get(dep, 0) for dep in deps)
                duration = (scenario_results.loc[idx, curr_stage]
                            if curr_stage == stage_name
                            else self.results.loc[idx, curr_stage])
                stage_times[curr_stage] = start_time + duration
            scenario_totals.append(max(stage_times.values()))

        baseline_mean = self.results['Total_Duration'].mean()
        optimized_mean = np.mean(scenario_totals)
        duration_reduction = baseline_mean - optimized_mean
        percent_improvement = (duration_reduction / baseline_mean) * 100

        total_cost = resource_params['cost_per_month'] * quantity * duration_months
        cost_saving = duration_reduction * 150_000_000  # Rp 150 juta/bulan biaya proyek
        net_benefit = cost_saving - total_cost
        roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0

        return {
            'stage': stage_name,
            'resource_type': resource_type,
            'quantity': quantity,
            'duration_months': duration_months,
            'baseline_mean': baseline_mean,
            'optimized_mean': optimized_mean,
            'duration_reduction': duration_reduction,
            'percent_improvement': percent_improvement,
            'total_cost': total_cost,
            'net_benefit': net_benefit,
            'roi': roi,
            'scenario_totals': scenario_totals
        }


analyzer = AnalisaResource(results, simulator.stages)

# Skenario optimasi untuk studi kasus gedung FITE
optimization_scenarios = [
    {'stage': 'Struktur_Beton_Lantai',      'resource_type': 'alat_berat',       'quantity': 2, 'duration_months': 4},
    {'stage': 'Struktur_Pondasi',           'resource_type': 'alat_berat',       'quantity': 1, 'duration_months': 2},
    {'stage': 'Instalasi_Peralatan_Lab',    'resource_type': 'konsultan_teknis', 'quantity': 2, 'duration_months': 2},
    {'stage': 'Instalasi_MEP',             'resource_type': 'insinyur',          'quantity': 2, 'duration_months': 3},
    {'stage': 'Finishing_Interior',         'resource_type': 'pekerja_khusus',   'quantity': 3, 'duration_months': 3},
    {'stage': 'Pemasangan_Dinding_Fasad',   'resource_type': 'pekerja_khusus',   'quantity': 2, 'duration_months': 2},
    {'stage': 'Uji_Kelayakan_Serah_Terima', 'resource_type': 'insinyur',         'quantity': 1, 'duration_months': 1},
]

scenario_results = []
print(f"\n{'No.':<4} {'Tahapan':<32} {'Resource':<20} {'Jml':<5} {'Durasi':<7} {'Pengurangan':<14} {'ROI'}")
print("-" * 95)
for i, sc in enumerate(optimization_scenarios, 1):
    r = analyzer.hitung_dampak(sc['stage'], sc['resource_type'], sc['quantity'], sc['duration_months'])
    scenario_results.append(r)
    print(f"  {i}  {r['stage'].replace('_',' '):<32} {r['resource_type']:<20} {r['quantity']:<5} "
          f"{r['duration_months']} bln   {r['duration_reduction']:.2f} bln ({r['percent_improvement']:.1f}%)   {r['roi']:.0f}%")

best_roi = max(scenario_results, key=lambda x: x['roi'])
best_reduction = max(scenario_results, key=lambda x: x['duration_reduction'])
print(f"\n🏆 ROI Terbaik     : {best_roi['stage'].replace('_',' ')} ({best_roi['roi']:.0f}%)")
print(f"🏆 Reduksi Terbesar: {best_reduction['stage'].replace('_',' ')} ({best_reduction['duration_reduction']:.2f} bulan)")

print(f"\nProb. selesai DENGAN percepatan kombinasi (3 skenario terbaik):")
for dl in [16, 20, 24]:
    baseline_p = np.mean(total_duration <= dl)
    print(f"  Deadline {dl} bulan -> Baseline: {baseline_p:.1%}")

print("\n" + "=" * 70)
print("SIMULASI DAN ANALISIS SELESAI")
print("=" * 70)