"""
hmm_pipeline_real_data.py
=====================
Pipeline complet per a la cerca HMM de bosons superradiants
amb dades reals de GW231123 (O4a LIGO).

Estructura
----------
1. Configuració global
2. calcula_parametres_hmm()   — paràmetres T_sft / T_coh per garantir viterbi
3. algoritme_viterbi()        — Viterbi amb spin-up
4. build_B()                  — matriu B amb dades reals
5. build_simulated_B()        — matriu B amb senyal injectat a partir de SFTs reals
6. get_sqrtSX_from_sfts()     — ASD real del detector a partir de llistes predefinides
7. Data helpers
   · get_data()               — descarrega dades de GWOSC
   · trossejar_i_crear_cache()— divideix .gwf en trossos
   · make_sfts()              — genera SFTs amb lalpulsar
8. Visualització
   · visualize_B()
   · compute_n_plot_viterbi()
9. upper_limit_loop()         — bucle de límits superiors amb diverses opcions
10. __main__                  — pipeline complet tot junt
"""

# ── Importacions ──────────────────────────────────────────────────────────────
import os
import gc
import glob
import logging
import subprocess
import concurrent.futures
import shutil
import time
import functools
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyfstat
from gwpy.timeseries import TimeSeries

# ── Silenciem logs de pyfstat ─────────────────────────────────────────────────
logging.getLogger('pyfstat').setLevel(logging.WARNING)
logger_temps = logging.getLogger('temps')
logger_temps.setLevel(logging.INFO)

def cronometra(func):
     """Decorator que mesura i registra el temps d'execució d'una funció."""
     @functools.wraps(func)
     def wrapper(*args, **kwargs):
        t0     = time.perf_counter()
        result = func(*args, **kwargs)
        dt     = time.perf_counter() - t0
        logger_temps.info(f"{func.__name__}  →  {dt:.1f}s  ({dt/60:.2f} min)")
        return result
     return wrapper


# =============================================================================
# 1. CONFIGURACIÓ GLOBAL
# =============================================================================
# Coordenades del cel per a GW231123 (radians)
ALPHA_GW = 3.37
DELTA_GW  = 0.45

actual_dir = os.path.basename(os.getcwd()) # comprovem que estem a la root del repo
if actual_dir == "src":
    print('You are launching the code from src/. This may create files in src/ and take ' \
    'much longer to run.')
    print("Please launch the code from the root, This is: python src/hmm_pipeline_real_data.py")
    sys.exit(1)



needed_dirs = [
    "data/raw",
    "data/processed/fake_data",
    "results/simulations/B_matrix",
    "results/simulations/V_matrix",
    "results/injections"
]

for dir in needed_dirs:
    os.makedirs(dir, exist_ok=True)



CSV_PATH  = "data/resultats_simulacio.csv"

# si no tinc la simulacio feta, es fa

if not os.path.exists(CSV_PATH):
    print(f"Creant CSV de simulació: {CSV_PATH}")
    subprocess.run(['python', 'superrad_script.py'], check=True)

# Temps GPS d'inici de les dades reals
T_START = 1384788400

# Rutes als efemèrides (ajusta si cal)
RUTA_EARTH = (
    "/home/marc81/root_trial/envs/gw_fix/lib/python3.10/"
    "site-packages/solar_system_ephemerides/ephemerides/"
    "earth/earth00-40-DE405.dat.gz"
)
RUTA_SUN = (
    "/home/marc81/root_trial/envs/gw_fix/lib/python3.10/"
    "site-packages/solar_system_ephemerides/ephemerides/"
    "sun/sun00-40-DE405.dat.gz"
)


# =============================================================================
# 2. CÀLCUL DE PARÀMETRES HMM
# =============================================================================

def calcula_parametres_hmm(
    fdot_max,
    n_min=5,
    drift_bin_max=0.9,
    tolerancia_tcoh=0.90,
):
    """
    Donada una fdot_max (Hz/s), troba el millor parell (T_sft, T_coh)
    que compleix la física del drift i maximitza T_coh.

    Paràmetres
    ----------
    fdot_max       : derivada màxima de freqüència |Ḟ| (Hz/s)
    n_min          : nombre mínim de SFTs per segment coherent
    drift_bin_max  : fracció màxima de bin que pot driftar en T_coh
    tolerancia_tcoh: acceptem T_coh >= tolerancia * T_coh_max

    Retorna
    -------
    dict amb claus: T_sft, T_coh, N_SFTs, T_coh_max_fisic,
                    T_coh_max_drift, deltaf, drift_Hz, drift_bin


    Note:
    LVK demana T_coh/T_sft >= 4 i nombre enter. Jo exigeixo que sigui 5 donat que he trobat problemes
    en algun moment. A més, el drift_bin_max no caldria, pero es per assegurar no trobar combinacions
    molt justes que causin qualsevol problema numèric que faci que Viterbi no funcioni bé.
    """
    
    fdot_max = abs(fdot_max)
    if fdot_max <= 0:
        raise ValueError("fdot_max ha de ser positiu.")

    tcoh_max_fisic = 1.0 / np.sqrt(2.0 * fdot_max)
    tcoh_max_drift = np.sqrt(drift_bin_max / (2.0 * fdot_max))

    t_sfts_permesos = [16, 32, 64, 128, 256, 512, 1024, 1800]
    candidats_valids = []

    for t_sft in t_sfts_permesos:
        if t_sft > tcoh_max_drift:
            continue
        n_candidat = int(tcoh_max_drift // t_sft)
        if n_candidat < n_min:
            continue
        tcoh_candidat = n_candidat * t_sft
        deltaf        = 1.0 / (2.0 * tcoh_candidat)
        drift_bin     = (fdot_max * tcoh_candidat) / deltaf
        if drift_bin <= drift_bin_max:
            candidats_valids.append({
                "tsft": t_sft, "tcoh": tcoh_candidat, "n": n_candidat,
                "deltaf": deltaf, "drift_bin": drift_bin,
            })

    if not candidats_valids:
        raise ValueError(f"No hi ha combinacions vàlides per fdot_max={fdot_max:.3e}")

    max_tcoh = max(c["tcoh"] for c in candidats_valids)
    bons     = [c for c in candidats_valids if c["tcoh"] >= tolerancia_tcoh * max_tcoh]
    millor   = max(bons, key=lambda c: c["tsft"])

    return {
        "T_sft":           millor["tsft"],
        "T_coh":           millor["tcoh"],
        "N_SFTs":          millor["n"],
        "T_coh_max_fisic": tcoh_max_fisic,
        "T_coh_max_drift": tcoh_max_drift,
        "deltaf":          millor["deltaf"],
        "drift_Hz":        fdot_max * millor["tcoh"],
        "drift_bin":       millor["drift_bin"],
    }


# =============================================================================
# 3. ALGORITME VITERBI AMB SPIN-UP
# =============================================================================

def algoritme_viterbi(
    B_matrix,
    f_min,
    deltaf,
    salts_permesos=[0, 1],
    probs_transicio={0: 0.5, 1: 0.5},
):
    """
    Viterbi amb constraint de spin-up: la freqüència només pot
    mantenir-se igual o pujar un bin per segment.

    Paràmetres
    ----------
    B_matrix       : (num_bins, num_segments) — estadística 2F
    f_min          : freqüència mínima del rang de cerca (Hz)
    deltaf         : resolució de bin (Hz)
    salts_permesos : llista de salts permesos [0, 1]
    probs_transicio: probabilitat de cada salt

    Retorna
    -------
    cami_optim_freqs : (num_segments,) freqüències del camí òptim
    cami_optim_bins  : (num_segments,) bins del camí òptim
    V                : (num_bins, num_segments) matriu acumulada de Viterbi
    """
    num_bins, num_segments = B_matrix.shape

    V = np.full_like(B_matrix, -np.inf, dtype=float)
    P = np.zeros_like(B_matrix, dtype=int)

    log_p = {s: np.log(p) for s, p in probs_transicio.items()}

    V[:, 0] = B_matrix[:, 0] # inicialitzem la matriu

    for t in range(1, num_segments):
        for i in range(num_bins):
            max_score          = -np.inf # (és com posar 0)
            millor_bin_anterior = -1
            for salt in salts_permesos: # recordar q es matriu bidiagonal
                j = i - salt # o i o i-1 
                if 0 <= j < num_bins:
                    score = V[j, t - 1] + log_p[salt]
                    if score > max_score:
                        max_score           = score # un score per cada salt, agafem el millor
                        millor_bin_anterior = j # pel backtracing, guardem d'on vinc
            V[i, t] = B_matrix[i, t] + max_score
            P[i, t] = millor_bin_anterior

    cami_optim_bins       = np.zeros(num_segments, dtype=int)
    cami_optim_bins[-1]   = np.argmax(V[:, -1])
    for t in range(num_segments - 1, 0, -1):
        cami_optim_bins[t - 1] = P[cami_optim_bins[t], t]

    cami_optim_freqs = f_min + cami_optim_bins * deltaf
    return cami_optim_freqs, cami_optim_bins, V


# =============================================================================
# 4. MATRIU B — DADES REALS
# =============================================================================

def build_B(sft_path, F0, F1, tau_gw, massa=None):
    """
    Calcula la matriu B de l'HMM usant SFTs de dades reals.

    Paràmetres
    ----------
    sft_path : glob pattern dels SFTs per a aquesta massa
               (ex. 'sfts_1.00e-13eV/*.sft')
    F0       : freqüència inicial (Hz)
    F1       : derivada de freqüència (Hz/s)
    tau_gw   : durada del senyal (s)
    massa    : valor de la massa en eV (només per al nom del fitxer .npy)

    Retorna
    -------
    B_matrix : np.ndarray (num_freq_bins, num_segments)
    """
    params     = calcula_parametres_hmm(F1, n_min=5, drift_bin_max=0.85)
    T_coh      = params["T_coh"]
    deltaf     = params["deltaf"]

    T_obs        = tau_gw + 1800
    num_segments = int(np.round(T_obs / T_coh))

    # Limitem num_segments a les dades realment disponibles als SFTs.
    # Parsegem el nom dels fitxers SFT (convenció T050017) per obtenir
    # el GPS d'inici i durada sense dependre de get_sft_as_arrays.
    sfts_disponibles = sorted(glob.glob(sft_path))
    if sfts_disponibles:
        try:
            # Usem get_sft_as_arrays per obtenir els timestamps reals dels SFTs.
            # times_dict: {detector: np.ndarray de GPS starts}
            _, times_dict, _ = pyfstat.utils.get_sft_as_arrays(sft_path)
            all_times = np.concatenate(list(times_dict.values()))
            t_sft_end = float(all_times.max()) + T_coh
            max_segs  = int((t_sft_end - T_START) / T_coh)
            if num_segments > max_segs:
                print(f"  Avís: T_obs ({T_obs:.0f}s) supera les dades disponibles "
                      f"({t_sft_end - T_START:.0f}s). "
                      f"Reduint de {num_segments} a {max_segs} segments.")
                num_segments = max(1, max_segs)
        except Exception as e:
            print(f"  Avís: no s'ha pogut llegir el rang dels SFTs ({e}).")

    f_cerca_min   = F0 - 0.1
    f_cerca_max   = F0 + 1.7 + abs(F1) * T_obs
    num_freq_bins = int(np.round((f_cerca_max - f_cerca_min) / deltaf)) + 1

    B_matrix = np.zeros((num_freq_bins, num_segments))

    for t in range(num_segments):
        t_min = T_START + t * T_coh
        t_max = t_min + T_coh

        search = pyfstat.GridSearch(
            label=f"segment_{t}",
            outdir="real_data/output_B_matrix",
            sftfilepattern=sft_path,
            F0s=[f_cerca_min, f_cerca_max, deltaf],
            F1s=[0.0], F2s=[0.0],
            Alphas=[ALPHA_GW], Deltas=[DELTA_GW],
            tref=t_min, minStartTime=t_min, maxStartTime=t_max,
        )
        try:
            search.run()
            if search.data is not None:
                twoF = search.data['twoF']
                mlen = min(len(twoF), num_freq_bins)
                B_matrix[:mlen, t] = twoF[:mlen]
                print(f"  Segment {t}: 2F ∈ [{twoF.min():.3e}, {twoF.max():.3e}]")
            else:
                print(f"  Segment {t}: cap dada (gap).")
        except RuntimeError as e:
            pass
        finally:
            del search
            gc.collect()

    print(f"Matriu B real calculada: {B_matrix.shape}")
    tag = f"{massa:.2e}eV" if massa is not None else f"{F0:.2f}Hz"
    np.save(f"B_matrix_real_{tag}.npy", B_matrix)
    return B_matrix




# =============================================================================
# 5.1 MATRIU B — DADES SIMULADES (INJECCIÓ) NO SÉ SI ESTÀ BÉ
# =============================================================================
# =============================================================================
# 5.2 MATRIU B — DADES SIMULADES A PARTIR D SFTS REALS (INJECCIÓ) NO SÉ SI ESTÀ BÉ
# =============================================================================
@cronometra
def build_simulated_B_from_real_sfts(
    sft_pattern, F0, F1, h0, tau_gw,
    seed=42,
    sqrtSX=1e-23,
    num_segments_max=None,
):
    params     = calcula_parametres_hmm(F1, n_min=5, drift_bin_max=0.85)
    T_sft      = params["T_sft"]
    T_coh      = params["T_coh"]
    deltaf     = params["deltaf"]

    T_obs        = tau_gw + 1800
    N_segments   = int(np.round(T_obs / T_coh))
    random_seed = np.random.RandomState(seed)

    F0_seed = F0 + random_seed.uniform(-0.005, 0.005) # per variar la freq d'injecció entre realitzacions

    # Limitem al nombre real de segments disponibles a les dades reals
    if num_segments_max is not None:
        N_segments = min(N_segments, num_segments_max)

    T_obs_real   = N_segments * T_coh

    f_cerca_min  = F0_seed - 0.1
    f_cerca_max  = F0_seed + 1.7 + abs(F1) * T_obs_real
    num_freq_bins = int(np.round((f_cerca_max - f_cerca_min) / deltaf)) + 1

    dist_baix      = F0_seed - f_cerca_min
    dist_dalt      = f_cerca_max - F0_seed
    Band_necessari = 2 * max(dist_baix, dist_dalt) + 8.0

    B_matrix = np.zeros((num_freq_bins,N_segments))

    sim_outdir = f"results/injections/B_sim_from_real_sfts/sim_seed{seed}_F0{F0:.2f}"
    os.makedirs(sim_outdir,exist_ok=True)

    cosi_inj = random_seed.uniform(-1.0, 1.0)
    psi_inj  = random_seed.uniform(-np.pi/4, np.pi/4)
    phi_inj  = random_seed.uniform(0.0, 2*np.pi)
    
    for t in range(N_segments):
        t_min = T_START + t * T_coh
        t_max = t_min + T_coh

        search = pyfstat.GridSearch(
            label=f"sim_seg{t}",
            outdir=sim_outdir, 
            sftfilepattern=sft_pattern,  # POINT TO REAL DATA
            F0s=[f_cerca_min, f_cerca_max, deltaf],
            F1s=[0.0], F2s=[0.0],
            Alphas=[ALPHA_GW], Deltas=[DELTA_GW],
            tref=t_min, minStartTime=t_min, maxStartTime=t_max,
            
            # INJECT ON THE FLY IN RAM
            injectSources={
                "F0": F0_seed,
                "F1": F1,
                "h0": h0,
                "cosi": cosi_inj, #optim
                "psi": psi_inj, #optim
                "phi": phi_inj, #optim
                "Alpha": ALPHA_GW,
                "Delta": DELTA_GW,
                "refTime": T_START
            }
        )
        try:
            search.run()
            if search.data is not None:
                twoF = search.data['twoF']
                mlen = min(len(twoF), num_freq_bins)
                B_matrix[:mlen, t] = twoF[:mlen]
        except RuntimeError:
            pass # Hi haurà gaps com a les dades reals !!!
        finally:
            del search
            gc.collect()
            shutil.rmtree(sim_outdir, ignore_errors=True)

    return B_matrix

# =============================================================================
# 6. ESTIMACIÓ DE sqrtSX DES DELS SFTs REALS
# =============================================================================

def get_sqrtSX_from_sfts(F0):
    """
    Retorna l'ASD del detector LIGO a la freqüència F0 interpolant
    la corba de soroll oficial d'O4a (GWOSC / LIGO-T2400074).

    No llegeix els SFTs directament perquè la normalització de les
    amplituds SFT requereix el T_sft i la calibració del detector,
    la qual cosa fa que l'estimació directa sigui poc fiable.

    Paràmetres
    ----------
    F0          : freqüència d'interès (Hz)


    Retorna
    -------
    sqrtSX : float (strain/√Hz)
    """
    # Corba de soroll oficial O4a de GWOSC (valors aproximats per banda).
    # Font: https://gwosc.org/O4/  — ASD mediana H1+L1 en strain/sqrt(Hz).
    # Usem una taula de punts representatius i interpolem a F0.
    # Ref: LIGO-T2400074 (O4a sensitivity)
    freqs_asd = np.array([
        10,   20,   30,   40,   50,   60,   70,   80,  100,
       120,  150,  200,  300,  400,  500,  600,  800, 1000,
      1200, 1500, 2000, 3000, 4000, 5000
    ], dtype=float)

    # ASD H1 O4a (strain/sqrt(Hz)) — valors representatius
    asd_vals = np.array([
        4.0e-22, 8.0e-23, 1.5e-23, 7.0e-24, 4.5e-24, 3.5e-24, 3.2e-24, 3.1e-24, 3.0e-24,
        2.9e-24, 2.8e-24, 2.9e-24, 3.2e-24, 3.8e-24, 4.5e-24, 5.5e-24, 8.0e-24, 1.2e-23,
        1.8e-23, 3.0e-23, 6.0e-23, 2.0e-22, 5.0e-22, 1.0e-21
    ], dtype=float)

    sqrtSX = float(np.interp(F0, freqs_asd, asd_vals))
    print(f"  sqrtSX interpolat de la corba O4a a {F0:.2f} Hz: {sqrtSX:.2e}")
    return sqrtSX


# =============================================================================
# 7. DATA HELPERS
# =============================================================================

def get_data(gps_start=T_START, duration=18000):
    """
    Descarrega dades obertes de GWOSC per a H1 i L1 i les guarda
    en format GWF amb la convenció T050017.
    """
    print("Descarregant dades de GWOSC...")
    gps_end = gps_start + duration
    try:
        print("  H1...")
        strain_H1 = TimeSeries.fetch_open_data('H1', gps_start, gps_end,
                                               timeout=60, verbose=True)
        print("  L1...")
        strain_L1 = TimeSeries.fetch_open_data('L1', gps_start, gps_end,
                                               timeout=60, verbose=True)
    except Exception as e:
        print(f"Error descarregant dades: {e}")
        raise

    strain_H1.name = "H1:STRAIN"
    strain_L1.name = "L1:STRAIN"

    fitxer_H1 = f'data/raw/H-H1_GWOSC-{gps_start}-{duration}.gwf'
    fitxer_L1 = f'data/raw/L-L1_GWOSC-{gps_start}-{duration}.gwf'
    strain_H1.write(fitxer_H1, format='gwf')
    strain_L1.write(fitxer_L1, format='gwf')
    print(f"Guardat: {fitxer_H1}, {fitxer_L1}")

    return fitxer_H1, fitxer_L1


def trossejar_i_crear_cache(fitxer_original, canal, chunk_size=900):
    """
    Divideix un fitxer GWF llarg en trossos de chunk_size segons
    i crea el fitxer .lcf (LAL cache) corresponent.
    """
    print(f"Trossejant {fitxer_original} ({canal})...")
    ts   = TimeSeries.read(fitxer_original, format='gwf', channel=canal)
    t0   = int(ts.t0.value)
    tend = int(ts.span[1])

    dir_trossos = "data/raw/frames_temporals"
    os.makedirs(dir_trossos, exist_ok=True)
    nom_cache = f"{canal[:2]}_data.lcf"

    with open(nom_cache, "w") as f_cache:
        for start in range(t0, tend, chunk_size):
            end = min(start + chunk_size, tend)
            dur = end - start
            chunk    = ts.crop(start, end)
            nom_tros = f"{canal[0]}-{canal[:2]}_GWOSC-{start}-{dur}.gwf"
            ruta     = os.path.abspath(os.path.join(dir_trossos, nom_tros))
            chunk.write(ruta, format='gwf')
            f_cache.write(
                f"{canal[0]} {canal[:2]}_GWOSC {start} {dur} file://localhost{ruta}\n"
            )
    print(f"  Cache creat: {nom_cache}")
    return nom_cache


def make_sfts(csv_path, index=4, sft_output_dir=None,
              lcf_H1="H1_data.lcf", lcf_L1="L1_data.lcf", n_workers=6):
    """
    Genera SFTs a partir dels fitxers LCF usant lalpulsar_MakeSFTs,
    amb paral·lelització per trossos i detectors.

    Paràmetres
    ----------
    csv_path       : ruta al CSV
    index          : fila del CSV (determina T_sft, fmin, band per a la massa)
    sft_output_dir : directori on guardar els SFTs. Si és None,
                     usa 'sfts_<massa>eV' construït automàticament.
    lcf_H1, lcf_L1 : fitxers de cache LAL per a H1 i L1
    n_workers      : nombre de workers paral·lels
    """
    df     = pd.read_csv(csv_path)
    F0     = df["Freq_inicial"].iloc[index]
    F1     = df["Freqdot_inicial"].iloc[index]
    tau_gw = df["Tau_gw"].iloc[index]
    massa  = df["Massa"].iloc[index]

    if sft_output_dir is None:
        sft_output_dir = f"sfts_{massa:.2e}eV"
    sfts_dir = sft_output_dir
    os.makedirs(sfts_dir, exist_ok=True)

    params = calcula_parametres_hmm(F1, n_min=5, drift_bin_max=0.85)
    tsft   = params["T_sft"]
    T_obs  = tau_gw + 1800

    f_cerca_min = F0 - 0.1
    f_cerca_max = F0 + 1.7 + abs(F1) * T_obs
    fmin        = f_cerca_min - 8.0
    band        = np.ceil(f_cerca_max + 8.0 - fmin)

    def _llegeix_lcf(lcf_path):
        patches = []
        with open(lcf_path) as f:
            for linia in f:
                parts = linia.strip().split()
                if len(parts) < 5:
                    continue
                t0  = int(parts[2])
                dur = int(parts[3])
                patches.append((t0, t0 + dur))
        return patches

    def _genera_patch(lcf_path, canal, t0, t1, patch_id):
        comanda = [
            "lalpulsar_MakeSFTs",
            "--frame-cache",        lcf_path,
            "--channel-name",       canal,
            "--gps-start-time",     str(t0),
            "--gps-end-time",       str(t1),
            "--sft-duration",       str(tsft),
            "--start-freq",         str(fmin),
            "--band",               str(band),
            "--high-pass-freq",     "15",
            "--window-type",        "Tukey",
            "--window-param",       "0.001",
            "--observing-run",      "4",
            "--observing-kind",     "RUN",
            "--observing-revision", "1",
            "--sft-write-path",     sfts_dir,
        ]
        try:
            subprocess.run(comanda, check=True, capture_output=True)
            print(f"  ✓ {canal} patch {patch_id:03d} ({t0}–{t1})")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {canal} patch {patch_id:03d}: {e.stderr.decode()}")
            raise

    tasques = []
    for lcf_path, canal in [(lcf_H1, "H1:STRAIN"), (lcf_L1, "L1:STRAIN")]:
        patches = _llegeix_lcf(lcf_path)
        print(f"{canal}: {len(patches)} patches")
        for i, (t0, t1) in enumerate(patches):
            tasques.append((lcf_path, canal, t0, t1, i))

    print(f"Llançant {len(tasques)} tasques amb {n_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_genera_patch, *t) for t in tasques]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    print("✓ Tots els SFTs generats")


# =============================================================================
# 8. VISUALITZACIÓ
# =============================================================================

def visualize_B(B_matrix, F0, F1, tau_gw, titol="Matriu B"):
    """Heatmap de la matriu B (estadística 2F)."""
    T_obs       = tau_gw + 1800
    f_cerca_min = F0 - 0.1
    f_cerca_max = F0 + 1.7 + abs(F1) * T_obs

    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        B_matrix, aspect='auto', origin='lower',
        extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max],
        cmap='viridis',
    )
    plt.colorbar(im).set_label(r'Estadística $2\mathcal{F}$', fontsize=12)
    plt.xlabel('Temps d\'observació (h)', fontsize=12)
    plt.ylabel('Freqüència $f$ (Hz)', fontsize=12)
    plt.title(titol, fontsize=14)
    plt.tight_layout()
    nom = f"results/simulations/B_matrix/matriu_B_heatmap_{F0:.2f}Hz.png"
    plt.savefig(nom, dpi=300)
    print(f"Guardat: {nom}")
    plt.close()


def compute_n_plot_viterbi(B_matrix, F0, F1, tau_gw):
    """Executa Viterbi i genera la figura comparativa B vs V."""
    params      = calcula_parametres_hmm(F1, n_min=5, drift_bin_max=0.85)
    T_coh       = params["T_coh"]
    deltaf      = params["deltaf"]

    T_obs        = tau_gw + 1800
    num_segments = int(np.round(T_obs / T_coh))
    f_cerca_min  = F0 - 0.1
    f_cerca_max  = F0 + 1.7 + abs(F1) * T_obs

    freqs_v, _, V = algoritme_viterbi(B_matrix, f_cerca_min, deltaf)
    temps_h       = np.linspace(0, T_obs / 3600, num_segments)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    im1 = ax1.imshow(B_matrix, aspect='auto', origin='lower',
                     extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max], cmap='viridis')
    ax1.set_ylabel('Freqüència (Hz)', fontsize=12)
    ax1.set_title(r'Matriu $\mathcal{B}$ — estadística $2\mathcal{F}$', fontsize=14)
    fig.colorbar(im1, ax=ax1).set_label(r'$2\mathcal{F}$', fontsize=12)

    V_norm = V - V.max(axis=0, keepdims=True)
    im2 = ax2.imshow(V_norm, aspect='auto', origin='lower',
                     extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max], cmap='plasma')
    ax2.plot(temps_h, freqs_v, color='cyan', lw=2, ls='--', label='Camí Viterbi')
    ax2.set_xlabel('Temps d\'observació (h)', fontsize=12)
    ax2.set_ylabel('Freqüència (Hz)', fontsize=12)
    ax2.set_title(r'Matriu $\mathcal{V}$ de Viterbi', fontsize=14)
    ax2.legend(loc='upper right')
    fig.colorbar(im2, ax=ax2).set_label('Probabilitat relativa', fontsize=12)

    plt.tight_layout()
    nom = f"results/simulations/V_matrix/viterbi_{F0:.2f}Hz.png"
    plt.savefig(nom, dpi=300)
    print(f"Guardat: {nom}  |  Freqüència final: {freqs_v[-1]:.4f} Hz")
    plt.close()
    return freqs_v


# =============================================================================
# 9. BUCLE DE LÍMITS SUPERIORS
# =============================================================================

@cronometra
def upper_limit_loop(
    csv_path,
    sft_dir_template="sfts_{massa:.2e}eV",
    n_injections=100,
    confidence_threshold=0.80,
    limit_error=20
):
    """
    Per a cada massa del CSV, calcula la confiança d'exclusió comparant:
      · scores de N injeccions simulades al h0 predit per SuperRad
      · score obtingut de les dades reals

    Cada massa té el seu propi directori de SFTs perquè T_sft canvia
    amb la massa. El path es construeix amb sft_dir_template.

    Paràmetres
    ----------
    csv_path            : ruta al CSV (Massa, Freq_inicial, Freqdot_inicial,
                          Amplitud_h0, Tau_gw)
    sft_dir_template    : template del directori de SFTs per massa.
                          Ha de contenir '{massa}', ex:
                            'sfts_{massa:.2e}eV'   -> sfts_1.00e-13eV/
                          El glob final afegeix /*.sft automàticament.
    n_injections        : nombre de realitzacions de soroll per massa
    confidence_threshold: llindar per excloure la massa (ex. 0.30)
    fap_target          : FAP desitjada per al llindar de soroll (ex. 0.02 = 2%)
    n_noise_trials      : nombre de trials de soroll pur per estimar el llind
    limit_error         : nombre de bins de freq (deltaf) per considerar una detecció

    Retorna
    -------
    resultats : llista de dicts amb massa, score_real, confidence, excluded
    """
    df = pd.read_csv(csv_path)
    resultats = []


    for index, row in df.iterrows():
        massa  = row["Massa"]
        F0     = row["Freq_inicial"]
        F1     = row["Freqdot_inicial"]
        h0     = row["Amplitud_h0"]
        tau_gw = row["Tau_gw"]

        # ── Path dels SFTs específics per a aquesta massa ─────────────────────
        sft_dir     = sft_dir_template.format(massa=massa)
        sft_pattern = f"{sft_dir}/*.sft"

        print(f"\n{'='*60}")
        print(f"Massa {massa:.2e} eV  |  F0={F0:.2f} Hz  | F1={F1:.2e} Hz/s  |  h0={h0:.2e}")
        print(f"SFTs:  {sft_pattern}")
        print(f"{'='*60}")

        # Comprova que existeixin SFTs per a aquesta massa
        if not glob.glob(sft_pattern):
            print(f"  AVÍS: no s'han trobat SFTs a '{sft_pattern}'. Saltant massa.")
            continue

        params       = calcula_parametres_hmm(F1, n_min=5, drift_bin_max=0.85)
        T_coh        = params["T_coh"]
        deltaf       = params["deltaf"]
        T_obs        = tau_gw + 1800
        num_segments = int(np.round(T_obs / T_coh))
        f_cerca_min  = F0 - 0.1


        # ── Score de les dades reals ──────────────────────────────────────────
        print("Calculant B real...")
        
        B_real            = build_B(sft_pattern, F0, F1, tau_gw, massa=massa)
        num_segments_real = B_real.shape[1]
        # Correccio: escala sqrtSX perque les simulacions tinguin
        # la mateixa mediana de 2F que les dades reals
        sqrtSX    = get_sqrtSX_from_sfts(F0)
    
        _, _, V_real     = algoritme_viterbi(B_real, f_cerca_min, deltaf)
        score_real = np.max(V_real[:, -1]) / num_segments_real
        
        # Condició d'èxit estricta (idèntica a is_detection)
        print(f"  Score real: {score_real:.4f}  ({num_segments_real} segments)")


        # ── Scores de les injeccions ──────────────────────────────────────────
        # IMPORTANT: les simulacions han de tenir el mateix nombre de segments
        # que les dades reals per que els scores siguin comparables.
        print(f"Fent {n_injections} injeccions ({num_segments_real} segments cada una)...")
        injection_scores = []
        recoveries_ok = []  # Llista per guardar els èxits reals
        for seed in range(n_injections):
            B_sim = build_simulated_B_from_real_sfts(sft_pattern, F0, F1, h0, tau_gw,
                                                     seed=seed, sqrtSX=sqrtSX,
                                                     num_segments_max=num_segments_real)
            
            # Extraiem les freqüències simulades igual que a les dades reals
            freqs_v_sim, _, V_sim = algoritme_viterbi(B_sim, f_cerca_min, deltaf)
            freq_final_sim = freqs_v_sim[-1]
            freq_inicial_sim = freqs_v_sim[0]
            tolerancia_hz = 0.005 + (limit_error * deltaf)
            condicio_posicio = abs(freq_inicial_sim - F0) <= tolerancia_hz # neix on toca?
            condicio_cinematica = abs(freq_final_sim - freq_inicial_sim) <= (limit_error * deltaf) # es lia molt?
            s = np.max(V_sim[:, -1]) / num_segments_real
            
            if not condicio_posicio:
                print(f"Falla la freq ini {freq_inicial_sim-F0:.4f} Hz > {tolerancia_hz:.4f} Hz")
                print(f"La freq se'n va a {freq_inicial_sim:.4f} Hz")
            if not condicio_cinematica:
                print(f"Falla la freq final {freq_final_sim-freq_inicial_sim:.4f} Hz > {limit_error * deltaf:.4f} Hz")
            
            injection_scores.append(s)

            if (s>score_real):
                print(f"dif de scores: {s-score_real:.4f}")
            
            # Condició d'èxit estricta (idèntica a is_detection)

            is_recovered = (s > score_real) and (condicio_posicio) and (condicio_cinematica)
            
            if is_recovered:
                print(f"Recovered!{seed}")
            recoveries_ok.append(is_recovered)
            
            if (seed + 1) % 10 == 0:
                print(f"  Injeccions completades: {seed+1}/{n_injections}")

        injection_scores = np.array(injection_scores)
        recoveries_ok = np.array(recoveries_ok)
        
        # La confiança és la fracció d'injeccions SUPERADES i RECUPERADES AL LLOC CORRECTE
        confidence = float(np.mean(recoveries_ok))
        excluded   =(confidence >= confidence_threshold)

        print(f"  Score real:          {score_real:.4f}")
        print(f"  Score inj (mitjana): {injection_scores.mean():.4f} ± {injection_scores.std():.4f}")
        print(f"  Score inj (min):     {injection_scores.min():.4f}")
        print(f"  Score inj (max):     {injection_scores.max():.4f}")

        print(f"  Confiança: {confidence:.3f}  |  Exclosa: {excluded}")

        resultats.append({
            "massa":      massa,
            "amplitud_h0": h0,
            "score_real": score_real,
            "inj_mean":   float(injection_scores.mean()),
            "inj_std":    float(injection_scores.std()),
            "confidence": confidence,
            "excluded":   excluded,
        })

    # ── Gràfica de resultats ──────────────────────────────────────────────────
    masses      = [r["massa"] for r in resultats]
    confidences = [r["confidence"] for r in resultats]
    excluded    = [r["excluded"] for r in resultats]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['tab:red' if e else 'tab:blue' for e in excluded]
    ax.scatter(masses, confidences, c=colors, zorder=3, s=60)
    ax.plot(masses, confidences, color='gray', lw=1, alpha=0.5)
    ax.axhline(confidence_threshold, color='red', ls='--', lw=1.5,
               label=f'Llindar {confidence_threshold:.0%}')
    ax.set_xlabel('Boson Mass (eV)', fontsize=12)
    ax.set_ylabel('Exlusion Confidence', fontsize=12)
    ax.set_title('HMM Upper Limits — GW231123', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/injections/upper_limits_hmm{n_injections}_path.png", dpi=300)
    print("\nGràfica guardada: upper_limits_hmm_path.png")
    plt.close()

    # Guardem els resultats en CSV
    pd.DataFrame(resultats).to_csv("results/injections/upper_limits_resultats_path.csv", index=False)
    print("Resultats guardats: upper_limits_resultats_path.csv")

    return resultats


# =============================================================================
# 10. PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Template del directori de SFTs — cada massa té el seu propi directori
    # perquè T_sft canvia amb la massa.
    # Ex: massa=1.00e-13 eV  ->  sfts_1.00e-13eV/*.sft
    SFT_DIR_TEMPLATE = "data/processed/sfts_{massa:.2e}eV"

    # ── Pas 1: Descarregar dades (només si no existeixen) ────────────────────
    fitxers_gwf = glob.glob("data/raw/*GWOSC*.gwf")
    if not fitxers_gwf:
        print("\nPAS 1: Descarregant dades de GWOSC")
        fitxer_H1, fitxer_L1 = get_data()
    else:
        print(f"\nPAS 1: Fitxers GWF ja existents ({fitxers_gwf}), saltant.")
        print(fitxers_gwf)
        fitxer_H1 = [f for f in fitxers_gwf if os.path.basename(f).startswith('H')][0]
        fitxer_L1 = [f for f in fitxers_gwf if os.path.basename(f).startswith('L')][0]

    # ── Pas 2: Trossejar i crear caches ──────────────────────────────────────
    if not os.path.exists("data/raw/H1_data.lcf"):
        print("\nPAS 2: Creant caches LCF")
        trossejar_i_crear_cache(fitxer_H1, "H1:STRAIN")
        trossejar_i_crear_cache(fitxer_L1, "L1:STRAIN")
    else:
        print("\nPAS 2: Caches LCF ja existents, saltant.")

    # ── Pas 3: Generar SFTs per a totes les masses ───────────────────────────
    # make_sfts ha de cridar-se per cada fila del CSV (cada massa té T_sft
    # diferent i per tant un directori de SFTs diferent).
    df_csv = pd.read_csv(CSV_PATH)
    for idx, row in df_csv.iterrows():
        massa_idx   = row["Massa"]
        sft_dir_idx = SFT_DIR_TEMPLATE.format(massa=massa_idx)
        if not glob.glob(f"{sft_dir_idx}/*.sft"):
            print(f"\nPAS 3 [{massa_idx:.2e} eV]: Generant SFTs -> {sft_dir_idx}/")
            make_sfts(CSV_PATH, index=idx, sft_output_dir=sft_dir_idx)
        else:
            print(f"\nPAS 3 [{massa_idx:.2e} eV]: SFTs ja existents, saltant.")

    # ── Pas 4: Cerca principal (massa de l'index 4) ───────────────────────────
    index  = 6
    F0     = df_csv["Freq_inicial"].iloc[index]
    F1     = df_csv["Freqdot_inicial"].iloc[index]
    tau_gw = df_csv["Tau_gw"].iloc[index]
    massa  = df_csv["Massa"].iloc[index]

    print(f"\nPAS 4: Càlcul de la matriu B i Viterbi per a la massa principal | massa:  {massa:.2e} eV")
    sft_pattern_principal = f"{SFT_DIR_TEMPLATE.format(massa=massa)}/*.sft"
    B_real = build_B(sft_pattern_principal, F0, F1, tau_gw, massa=massa)
    visualize_B(B_real, F0, F1, tau_gw, titol="Matriu B — dades reals GW231123")
    compute_n_plot_viterbi(B_real, F0, F1, tau_gw)

    # ── Pas 5: Límits superiors sobre totes les masses ────────────────────────
    print("\nPAS 5: Bucle de límits superiors")
    resultats = upper_limit_loop(
        csv_path=CSV_PATH,
        sft_dir_template=SFT_DIR_TEMPLATE,
        n_injections=100,
        confidence_threshold=0.80,
    )

    # Resum final
    excloses = [r for r in resultats if r["excluded"]]
    print(f"\nMasses excloses al {0.80:.0%} de confiança:")
    for r in excloses:
        print(f"  {r['massa']:.2e} eV  (confiança={r['confidence']:.3f})")

