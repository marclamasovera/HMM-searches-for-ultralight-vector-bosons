import numpy as np
import matplotlib.pyplot as plt
import pyfstat
import pandas as pd
import gc
import matplotlib
matplotlib.use('Agg')  # Forcem Matplotlib a treballar sense pantalla
import logging
import os
import subprocess

CSV_PATH = "data/resultats_simulacio.csv"

if not os.path.exists(CSV_PATH):
    print(f"AVÍS: No s'ha trobat {CSV_PATH}. Generant les dades inicials amb SuperRad...")
    # Executa automàticament l'script de simulació
    subprocess.run(["python", "src/superrad_script.py"], check=True)
    print("✓ Dades generades correctament. Continuant amb el pipeline...")

logger = logging.getLogger('pyfstat')
logger.setLevel(logging.WARNING)  # O DEBUG per més detall, o WARNING per menys

if __name__ == "__main__":



    needed_dirs = [
        "data/raw",
        "data/processed/fake_data",
        "results/simulations/B_matrix",
        "results/simulations/V_matrix",
        "results/simulations/V_n_B_matrices",
        "results/injections/B_matrix",
        "results/injections/V_matrix",
        "results/injections/V_n_B_matrices",
        

    ]
    for dir in needed_dirs:
        os.makedirs(dir, exist_ok=True)

def calcula_parametres_hmm(
    fdot_max,
    n_min=5,
    drift_bin_max=0.9,
    tolerancia_tcoh=0.90 # Acceptem perdre fins a un 10% de SNR per guanyar velocitat
): 
    """ Serveix per donat una fdot max, trobar el T_cohi l'integer >= 4 per fer el tsft """


    fdot_max = abs(fdot_max)

    if fdot_max <= 0:
        raise ValueError("fdot_max ha de ser positiu.")

    tcoh_max_fisic = 1.0 / np.sqrt(2.0 * fdot_max) # el físic
    tcoh_max_drift = np.sqrt(drift_bin_max / (2.0 * fdot_max)) # el límit que poso jo (drift_max_bin < 1 sempre)

    t_sfts_permesos = [16, 32, 64, 128, 256, 512, 1024,1800] # per ajudar en termes computacionals
    candidats_valids = []

    # 1. Recopilem TOTES les combinacions que compleixen la física
    for t_sft in t_sfts_permesos:
        if t_sft > tcoh_max_drift:
            continue
            
        n_candidat = int(tcoh_max_drift // t_sft)

        if n_candidat >= n_min:
            tcoh_candidat = n_candidat * t_sft
            deltaf = 1.0 / (2.0 * tcoh_candidat)
            drift = fdot_max * tcoh_candidat
            drift_bin = drift / deltaf

            if drift_bin <= drift_bin_max:
                candidats_valids.append({
                    "tsft": t_sft,
                    "tcoh": tcoh_candidat,
                    "n": n_candidat,
                    "deltaf": deltaf,
                    "drift_bin": drift_bin
                })

    if not candidats_valids:
        raise ValueError(f"No hi ha combinacions vàlides per fdot_max={fdot_max:.3e}")

    # 2. Trobem quin és el temps màxim absolut que podríem aconseguir
    max_tcoh_absolut = max(c["tcoh"] for c in candidats_valids)

    # 3. Filtrem només els candidats "excel·lents" (que ens donen almenys un 90% d'aquest màxim)
    bons_candidats = [c for c in candidats_valids if c["tcoh"] >= tolerancia_tcoh * max_tcoh_absolut]

    # triem el que té el T_sft més gran
    millor = max(bons_candidats, key=lambda c: c["tsft"])

    drift_Hz = fdot_max * millor["tcoh"]

    return {
        "T_sft": millor["tsft"],
        "T_coh": millor["tcoh"],
        "N_SFTs": millor["n"],
        "T_coh_max_fisic": tcoh_max_fisic,
        "T_coh_max_drift": tcoh_max_drift,
        "deltaf": millor["deltaf"],
        "drift_Hz": drift_Hz,
        "drift_bin": millor["drift_bin"],
    }

######################################################################################################################
def algoritme_viterbi(
    B_matrix,
    f_min,
    deltaf,
    salts_permesos=[0, 1], # només puc pujar o quedar-me (constraint físic de spin-up)
    probs_transicio={0: 0.5, 1: 0.5} # Posem la probabilitat igual
):
    """
    Viterbi amb spin-up: la freqüència només pot mantenir-se igual o pujar.
    
    i = bin actual al temps t
    j = bin anterior al temps t-1
    salt = i - j
    
    Per spin-up:
        salt = 0  -> mateixa freqüència
        salt = 1  -> puja un bin
    """

    num_bins, num_segments = B_matrix.shape # des de l'input

    V = np.full_like(B_matrix, -np.inf, dtype=float)
    P = np.zeros_like(B_matrix, dtype=int)

    log_p = {salt: np.log(prob) for salt, prob in probs_transicio.items()} # fem log-likelihood

    V[:, 0] = B_matrix[:, 0] # la primera columna de V és la primera de B 

    for t in range(1, num_segments): # O(n_t)
        for i in range(num_bins): # (O(n_bins x n_t))
            max_score = -np.inf # posem tot a log(0) = -inf per trobar el màx (el camí)
            millor_bin_anterior = -1

            for salt in salts_permesos: # només 2 salts per spin-up, O(1) per iteració,
                                        #millora del viterbi clàssic, ja que només permetem 
                                        # pujar o quedar-se a un bin             
                j = i - salt

                if 0 <= j < num_bins: # si tenim un bin anterior vàlid
                    score_candidat = V[j, t-1] + log_p[salt] # calculem score

                    if score_candidat > max_score: # actualitzem el pas
                        max_score = score_candidat
                        millor_bin_anterior = j

            V[i, t] = B_matrix[i, t] + max_score # omplim la matriu V
            P[i, t] = millor_bin_anterior # guardem el pas d'on venim per fer el backtracking

    cami_optim_bins = np.zeros(num_segments, dtype=int) 

    cami_optim_bins[-1] = np.argmax(V[:, -1]) # quin es l'últim bin amb el millor score
                                              # serà l'últim pas del camí òptim, farem 
                                              # backtracking a partir d'aquí

    for t in range(num_segments - 1, 0, -1):
        cami_optim_bins[t-1] = P[cami_optim_bins[t], t]

    cami_optim_freqs = f_min + cami_optim_bins * deltaf

    return cami_optim_freqs, cami_optim_bins, V
#######################################################################################################################

def build_B(sft_path,csv_path):

    df = pd.read_csv(csv_path)
    F0 = df["Freq_inicial"].iloc[4]
    F1 = df["Freqdot_inicial"].iloc[4]
    tau_gw = df["Tau_gw"].iloc[4]

    params = calcula_parametres_hmm(
        F1,
        n_min=5,
        drift_bin_max=0.85
    )

    T_sft = params["T_sft"]
    T_coh = params["T_coh"]
    N_SFTs = params["N_SFTs"]


    T_obs = tau_gw + 1800 # Afegeixo 30 minuts extra per assegurar-me de cobrir tot el senyal
   
    num_segments = int(np.round(T_obs / T_coh))  # N_T (T_obs total / T_coh)

    T_obs_real = num_segments * T_coh

    t_start = 1384788400 

    deltaf = 1.0 / (2.0 * T_coh)        # Resolució del bin segons la regla de l'LVK

    # Coordenades del cel per a l'esdeveniment GW231123 (en radians)
    alpha_gw = 3.37
    delta_gw = 0.45

    f_cerca_min = F0 - 0.1
    f_cerca_max = F0 + 1.7 + abs(F1) * T_obs

    # Calculem el nombre de files (bins de freqüència)
    num_freq_bins = int(np.round((f_cerca_max - f_cerca_min) / deltaf)) + 1

    # Creem la Matriu B buida: (estats_ocults, passos_de_temps)
    B_matrix = np.zeros((num_freq_bins, num_segments))

  
    for t in range(num_segments):
        
        # Marquem l'inici i el final del segment coherent actual
        t_min = t_start + t * T_coh
        t_max = t_min + T_coh
        
    
        search = pyfstat.GridSearch(
            label=f"segment_{t}",
            outdir="results/simulations/B_matrix",
            sftfilepattern=sft_path, 
            F0s=[f_cerca_min, f_cerca_max, deltaf],# L'eix Y (les freqüències que avaluem)
            F1s=[0.0],              # No busquem deriva de freqüència en aquesta prova
            F2s=[0.0], 
            Alphas=[alpha_gw],
            Deltas=[delta_gw],
            tref=t_min,
            minStartTime=t_min,                      
            maxStartTime=t_max
        )

        try:
            search.run()
            
            if search.data is not None:
                min_val = search.data['twoF'].min()
                max_val = search.data['twoF'].max()
                print(f"Segment {t}: SFTs trobats. 2F varia entre {min_val:.7e} i {max_val:.7e}")
                
                twoF_values = search.data['twoF']
            
                # Guardem a la matriu B
                minim_len = min(len(twoF_values), num_freq_bins)
                B_matrix[:minim_len, t] = twoF_values[:minim_len]
            else:
                print(f"Segment {t}: Cap dada retornada (possible gap de dades).")

        except RuntimeError as e:
            # Catch the specific floating point / LALSuite errors caused by empty or zeroed data
            if "Floating point overflow" in str(e) or "Invalid argument" in str(e) or "empty multiSFT catalog" in str(e).lower():
                print(f"Segment {t}: LALSuite ha detectat un gap de dades o SFTs invàlids. S'ignora el segment.")
            else:
                # If it's a completely different RuntimeError, we still want to know about it
                raise

            del search
        gc.collect()

    print(f"Matriu B calculada amb èxit {B_matrix.shape}")

    np.save(f"B_matrix_real_data{df['Massa'].iloc[4]:.2e}eV.npy", B_matrix)
    np.save(f"params_hmm_real_data{df['Massa'].iloc[4]:.2e}eV.npy", {
        "f_cerca_min": f_cerca_min,
        "f_cerca_max": f_cerca_max,
        "deltaf": deltaf,
        "num_segments": num_segments,
        "T_obs": T_obs,
        "T_coh": T_coh,
    }, allow_pickle=True)

    return B_matrix

#######################################################################################################################
if __name__ == "__main__":
    """
    Script per generar dades i fer .sft per trobar la matriu B de l'HMM. 

    Després calculo el HMM per trobar la freqüència més probable a cada segment

    Finalment, calculo la freqüència final més probable per cada temps d'observació."""

    CSV_PATH = "data/resultats_simulacio.csv"

    if not os.path.exists(CSV_PATH):
        print(f"AVÍS: No s'ha trobat {CSV_PATH}. Generant les dades inicials amb SuperRad...")
    # Executa automàticament l'script de simulació
        subprocess.run(["python", "src/superrad_script.py"], check=True)
        print("Dades generades correctament. Continuant amb el pipeline...")

    ruta_earth = "/home/marc81/root_trial/envs/gw_fix/lib/python3.10/site-packages/solar_system_ephemerides/ephemerides/earth/earth00-40-DE405.dat.gz"
    ruta_sun = "/home/marc81/root_trial/envs/gw_fix/lib/python3.10/site-packages/solar_system_ephemerides/ephemerides/sun/sun00-40-DE405.dat.gz"

    df = pd.read_csv(CSV_PATH)
    alpha_gw = 3.37
    delta_gw = 0.45

    for index in range(len(df)):

        F0 = df["Freq_inicial"].iloc[index]
        F1 = df["Freqdot_inicial"].iloc[index]
        h0_ampl = df["Amplitud_h0"].iloc[index]
        tau_gw = df["Tau_gw"].iloc[index]

        params = calcula_parametres_hmm(
            F1,
            n_min=5,
            drift_bin_max=0.85
        )

        T_sft = params["T_sft"]
        T_coh = params["T_coh"]
        N_SFTs = params["N_SFTs"]

        print("CONFIGURACIÓ ADOPTADA:")
        print(params)
        print(f" Massa: {df['Massa'].iloc[index]:.2e} eV | F0: {df['Freq_inicial'].iloc[index]:.2f} Hz | F1: {df['Freqdot_inicial'].iloc[index]:.2e} Hz/s | h0: {h0_ampl:.2e}",
            f" | Tau_gw: {tau_gw:.2e} s")

        T_obs = tau_gw + 1800 # Afegeixo 30 minuts extra per assegurar-me de cobrir tot el senyal

        N_segments = int(T_obs / T_coh)      
        T_obs_real = N_segments * T_coh

        t_start = 1384788400 
        # Hz extra per Doppler i arrodoniments

        # Rang de freqüències que necessites cobrir
        f_cerca_min = F0 - 0.1
        f_cerca_max = F0 + 1.7 + abs(F1) * T_obs  # el senyal pot pujar fins aquí

        dist_baix = F0 - f_cerca_min          # distància cap avall
        dist_dalt  = f_cerca_max - F0         # distància cap amunt
        marge = 8.0                            # marge fix generós
        # Band per generar el soroll gaussià.
        Band_necessari = 2 * max(dist_baix, dist_dalt) + marge

        # Centre i amplada real necessària
        f_centre = (f_cerca_min + f_cerca_max) / 2.0

        print(f"Rang de freqüències a cobrir: {f_cerca_min:.2f} Hz - {f_cerca_max:.2f} Hz (Band = {Band_necessari:.2f} Hz)")
        # Generem el soroll gaussià i injectem el senyal del bosó superradiant amb els paràmetres 
        # calculats amb superrad

        writer = pyfstat.Writer(
            label="injectionboso1",
            outdir=f"data/processed/fake_data/dades_simulades_{df['Massa'].iloc[index]:.2e}eV",
            tstart=t_start,
            duration=T_obs_real,
            Tsft=T_sft,
            
            # --- Paràmetres del Soroll ---
            detectors="H1,L1",       # Simulem el detector Hanford i Livingston (LIGO)
            sqrtSX=1e-23,         # Nivell de soroll Gaussià (PSD).
            Band=Band_necessari,  
            
            # --- Paràmetres del Senyal Injectat (Ona de SuperRad) ---
            F0=F0,      # Freqüència inicial
            F1=F1,             # Evolució de la freqüència
            h0=h0_ampl,       # Amplitud del senyal
            cosi=1.0,             # Angle d'inclinació de la font (1.0 = òptim)
            Alpha=alpha_gw,            # Ascensió recta (RA) del forat negre
            Delta=delta_gw,             # Declinació (Dec) del forat negre
            earth_ephem=ruta_earth,
            sun_ephem=ruta_sun,

            randSeed=42 # Consistència i reproducibilitat de les dades simulades.
        )

        # 3. Generem els arxius SFT
        print("Dades SFT simulades i guardades a la carpeta 'dades_simulades")
        writer.make_data()

        # --- 1. DEFINICIÓ DE PARÀMETRES ---
        # Aquests valors són d'exemple per a la teva prova o per a GW231123           # Temps GPS d'inici (ex. GW231123 [5])
        t_coh = T_coh                   # T_coh definitiu sintonitzat (en segons)
        num_segments = int(np.round(T_obs / t_coh))  # N_T (T_obs total / T_coh)
                        # Freqüència màxima a buscar
        deltaf = 1.0 / (2.0 * t_coh)        # Resolució del bin segons la regla de l'LVK [6]

        # Coordenades del cel per a l'esdeveniment GW231123 (en radians) [7]
        alpha_gw = 3.37
        delta_gw = 0.45

        # --- 2. INICIALITZACIÓ DE LA MATRIU B ---
        # Calculem el nombre de files (bins de freqüència)
        num_freq_bins = int(np.round((f_cerca_max - f_cerca_min) / deltaf)) + 1

        # Creem la Matriu B buida: (estats_ocults, passos_de_temps)
        B_matrix = np.zeros((num_freq_bins, num_segments))

        # --- 3. BUCLE DE CÀLCUL PER A CADA SEGMENT (Columnes) ---
        for t in range(N_segments):
            
            # Marquem l'inici i el final del segment coherent actual
            t_min = t_start + t * t_coh
            t_max = t_min + t_coh
            
            # 4. CONFIGURACIÓ DE LA CERCA AMB PYFSTAT (Unidimensional)
            search = pyfstat.GridSearch(
                label=f"segment_{t}",
                outdir="output_B_matrix",
                sftfilepattern=f"data/processed/fake_data/dades_simulades_{df['Massa'].iloc[index]:.2e}eV/*.sft",  # Ruta on tens els SFTs simulats
                F0s=[f_cerca_min, f_cerca_max, deltaf],# L'eix Y (les freqüències que avaluem)
                F1s=[0.0],              # No busquem deriva de freqüència en aquesta prova
                F2s=[0.0], 
                Alphas=[alpha_gw],
                Deltas=[delta_gw],
                tref=t_min,
                minStartTime=t_min,                      # Límits d'aquest bloc curt
                maxStartTime=t_max
            )

            
            # Executem el càlcul matemàtic de l'F-statistic sobre les dades
            search.run()
            
            if search.data is not None:
                min_val = search.data['twoF'].min()
                max_val = search.data['twoF'].max()
                print(f"Segment {t}: SFTs trobats. 2F varia entre {min_val:.7e} i {max_val:.7e}")
                df_results = search.data['twoF']
            else:
                print(f"Segment {t}: ERROR - search.data està BUIT! (Cap SFT ha entrat a la finestra)")
            
            # Extraiem la columna amb l'estadística 2F
            twoF_values = search.data['twoF']
            
            # Ho guardem a la matriu B (comprobant longituds per seguretat)
            minim_len = min(len(twoF_values), num_freq_bins)
            B_matrix[:minim_len, t] = twoF_values[:minim_len]

            # Neteja de memòria forçada
            del search
            gc.collect()

        print("Matriu B calculada amb èxit! Dimensions:", B_matrix.shape)

        np.save(f"B_matrix_{df['Massa'].iloc[index]:.2e}eV.npy", B_matrix)
        np.save(f"params_hmm_{df['Massa'].iloc[index]:.2e}eV.npy", {
            "f_cerca_min": f_cerca_min,
            "f_cerca_max": f_cerca_max,
            "deltaf": deltaf,
            "num_segments": num_segments,
            "T_obs": T_obs,
            "T_coh": T_coh,
        }, allow_pickle=True)

        print("Generant la visualització...")

        plt.figure(figsize=(12, 6))

        # Utilitzem imshow per representar la matriu com a mapa de calor.
        # - aspect='auto' permet que la graella s'ajusti a la finestra.
        # - origin='lower' fa que l'eix Y (freqüència) creixi cap amunt, com és lògic.
        # - extent mapeja els píxels als valors reals de temps (hores) i freqüència (Hz).
        imatge = plt.imshow(B_matrix, aspect='auto', origin='lower', 
                            extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max],
                            cmap='viridis') # 'viridis' és el mapa de colors estàndard en ciència

        # Afegim la barra de color per saber quanta energia (2F) hi ha
        cbar = plt.colorbar(imatge)
        plt.colorbar(im).set_label(r'$2\mathcal{F}$-statistic', fontsize=12)
        plt.xlabel('Observation time (h)', fontsize=12)
        plt.ylabel('Frequency $f$ (Hz)', fontsize=12)

        plt.title('$\mathcal{B}$ Matrix', fontsize=14)

        plt.tight_layout()

        # Guardem la imatge i la mostrem
        plt.savefig(f'results/simulations/B_matrix/matriu_B_heatmap_{df["Massa"].iloc[index]:.2e}eV.png', dpi=300)
        print("Imatge guardada amb èxit! Obre 'matriu_B_heatmap.png' des del teu Windows.")
        plt.show()

        # --- EXECUCIÓ ---
        freqs_viterbi, bins_viterbi, matriu_acumulada = algoritme_viterbi(B_matrix, f_cerca_min, deltaf)

        print(f"Camí de Viterbi calculat amb èxit! Freqüència final detectada: {freqs_viterbi[-1]:.4f} Hz")


        # Assegura't de tenir aquestes variables del bloc anterior:
        # B_matrix (les dades), V (matriu acumulada), freqs_viterbi (el camí), f_cerca_min, f_cerca_max, T_obs, num_segments

        temps_hores = np.linspace(0, T_obs / 3600, N_segments)

        # Creem una figura amb 2 panells (un a sobre de l'altre)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # ==========================================
        # PANELL 1: La Matriu B (Dades Crues)
        # ==========================================
        imatge_B = ax1.imshow(B_matrix, aspect='auto', origin='lower', 
                            extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max], cmap='viridis')
        # Superposem el camí per veure com d'invisible era
        # ax1.plot(temps_hores, freqs_viterbi, color='red', linewidth=2, label="Camí Viterbi")

        ax1.set_ylabel('Frequency $f$ (Hz)', fontsize=12)
        ax1.set_title('$\mathcal{B}$ Matrix)', fontsize=14)
        # ax1.legend(loc="upper right")
        cbar1 = fig.colorbar(imatge_B, ax=ax1)
        cbar1.set_label('$2\mathcal{F}$', fontsize=12)

        # ==========================================
        # PANELL 2: La Matriu V (Viterbi Acumulat)
        # ==========================================
        # Normalitzem la matriu V per columna perquè es vegi bé el contrast visualment
        # Això evita que els valors del final (que són la suma de tots) eclipsin els del principi
        V_normalitzada = np.zeros_like(matriu_acumulada)

        for t in range(num_segments):
            V_normalitzada[:, t] = matriu_acumulada[:, t] - np.max(matriu_acumulada[:, t])

        imatge_V = ax2.imshow(V_normalitzada, aspect='auto', origin='lower', 
                            extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max], cmap='plasma') # Usem un altre color (plasma)
        # Superposem el camí per comprovar que segueix la cresta
        # ax2.plot(temps_hores, freqs_viterbi, color='cyan', linewidth=2, linestyle='--', label="Camí Viterbi (Cresta)")

        ax2.set_xlabel('Observation time (h)', fontsize=12)
        ax2.set_ylabel('Frequency (Hz)', fontsize=12)
        ax2.set_title(r'Viterbi $\mathcal{V}$ matrix', fontsize=14)
        ax2.legend(loc='upper right')
        fig.colorbar(imatge_V, ax=ax2).set_label('Cumulative log-likelihood', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'results/simulations/V_n_B_matrices/comprensio_viterbi_{df["Massa"].iloc[index]:.2e}eV.png', dpi=300)
        print(f"Visualització analítica guardada a 'comprensio_viterbi_{df['Massa'].iloc[index]:.2e}eV.png'")

        plt.figure(figsize=(12, 6))
        im = plt.imshow(V_normalitzada, aspect='auto', origin='lower',
                        extent=[0, T_obs / 3600, f_cerca_min, f_cerca_max], cmap='plasma')
        plt.plot(temps_hores, freqs_viterbi, color='cyan', lw=2, ls='--', label='Viterbi Path')
        plt.set_xlabel('Observation time (h)', fontsize=12)
        plt.set_ylabel('Frequency (Hz)', fontsize=12)
        plt.set_title(r'Viterbi $\mathcal{V}$ matrix', fontsize=14)
        plt.legend(loc='upper right')
        plt.colorbar(im, ax=ax2).set_label('Cumulative log-likelihood', fontsize=12)

        plt.tight_layout()
        nom = f"results/simulations/V_matrix/viterbi_{F0:.2f}Hz.png"
        plt.savefig(nom, dpi=300)
        print(f"Guardat: {nom}  |  Freqüència final: {freqs_viterbi[-1]:.4f} Hz")
        plt.close()