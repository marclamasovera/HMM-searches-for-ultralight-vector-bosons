from superrad import ultralight_boson as ub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" Script per generar dades simulades d'una senyal de bosó ultralleuger

Serveix per extreure freqüències, derivades de la freqüència i ergo temps de coherència 
per separar els bins de la freq per després mirar la sensibilitat de les cerques a diferents masses de bosó.

Com he mencionat abans, busquem diverses masses i si volem podem fer un plot per veure com
canvien els paràmetres del senyal (freq, f-dot, T_coh) en funció de la massa del bosó,
i així trobar una massa òptima."""

bc = ub.UltralightBoson(spin=1,model='relativistic') # Define our ultralight boson model

M_bh = 222.0
abh = 0.84
d_obs = 2200.0 # (en Mpc)
mu_boson = 1e-13

taus = []
fdots = []
cohs = []
iters_prop = []
F0s = []
F1s = []
h0_ampls = []
optimal_mass = 1.652e-13
masses  = np.linspace(optimal_mass*0.85,optimal_mass*1.025,6).tolist()
masses.append(optimal_mass)

for mu_boson in masses:

    print(f"Simulant per massa de bosó: {mu_boson:.2e} eV")

    wf = bc.make_waveform(Mbh=M_bh,abh=abh,mu=mu_boson,units="physical",evo_type='full')
    t_start = -wf.cloud_growth_time()
    tau_gw = wf.gw_time()
    t_end = 2 * wf.gw_time()
    t_array = np.linspace(t_start, t_end, 2000)


    h_plus, h_cross, delta = wf.strain_amp(t_array, thetaObs=np.pi/4, dObs=d_obs) # important:
                                                                                  # el d_obs!!
    freq_gw = wf.freq_gw(t_array)

    taus.append(tau_gw)

    print(f"Durada estimada del senyal (tau_gw): {tau_gw} segons (aprox {tau_gw/(3600):.2e} hores)")
    f_dot = np.gradient(freq_gw, t_array)
    f_dot_max = np.max(f_dot)
    fdots.append(f_dot_max)
    print(f"Deriva màxima de freqüència (f-dot): {f_dot_max:.2e} Hz/s")
    T_coh = 1/((2*f_dot_max)**(1/2))

    iters_prop.append(tau_gw/T_coh)

    cohs.append(T_coh)
    print(f"Temps de coherència recomanat (T_coh): {T_coh:.2f} segons")

    F0 = float(np.squeeze(wf.freq_gw(0.0)))
    F1 = float(np.squeeze(wf.freqdot_gw(0.0)))
    h0_ampl = float(np.max(np.abs(h_plus)))

    print(h0_ampl)
    F0s.append(F0)
    F1s.append(F1)
    h0_ampls.append(h0_ampl)

    print("#######################################################")

df = pd.DataFrame({
    "Massa" : masses,
    "Tau_gw" : taus,
    "f_dot_max" : fdots,
    "T_coh" : cohs,
    "Iters_propagació" : iters_prop,
    "Freq_inicial" : F0s,
    "Freqdot_inicial" : F1s,
    "Amplitud_h0" : h0_ampls

})

df.to_csv("data/resultats_simulacio.csv", index=False)

print("Simulació completada. Resultats guardats a 'resultats_simulacio.csv'")