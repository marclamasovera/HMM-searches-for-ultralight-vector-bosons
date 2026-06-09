Final project for the Instrumentation, Data Analysis and Machine Learning course of the Astrophysics, Particle Physics and Cosmology Master.

THIS IS NOT INTENDED TO BE A FULL RIGOUROUS BOSON SEARCH. The results obtained in the real data pipeline are the result of a demonstration on how to use an HMM method 

All the code is self-consistent. 

### Superrad Script:
Uses the python package `SuperRad` in order to simulate superradiant ultralight boson CW waveforms and extract waveform properties to then study this extensions of the Standard Model of particles

### Simul.py
Uses the simulated properties from the previous script and injects the signal into gaussian simulated noise. Then tries to recover the signal using the Viterbi Algorithm

### hmm_pipeline_real_data.py
Fetches and downloads 5 hours of real strain data from the L1 and H1 detectors for the GW231123 event, then computes for every boson mass a SFT according to the required algorithm proposed by the LVK collaboration
(`calcula_parametres_hmm`).

Then computes 100 injections into the real data for every boson mass (so for every frequency band searched) and sees if the Viterbi Algortihm recovers the signal, based on a Loudest Event criteria.

Finally, a path matching injection campaign is taken in place, finding that no boson mass can be given as a upper limit
