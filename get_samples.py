import os

# This is for me because I'm doing the analysis locally.

outpath = "samples"
os.makedirs(outpath, exist_ok=True)
samples_path = "/eos/cms/store/group/phys_susy/LQ2/LQ_2017merged/"

os.system(f"scp -r lburack@lxplus.cern.ch:{samples_path} {outpath}")


# samples = ["Signal_LQToBMu_M_2500_single.root",
#     "Signal_LQToBMu_M_1000_single_coupling0p1.root",  "Signal_LQToBMu_M_2600_single.root",
#     "Signal_LQToBMu_M_1000_single_coupling0p3.root",  "Signal_LQToBMu_M_2700_single.root",
#     "Signal_LQToBMu_M_1000_single_coupling0p5.root",  "Signal_LQToBMu_M_2800_single.root",
#     "Signal_LQToBMu_M_1000_single.root",              "Signal_LQToBMu_M_2900_single.root",
#     "Signal_LQToBMu_M_1100_single.root",              "Signal_LQToBMu_M_3000_single.root",
#     "Signal_LQToBMu_M_1200_single.root",              "Signal_LQToBMu_M_300_single_coupling0p1.root",
#     "Signal_LQToBMu_M_1300_single.root",              "Signal_LQToBMu_M_300_single_coupling0p3.root",
#     "Signal_LQToBMu_M_1400_single.root",              "Signal_LQToBMu_M_300_single_coupling0p5.root",
#     "Signal_LQToBMu_M_1500_single.root",              "Signal_LQToBMu_M_300_single.root",
#     "Signal_LQToBMu_M_1600_single_coupling0p1.root",  "Signal_LQToBMu_M_3500_single.root",
#     "Signal_LQToBMu_M_1600_single_coupling0p3.root",  "Signal_LQToBMu_M_4000_single.root",
#     "Signal_LQToBMu_M_1600_single_coupling0p5.root",  "Signal_LQToBMu_M_400_single.root",
#     "Signal_LQToBMu_M_1600_single.root",              "Signal_LQToBMu_M_500_single_coupling0p1.root",
#     "Signal_LQToBMu_M_1700_single.root",              "Signal_LQToBMu_M_500_single_coupling0p3.root",
#     "Signal_LQToBMu_M_1800_single.root",              "Signal_LQToBMu_M_500_single_coupling0p5.root",
#     "Signal_LQToBMu_M_1900_single.root",              "Signal_LQToBMu_M_500_single.root",
#     "Signal_LQToBMu_M_2000_single.root",              "Signal_LQToBMu_M_600_single.root",
#     "Signal_LQToBMu_M_2100_single.root",              "Signal_LQToBMu_M_700_single.root",
#     "Signal_LQToBMu_M_2200_single.root",              "Signal_LQToBMu_M_800_single.root",
#     "Signal_LQToBMu_M_2300_single.root",              "Signal_LQToBMu_M_900_single.root",
#     "Signal_LQToBMu_M_2400_single.root",
# ]

# for sample in samples:
#     if not os.path.exists(os.path.join(outpath, sample)):
#         command = f"scp -r lburack@lxplus.cern.ch:{samples_path} {outpath}"
#         os.system(command)