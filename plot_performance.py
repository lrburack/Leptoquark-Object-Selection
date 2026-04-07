import matplotlib.pyplot as plt
import pickle
import os
from config import MODELS_DIRECTORY, MODEL_FILENAME, MASSPOINTS, MODEL_PERFORMANCE, PERFORMANCE_PLOT_PATH
import numpy as np

model_names = ["NN_4mu_5jet", "NN_3mu_4jet", "NN_2mu_3jet", "pt"]
# Name compare_model1_model2_
figname = os.path.join(PERFORMANCE_PLOT_PATH, "compare")
for model_name in model_names: figname += f"_{model_name}"

fig, ax = plt.subplots()
fig2, axs = plt.subplots(3, 1, sharex=True)
for model_name in model_names:
    masspoint = np.array(MASSPOINTS)
    # Store selection rates for all objects, lq muon, creation muon, lq jet 
    selection_rate = np.empty((4, len(masspoint)))

    # List the mass ranges in this directory and then loop over them
    mass_ranges = [d for d in os.listdir(os.path.join(MODELS_DIRECTORY, model_name)) if os.path.isdir(os.path.join(MODELS_DIRECTORY, model_name, d))]
    for mass_range in mass_ranges:
        with open(os.path.join(MODELS_DIRECTORY, model_name, mass_range, MODEL_PERFORMANCE), "rb") as f:
            performance = pickle.load(f)
        
        mass_low, mass_high = map(int, mass_range.split("M")[1].split("-"))
        selection_rate[0, (masspoint >= mass_low) & (masspoint <= mass_high)] = performance["selection_rate"]
        selection_rate[1, (masspoint >= mass_low) & (masspoint <= mass_high)] = performance["muon_rate"]
        selection_rate[2, (masspoint >= mass_low) & (masspoint <= mass_high)] = performance["creation_muon_rate"]
        selection_rate[3, (masspoint >= mass_low) & (masspoint <= mass_high)] = performance["jet_rate"]
    
    ax.plot(masspoint, selection_rate[0], label=model_name, marker="o")
    axs[0].plot(masspoint, selection_rate[1], label=model_name, marker="o")
    axs[1].plot(masspoint, selection_rate[2], label=model_name, marker="o")
    axs[2].plot(masspoint, selection_rate[3], label=model_name, marker="o")
ax.legend()
ax.set_xlabel("LQ Mass (GeV)")
ax.set_ylabel("All three objects")
fig.suptitle("Selection rate comparison")

fig.savefig(f"{figname}_all.png")

axs[0].legend()
axs[-1].set_xlabel("LQ Mass (GeV)")
axs[0].set_ylabel("LQ Muon")
axs[1].set_ylabel("Creation Muon")
axs[2].set_ylabel("LQ Jet")
fig2.suptitle("Selection rate comparison")
fig2.subplots_adjust(hspace=0)

fig2.savefig(f"{figname}_each.png")