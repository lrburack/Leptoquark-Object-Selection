import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import awkward as ak
import re
from helpers import dr, invariant_mass
from config import ANALYSIS_PLOT_PATH, PROCESSED_SAMPLES_PATH

# sample file: processed_Signal_LQToBMu_M_500_single.root
# selected objects: ["gen_lq_muon", "gen_creation_muon", "gen_lq_quark", "reco_lq_muon", "reco_creation_muon", "reco_lq_jet"]

file = sys.argv[1]
mass = int(re.search(r"_M_(\d+)_", file).group(1))

signal_path = PROCESSED_SAMPLES_PATH
outpath = os.path.join(ANALYSIS_PLOT_PATH, f"M{mass}")
os.makedirs(outpath, exist_ok=True)

with open(os.path.join(signal_path, file), "rb") as f:
    processed = pickle.load(f)

nevents = len(processed["GenPart_pt"])
allevents = np.arange(nevents)

styles = {
    'gen': {"color": "blue"},
    'reco': {"color": "blue", "linestyle": "--"},
    'alt gen': {"color": "red"},
    'alt reco': {"color": "red", "linestyle": "--"}
}

def getobj(b, selb):
    return processed[b][allevents, processed[selb]]

# Create a histogram showing the pt rank of the muon in each event
figname = "muonrank.png"
fig, ax1 = plt.subplots(1, figsize=(4.5,3))
fig.suptitle(f"{mass}GeV Leptoquark")
bins = np.arange(5) + 0.5

final_state_muons = (np.abs(processed["GenPart_pdgId"]) == 13) & (processed["GenPart_status"] == 1)
gen_lq_muon_rank = ak.sum(processed["GenPart_pt"][final_state_muons] > getobj("GenPart_pt", "gen_lq_muon"), axis=1) + 1
gen_creation_muon_rank = ak.sum(processed["GenPart_pt"][final_state_muons] > getobj("GenPart_pt", "gen_creation_muon"), axis=1) + 1
ax1.hist(gen_lq_muon_rank, bins=bins, label="LQ-muon (Gen)", histtype='step', density=True, **styles["gen"])
ax1.hist(gen_creation_muon_rank, bins=bins, label="IS-muon (Gen)", histtype='step', density=True, **styles["alt gen"])
ax1.set_ylabel("Fraction of events")
ax1.set_xlabel(r"$p_T$ rank")
ax1.set_xticks(np.array([0, 1, 2, 3, 4, 5]))
ax1.set_ylim([0,1])

candidate_muons = processed["Muon_pt"]
reco_lq_muon_rank = ak.sum(candidate_muons > getobj("Muon_pt", "reco_lq_muon"), axis=1) + 1
reco_creation_muon_rank = ak.sum(candidate_muons > getobj("Muon_pt", "reco_creation_muon"), axis=1) + 1
ax1.hist(reco_lq_muon_rank, bins=bins, label="LQ-muon (Reco)", histtype='step', density=True, **styles["reco"])
ax1.hist(reco_creation_muon_rank, bins=bins, label="IS-muon (Reco)", histtype='step', density=True, **styles["alt reco"])
ax1.legend()

fig.tight_layout()
plt.savefig(os.path.join(outpath, figname))

LQ_id = 9000007

figname = "jetrank.png"
fig, ax1 = plt.subplots(1, figsize=(4.5,3))
fig.suptitle(f"{mass}GeV Leptoquark")
candidate_quarks = processed["GenPart_pt"][np.abs(processed["GenPart_pdgId"]) <= 8]
gen_lq_quark_rank = ak.sum(candidate_quarks > getobj("GenPart_pt", "gen_lq_quark"), axis=1) + 1
ax1.hist(gen_lq_quark_rank, bins=bins, label="LQ-jet (Gen)", histtype='step', density=True, **styles["gen"])

candidate_jets = processed["Jet_pt"][processed["Jet_jetId"] >= 6]
reco_lq_jet_rank = ak.sum(candidate_jets > getobj("Jet_pt", "reco_lq_jet"), axis=1) + 1
ax1.hist(reco_lq_jet_rank, bins=bins, label="LQ-jet (Reco)", histtype='step', density=True, **styles["reco"])
# ax1.set_ylim([0,1])
ax1.set_xticks(np.array([0, 1, 2, 3, 4, 5]))
ax1.set_ylabel("Fraction of events")
ax1.set_xlabel(r"$p_T$ rank")
ax1.legend()
ax1.set_ylim([0,1])

fig.tight_layout()
plt.savefig(os.path.join(outpath, figname))

figname = "check_recojet_matching.png"
fig, axs = plt.subplots(1,3)
axs[0].set_title("eta")
axs[1].set_title("phi")
axs[2].set_title("pt")
axs[0].hist(np.clip(getobj("GenPart_eta", "gen_lq_quark") - getobj("Jet_eta", "reco_lq_jet"), -5, 5), histtype="step", label="gen lq quark", **styles["gen"])
axs[1].hist(np.clip(getobj("GenPart_phi", "gen_lq_quark") - getobj("Jet_phi", "reco_lq_jet"), -5, 5), histtype="step", label="gen lq quark", **styles["gen"])
axs[2].hist(np.clip(getobj("GenPart_pt", "gen_lq_quark") - getobj("Jet_pt", "reco_lq_jet"), -30, 100), histtype="step", label="gen lq quark", **styles["gen"])
plt.savefig(os.path.join(outpath, figname))

figname = "ptselection.png"
fig, [ax1, ax2] = plt.subplots(1, 2)
fig.suptitle(figname)

cm = np.array([
    [np.sum((gen_lq_muon_rank == 1) & (gen_lq_quark_rank == 1)), np.sum(~(gen_lq_muon_rank == 1) & (gen_lq_quark_rank == 1))],
    [np.sum((gen_lq_muon_rank == 1) & ~(gen_lq_quark_rank == 1)), np.sum(~(gen_lq_muon_rank == 1) & ~(gen_lq_quark_rank == 1))],
], dtype=int) / nevents
# optional: normalized version (percentages)
im = ax1.imshow(cm, origin="upper")  # or use cm_norm for percentages
for (i, j), v in np.ndenumerate(cm):
    ax1.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=11,color="white")
ax1.set_xticks([0, 1], ["correct", "incorrect"])
ax1.set_yticks([0, 1], ["correct", "incorrect"], rotation=90)
ax1.set_xlabel("Muon selection")
ax1.set_ylabel("Jet selection")
ax1.set_title("Gen information")

cm = np.array([
    [np.sum((reco_lq_muon_rank == 1) & (reco_lq_jet_rank == 1)), np.sum(~(reco_lq_muon_rank == 1) & (reco_lq_jet_rank == 1))],
    [np.sum((reco_lq_muon_rank == 1) & ~(reco_lq_jet_rank == 1)), np.sum(~(reco_lq_muon_rank == 1) & ~(reco_lq_jet_rank == 1))],
], dtype=int) / nevents
# optional: normalized version (percentages)
im = ax2.imshow(cm, origin="upper")  # or use cm_norm for percentages
for (i, j), v in np.ndenumerate(cm):
    ax2.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=11,color="white")
ax2.set_xticks([0, 1], ["correct", "incorrect"])
ax2.set_yticks([0, 1], ["correct", "incorrect"], rotation=90)
ax2.set_xlabel("Muon selection")
ax2.set_ylabel("Jet selection")
ax2.set_title("Reco information")
fig.tight_layout()
plt.savefig(os.path.join(outpath, figname))

figname = f"muonetadistribution.png"
fig, ax = plt.subplots(1)
bins = np.linspace(-3, 3, 21)
ax.hist(getobj("GenPart_eta", "gen_lq_muon"), bins=bins, histtype="step", label="LQ Muon", **styles["gen"])
ax.hist(getobj("GenPart_eta", "gen_creation_muon"), bins=bins, histtype="step", label="Creation Muon", **styles["alt gen"])
ax.set_xlabel("eta")
ax.set_ylabel("Number of events")
ax.legend()
plt.savefig(os.path.join(outpath, figname))

figname = f"drdistribution.png"
fig, ax = plt.subplots(1)
bins = np.linspace(0, 6, 16)
quark_lq_muon_dr = dr(getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"),
                      getobj("GenPart_eta", "gen_lq_muon"), getobj("GenPart_phi", "gen_lq_muon"))
ax.hist(quark_lq_muon_dr, histtype="step", label="Gen LQ Muon", **styles["gen"], bins=bins)
jet_lq_muon_dr = dr(getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet"),
                      getobj("Muon_eta", "reco_lq_muon"), getobj("Muon_phi", "reco_lq_muon"))
ax.hist(jet_lq_muon_dr, histtype="step", label="Reco LQ Muon", **styles["reco"], bins=bins)
quark_creation_muon_dr = dr(getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"),
                      getobj("GenPart_eta", "gen_creation_muon"), getobj("GenPart_phi", "gen_creation_muon"))
ax.hist(quark_creation_muon_dr, histtype="step", label="Creation Muon", **styles["alt gen"], bins=bins)
jet_creation_muon_dr = dr(getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet"),
                      getobj("Muon_eta", "reco_creation_muon"), getobj("Muon_phi", "reco_creation_muon"))
ax.hist(jet_creation_muon_dr, histtype="step", label="Reco Creation Muon", **styles["alt reco"], bins=bins)
ax.set_xlabel("dr to true LQ quark")
ax.set_ylabel("Number of events")
ax.legend()
plt.savefig(os.path.join(outpath, figname))

figname = f"pt.png"
fig, ax = plt.subplots(1)
ax.hist(getobj("GenPart_pt", "gen_lq_muon"), histtype="step", label="Gen LQ Muon", **styles["gen"])
ax.hist(getobj("Muon_pt", "reco_lq_muon"), histtype="step", label="Reco LQ Muon", **styles["reco"])
ax.hist(getobj("GenPart_pt", "gen_creation_muon"), histtype="step", label="Gen Creation Muon", **styles["alt gen"])
ax.hist(getobj("Muon_pt", "reco_creation_muon"), histtype="step", label="Reco Creation Muon", **styles["alt reco"])
ax.set_xlabel(r"$p_T$ (GeV)")
ax.set_ylabel("Number of events")
ax.legend()
plt.savefig(os.path.join(outpath, figname))

gen_inv_mass = invariant_mass(
    getobj("GenPart_pt", "gen_lq_muon"), getobj("GenPart_eta", "gen_lq_muon"), getobj("GenPart_phi", "gen_lq_muon"),
    getobj("GenPart_pt", "gen_lq_quark"), getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"), m2=0.10566
)
reco_inv_mass = invariant_mass(
    getobj("Muon_pt", "reco_lq_muon"), getobj("Muon_eta", "reco_lq_muon"), getobj("Muon_phi", "reco_lq_muon"),
    getobj("Jet_pt", "reco_lq_jet"), getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet"), m2=0.10566
)
figname = f"invariantmass.png"
fig, [ax1, ax2] = plt.subplots(1,2)
ax1.hist(gen_inv_mass, histtype="step", label="Gen LQ Muon", **styles["gen"])
ax1.hist(reco_inv_mass, histtype="step", label="Reco LQ Muon", **styles["reco"])
ax2.hist(gen_inv_mass - reco_inv_mass, histtype="step")
ax1.set_xlabel("Invariant Mass")
ax2.set_xlabel("Reco error")
ax1.set_ylabel("Number of events")
ax1.legend()
plt.savefig(os.path.join(outpath, figname))

figname = f"recommended_dr_cut.png"
fig, ax = plt.subplots(1)
gen_quark_lq_muon_dr = dr(getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"),
                      getobj("GenPart_eta", "gen_lq_muon"), getobj("GenPart_phi", "gen_lq_muon"))
reco_jet_lq_muon_dr = dr(getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet"),
                      getobj("Muon_eta", "reco_lq_muon"), getobj("Muon_phi", "reco_lq_muon"))
cuts = np.linspace(0, 6, 100)
gen_eff = np.array([np.sum(gen_quark_lq_muon_dr > c) for c in cuts]) / nevents
reco_eff = np.array([np.sum(reco_jet_lq_muon_dr > c) for c in cuts]) / nevents
ax.plot(cuts, gen_eff, label="Gen LQ muon-jet DR", **styles["gen"])
ax.plot(cuts, reco_eff, label="Reco LQ muon-jet DR", **styles["reco"])
reco_percentile_cut_ind = np.where(reco_eff < 0.99)[0][0]
ax.axvline(cuts[reco_percentile_cut_ind], color="black", linestyle="--", label=f"99% eff cut: {cuts[reco_percentile_cut_ind]:.2f}")
ax.axhline(reco_eff[reco_percentile_cut_ind], color="black", linestyle="--")

gen_quark_creation_muon_dr = dr(getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"),
                      getobj("GenPart_eta", "gen_creation_muon"), getobj("GenPart_phi", "gen_creation_muon"))
reco_jet_creation_muon_dr = dr(getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet"),
                      getobj("Muon_eta", "reco_creation_muon"), getobj("Muon_phi", "reco_creation_muon"))
gen_eff = np.array([np.sum(gen_quark_creation_muon_dr > c) for c in cuts]) / nevents
reco_eff = np.array([np.sum(reco_jet_creation_muon_dr > c) for c in cuts]) / nevents
ax.plot(cuts, gen_eff, label="Gen creation muon-jet DR", **styles["alt gen"])
ax.plot(cuts, reco_eff, label="Reco creation muon-jet DR", **styles["alt reco"])
ax.set_xlabel("dr cut")
ax.set_ylabel("Efficiency")
ax.set_title(figname)
ax.legend()
plt.savefig(os.path.join(outpath, figname))

# Plot for invariant mass rankings
figname = "invariant_mass_closeness.png"
recoranks = np.zeros(nevents)
reco_inv_mass = invariant_mass(
    getobj("Muon_pt", "reco_lq_muon"), getobj("Muon_eta", "reco_lq_muon"), getobj("Muon_phi", "reco_lq_muon"),
    getobj("Jet_pt", "reco_lq_jet"), getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet")
)
print("waiting...")
for i in range(1000):
    for mu in range(len(processed["Muon_pt"][i])):
        for jet in range(len(processed["Jet_pt"][i])):
            if mu == processed["reco_lq_muon"][i] and jet == processed["reco_lq_jet"][i]:
                continue
            inv_mass = invariant_mass(
                processed["Muon_pt"][i][mu], processed["Muon_eta"][i][mu], processed["Muon_phi"][i][mu],
                processed["Jet_pt"][i][jet], processed["Jet_eta"][i][jet], processed["Jet_phi"][i][jet]
            )
            if inv_mass > reco_inv_mass[i]:
                recoranks[i] += 1

genranks = np.zeros(nevents)
gen_inv_mass = invariant_mass(
    getobj("GenPart_pt", "gen_lq_muon"), getobj("GenPart_eta", "gen_lq_muon"), getobj("GenPart_phi", "gen_lq_muon"),
    getobj("GenPart_pt", "gen_lq_quark"), getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"), m2=0.10566
)
print("waiting...")
for i in range(1000):
    for mu in range(len(processed["GenPart_pt"][i])):
        for jet in range(len(processed["GenPart_pt"][i])):
            if mu == processed["gen_lq_muon"][i] and jet == processed["gen_lq_quark"][i]:
                continue
            inv_mass = invariant_mass(
                processed["Muon_pt"][i][mu], processed["Muon_eta"][i][mu], processed["Muon_phi"][i][mu],
                processed["Jet_pt"][i][jet], processed["Jet_eta"][i][jet], processed["Jet_phi"][i][jet]
            )
            if inv_mass > reco_inv_mass[i]:
                genranks[i] += 1

fig, ax = plt.subplots(1)
bins = np.arange(5) + 0.5
counts, _ = np.histogram(recoranks, bins)        
ax.stairs(counts / np.sum(counts), bins)
counts, _ = np.histogram(genranks, bins)        
ax.stairs(counts / np.sum(counts), bins)
fig.savefig(os.path.join(outpath, figname))

# Save the recommended cut for this mass point
# path = os.path.join(outpath, "recommended_dr_cuts.pkl")
# if not os.path.exists(path):
#     with open(path, "wb") as f:
#         pickle.dump({}, f)
# with open(path, "rb") as f:
#     dr_allmass = pickle.load(f)
# dr_allmass[mass] = reco_jet_lq_muon_dr
# with open(path, "wb") as f:
#     pickle.dump(dr_allmass, f)


# Fraction signal muon below 30 GeV.
# low_lq_muon = np.sum(getobj("Muon_pt", "reco_lq_muon") < 30) / nevents
# low_creation_muon = np.sum(getobj("Muon_pt", "reco_creation_muon") < 30) / nevents
# print("Low lq muon fraction:", low_lq_muon)
# print("Low creation muon fraction:", low_creation_muon)
# if not "low_lq_muon" in multiple_masspoint_data:
#     multiple_masspoint_data["low_lq_muon"] = {}
# multiple_masspoint_data["low_lq_muon"][mass] = low_lq_muon
# if not "low_creation_muon" in multiple_masspoint_data:
#     multiple_masspoint_data["low_creation_muon"] = {}
# multiple_masspoint_data["low_creation_muon"][mass] = low_creation_muon

# with open(multiple_masspoint_data_path, "wb") as f:
#     pickle.dump(multiple_masspoint_data, f)

# Also find the closest muon to the lq jet
# doevents = 1000
# closest_jet_dr = np.zeros(doevents)
# pt_cut = 50
# for ev in range(doevents):
#     jet = processed["reco_lq_jet"][ev]
#     muons = np.where(processed["Muon_pt"][ev] > pt_cut)[0]
#     dists = dr(processed["Muon_eta"][ev][muons], processed["Muon_phi"][ev][muons], processed["Jet_eta"][ev][jet], processed["Jet_phi"][ev][jet])
#     print(processed["Muon_eta"][ev])
#     print(processed["Muon_phi"][ev])
#     print(processed["Jet_eta"][ev][jet], processed["Jet_phi"][ev][jet])
#     print(dists)
#     print(processed["Jet_jetId"][ev][jet])
#     print()
#     closest_jet_dr[ev] = np.min(dists)

# print(np.sum(closest_jet_dr == 0))

# figname = f"closest_muon_ptcut={pt_cut}_{file[:-5]}.png"
# fig, ax = plt.subplots(1)
# cuts = np.linspace(0, 1, 100)
# reco_eff = np.array([np.sum(closest_jet_dr > c) for c in cuts]) / doevents
# ax.plot(cuts, reco_eff, label=f"Closest muon (pt>{pt_cut}GeV) to the LQ jet", **styles["reco"])
# ax.set_xlabel("dr cut")
# ax.set_ylabel("Efficiency")
# ax.legend()
# plt.savefig(os.path.join(outpath, figname))

# # Save the recommended cut for this mass point
# path = os.path.join(outpath, "closest_jet_.pkl")
# if not os.path.exists(path):
#     with open(path, "wb") as f:
#         pickle.dump({}, f)
# with open(path, "rb") as f:
#     closest_jet = pickle.load(f)
# closest_jet[mass] = closest_jet_dr
# with open(path, "wb") as f:
#     pickle.dump(closest_jet, f)

# Try a simple bayesian approach where you take every possible muon/quark pair and do the following:
# Compute the probability that the muon is the true lq muon given its pt relative to the highest pt muon.
# Then compute the probability that the quark is the true lq quark given its pt and dr from the muon.
# Could even allow the probability calculation to depend on the LQ mass by calculating what the LQ mass would be for each pair
# and then using different calculation depending on this invariant mass.

# print(invariant_mass(
#     getobj("GenPart_pt", "gen_lq_muon"), getobj("GenPart_eta", "gen_lq_muon"), getobj("GenPart_phi", "gen_lq_muon"),
#     getobj("GenPart_pt", "gen_lq_quark"), getobj("GenPart_eta", "gen_lq_quark"), getobj("GenPart_phi", "gen_lq_quark"), m2=0.10566
# ))

# This approach could in theory include any kinematic information as part of the probability calculation,
# but I think for now I will just include pt and dr.
# This approach could also include any order correlation, but for now I will treat pt and dr as separate
# ptbins = np.linspace(0, 1, 20)
# lq_muon_pt_binprob, _ = np.histogram(getobj("GenPart_pt", "gen_lq_muon") / maxmuonpt, bins=ptbins)
# lq_muon_pt_binprob = lq_muon_pt_binprob / nevents
#
# drbins = np.linspace(0, 6, 20)
# lq_muon_dr_binprob, _ = np.histogram(quark_lq_muon_dr, bins=drbins)
# lq_muon_dr_binprob = lq_muon_dr_binprob / nevents
#
# maxjetpt = np.max(processed["GenPart_pt"][np.abs(processed["GenPart_pdgId"]) <= 8], axis=1)
# jetptbins = np.linspace(0, 1, 20)
# lq_jet_pt_binprob, _ = np.histogram(getobj("GenPart_pt", "gen_lq_quark") / maxjetpt, bins=jetptbins)
# lq_jet_pt_binprob = lq_muon_dr_binprob / nevents
#
# nmuon_cand = 2
# muon_candidates = ak.argsort(
#     ak.mask(processed["GenPart_pt"], final_state_muons),
#     axis=-1, ascending=False
# )[:, :nmuon_cand]
# njet_cand = 5
# jet_candidates = ak.argsort(
#     ak.mask(processed["GenPart_pt"], np.abs(processed["GenPart_pdgId"]) <= 8),
#     axis=-1, ascending=False
# )[:, :njet_cand]
#
# probabilities = np.zeros((nevents, nmuon_cand, njet_cand))
#
# for ev in range(nevents):
#     for imu, mu in enumerate(muon_candidates[ev]):
#         muprob = lq_muon_pt_binprob[np.digitize(processed["GenPart_pt"][ev][mu] / maxmuonpt[ev], ptbins, right=True) - 1]
#         for ijet, jet in enumerate(jet_candidates[ev]):
#             drprob = lq_muon_dr_binprob[np.digitize(
#                 np.clip(dr(processed["GenPart_eta"][ev][mu], processed["GenPart_phi"][ev][mu], processed["GenPart_eta"][ev][jet], processed["GenPart_phi"][ev][jet]), drbins[0], drbins[-1]),
#                 drbins, right=True
#             ) - 1]
#             print(jet_candidates[ev])
#             print(jet)
#             jetptprob = lq_jet_pt_binprob[np.digitize(processed["GenPart_pt"][ev][jet] / maxjetpt[ev], jetptbins, right=True) - 1]
#             probabilities[ev, imu, ijet] = muprob * jetptprob
#
# best_idx = np.argmax(probabilities.reshape(nevents, -1), axis=1)
# best_mu, best_jet = np.unravel_index(best_idx, probabilities.shape[1:])
# # print(best_mu[:5], best_jet[:5])
# # print(muon_candidates[:5])
# # print(jet_candidates[:5])
# # decode back into (mu, jet) indices
#
#
# # print(jet_candidates[best_jet][:20])
# # print(muon_candidates[best_mu][:20])
#
# selected_muons = muon_candidates[allevents, best_mu]
# selected_jets = jet_candidates[allevents, best_jet]
#
# print(selected_muons[:20])
# print(selected_jets[:20])
#
# print(processed["gen_lq_muon"][:20])
# print(processed["gen_lq_quark"][:20])
# SOMETHING IS WRONG WITH JET PT


# histograms (use sums=1 so they’re proper probabilities)
# ptbins = np.linspace(0, 1, 20)
# mu_pt = getobj("Muon_pt", "reco_lq_muon")      # length = nevents
# mu_pt_rel = mu_pt / reco_maxmuonpt             # same shape
# lq_muon_pt_counts, _ = np.histogram(ak.to_numpy(mu_pt_rel), bins=ptbins)
# lq_muon_pt_binprob = lq_muon_pt_counts / lq_muon_pt_counts.sum()

# rank1prob = np.sum(reco_lq_muon_rank == 1) / nevents
# lowerprob = 1 - rank1prob

# reco_quark_lq_muon_dr = dr(
#     getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet"),
#     getobj("Muon_eta", "reco_lq_muon"), getobj("Muon_phi", "reco_lq_muon")
# )
# drbins = np.linspace(0, 6, 7)
# lq_muon_dr_counts, _ = np.histogram(ak.to_numpy(reco_quark_lq_muon_dr), bins=drbins)
# lq_muon_dr_binprob = lq_muon_dr_counts / lq_muon_dr_counts.sum()
# print(lq_muon_dr_binprob)

# # per-event max jet pt among valid jets
# valid_jet_pt = ak.mask(processed["Jet_pt"], processed["Jet_jetId"] >= 6)
# maxjetpt = ak.to_numpy(ak.max(valid_jet_pt, axis=-1, mask_identity=False))  # shape (nevents,)

# jetptbins = np.linspace(0, 1, 20)
# reco_lq_jet_pt = getobj("Jet_pt", "reco_lq_jet")
# jet_pt_rel = reco_lq_jet_pt / maxjetpt
# lq_jet_pt_counts, _ = np.histogram(ak.to_numpy(jet_pt_rel), bins=jetptbins)
# lq_jet_pt_binprob = lq_jet_pt_counts / lq_jet_pt_counts.sum()

# # candidates
# nmuon_cand = 2
# muon_candidates = ak.argsort(processed["Muon_pt"], axis=-1, ascending=False)[:, :nmuon_cand]

# njet_cand = 4
# jet_candidates = ak.argsort(valid_jet_pt, axis=-1, ascending=False)[:, :njet_cand]

# probabilities = np.zeros((nevents, nmuon_cand, njet_cand))

# def getbinprob(value, probs, bins):
#     v = np.clip(value, bins[0], np.nextafter(bins[-1], -np.inf))
#     idx = np.digitize(v, bins, right=True) - 1
#     idx = np.clip(idx, 0, len(probs)-1)
#     return probs[idx]

# for ev in range(nevents):
#     for imu, mu in enumerate(muon_candidates[ev]):
#         # muprob = getbinprob(processed["Muon_pt"][ev][mu] / reco_maxmuonpt[ev],
#         #                     lq_muon_pt_binprob, ptbins)
#         if processed["Muon_pt"][ev][mu] == reco_maxmuonpt[ev]:
#             muprob = rank1prob
#         else:
#             muprob = lowerprob
#         for ijet, jet in enumerate(jet_candidates[ev]):
#             drval = dr(processed["Jet_eta"][ev][jet], processed["Jet_phi"][ev][jet],
#                        processed["Muon_eta"][ev][mu], processed["Muon_phi"][ev][mu])
#             drprob = getbinprob(drval, lq_muon_dr_binprob, drbins)
#             jetptprob = getbinprob(processed["Jet_pt"][ev][jet] / maxjetpt[ev],
#                                    lq_jet_pt_binprob, jetptbins)
#             probabilities[ev, imu, ijet] = jetptprob * muprob * drprob

# best_idx = np.argmax(probabilities.reshape(nevents, -1), axis=1)
# best_mu, best_jet = np.unravel_index(best_idx, probabilities.shape[1:])

# print(probabilities[:10])
# best_idx = np.argmax(probabilities.reshape(nevents, -1), axis=1)
# best_mu, best_jet = np.unravel_index(best_idx, probabilities.shape[1:])


# selected_muons = muon_candidates[allevents, best_mu]
# selected_jets = jet_candidates[allevents, best_jet]

# muon_correct = selected_muons == processed["reco_lq_muon"]
# jet_correct = selected_jets == processed["reco_lq_jet"]

# figname = f"bayesselection_{file[:-5]}.png"
# fig, ax = plt.subplots(1)
# cm = np.array([
#     [np.sum(muon_correct & jet_correct), np.sum(~muon_correct & jet_correct)],
#     [np.sum(muon_correct & ~jet_correct), np.sum(~muon_correct & ~jet_correct)],
# ], dtype=int) / nevents
# # optional: normalized version (percentages)
# im = ax.imshow(cm, origin="upper")  # or use cm_norm for percentages
# for (i, j), v in np.ndenumerate(cm):
#     ax.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=11,color="white")
# ax.set_xticks([0, 1], ["correct", "incorrect"])
# ax.set_yticks([0, 1], ["correct", "incorrect"], rotation=90)
# ax.set_xlabel("Muon selection")
# ax.set_ylabel("Jet selection")
# ax.set_title("Reco information")
# fig.tight_layout()
# plt.savefig(os.path.join(outpath, figname))