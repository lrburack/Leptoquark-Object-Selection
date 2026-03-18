import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import awkward as ak
import re
from helpers import dr, dr_ak, invariant_mass, invariant_mass_pairwise
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from Classifier import FinalModel

# sample file: processed_Signal_LQToBMu_M_500_single.root
# selected objects: ["gen_lq_muon", "gen_creation_muon", "gen_lq_quark", "reco_lq_muon", "reco_creation_muon", "reco_lq_jet"]

match = re.search(r"_M_(\d+)", sys.argv[1])
masspoints = [int(match.group(1))]
nmuoncand = 4
njetcand = 3

# Used to get rid of the cases where the creation muon = signal muon
mu_idx = np.arange(nmuoncand)
mask = mu_idx[:, None] != mu_idx[None, :]

run_name = f"test4_M{masspoints[0]}"
signal_path = "/Users/lrburack/Documents/CERN/Leptoquark/processed_withbtag/"
results_file = "simultaneous_results.pkl"
save_fig = f"algo_figures/{run_name}.png"
model_save_path = f"models/{run_name}.pkl"

processed = np.empty(len(masspoints), dtype=object)

def getobj(b, selb):
    return events[b][allevents, events[selb]]

retention_rates = np.zeros(len(masspoints), dtype=float)

for i, mass in enumerate(masspoints):
    print("\nProcessing masspoint: ", mass)
    file = f"processed_Signal_LQToBMu_M_{mass}_single.root"
    with open(os.path.join(signal_path, file), "rb") as f:
        events = pickle.load(f)

    nevents = len(events["GenPart_pt"])
    allevents = np.arange(nevents)

    # Find the candidate objects.
    muonranks = ak.argsort(events["Muon_pt"], axis=1, ascending=False)
    cand_muons = (ak.local_index(muonranks, axis=1) < nmuoncand)[ak.argsort(muonranks, axis=1)]
    jet_conditions = (events["Jet_jetId"] >= 6)
    jetranks = ak.argsort(ak.where(jet_conditions, events["Jet_pt"], -1e9), axis=1, ascending=False)
    cand_jets = ((ak.local_index(jetranks, axis=1) < njetcand)[ak.argsort(jetranks, axis=1)]) & jet_conditions

    true_muon = events["reco_lq_muon"]
    true_jet = events["reco_lq_jet"]
    true_creation_muon = events["reco_creation_muon"]
    true_muon_dr = np.array(dr(getobj("Muon_eta", "reco_lq_muon"), getobj("Muon_phi", "reco_lq_muon"), getobj("Jet_eta", "reco_lq_jet"), getobj("Jet_phi", "reco_lq_jet")))
    true_muon_ptrank = np.array(ak.sum(events["Muon_pt"][cand_muons] > getobj("Muon_pt", "reco_lq_muon"), axis=1) + 1)
    true_jet_ptrank = np.array(ak.sum(events["Jet_pt"][cand_jets] > getobj("Jet_pt", "reco_lq_jet"), axis=1) + 1)

    print("We have made some initial selections. Lets check that the objects we want are still present.")
    print(f"Signal muon retention: {np.sum(cand_muons[allevents, true_muon]) / nevents}")
    print(f"Creation muon retention: {np.sum(cand_muons[allevents, true_creation_muon]) / nevents}")
    print(f"Signal jet retention: {np.sum(cand_jets[allevents, true_jet]) / nevents}")
    print(f"Events with all: {np.sum(cand_muons[allevents, true_muon] & cand_jets[allevents, true_jet] & cand_muons[allevents, true_creation_muon]) / nevents}")
    retention_rates[i] = np.sum(cand_muons[allevents, true_muon] & cand_jets[allevents, true_jet]) / nevents

    print("Some events may not have the full number of candidates")
    print(f"Events with < {nmuoncand} muon candidates: {np.sum(ak.sum(cand_muons, axis=1) != nmuoncand)}")
    print(f"Events with < {njetcand} jet candidates: {np.sum(ak.sum(cand_jets, axis=1) != njetcand)}")

    # Build the dataset for selection. 
    # Features: muon pt, jet pt, muon pt rank, jet pt rank, dr between muon and jet
    nfeatures = 12
    # Allow every combination of the three objects to have their own features.
    # The only silly thing about doing it this way is that you have to explicitly remove the cases where the creation muon = signal muon.
    #             event   signal mu, creation mu, signal jet, features
    X = np.zeros((nevents, nmuoncand, nmuoncand, njetcand, nfeatures), dtype=np.float32)
    muon_pts = ak.fill_none(ak.pad_none(events["Muon_pt"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy()
    X[:, :, :, :, 0] = muon_pts[:, :, np.newaxis, np.newaxis]
    X[:, :, :, :, 1] = muon_pts[:, np.newaxis, :, np.newaxis]
    jet_pts = ak.fill_none(ak.pad_none(events["Jet_pt"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy()
    X[:, :, :, :, 2] = jet_pts[:, np.newaxis, np.newaxis,:]
    muon_ranks = ak.local_index(ak.fill_none(ak.pad_none(events["Muon_pt"][cand_muons], nmuoncand, axis=1, clip=True), -1), axis=1) + 1
    X[:, :, :, :, 3] = muon_ranks[:, :, np.newaxis, np.newaxis]
    X[:, :, :, :, 4] = muon_ranks[:, np.newaxis, :, np.newaxis]
    jet_ranks = ak.local_index(ak.fill_none(ak.pad_none(events["Jet_pt"][cand_jets], njetcand, axis=1, clip=True), -1), axis=1) + 1
    X[:, :, :, :, 5] = jet_ranks[:, np.newaxis, np.newaxis, :]
    # Im going to include all of the drs between each pair of objects (three total)
    # First, dr between signal muon and jet
    drs = dr_ak(
        ak.fill_none(ak.pad_none(events["Muon_eta"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(), 
        ak.fill_none(ak.pad_none(events["Muon_phi"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(),
        ak.fill_none(ak.pad_none(events["Jet_eta"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy(), 
        ak.fill_none(ak.pad_none(events["Jet_phi"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy()
    )
    drs[drs==0] = -1
    X[:, :, :, :, 6] = drs[:, :, np.newaxis, :]
    # Now, dr between creation muon and jet
    X[:, :, :, :, 7] = drs[:, np.newaxis, :, :]
    # Now, dr between signal muon and creation muon
    drs = dr_ak(
        ak.fill_none(ak.pad_none(events["Muon_eta"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(), 
        ak.fill_none(ak.pad_none(events["Muon_phi"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(),
        ak.fill_none(ak.pad_none(events["Muon_eta"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(), 
        ak.fill_none(ak.pad_none(events["Muon_phi"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy()
    )
    drs[drs==0] = -1
    X[:, :, :, :, 8] = drs[:, :, :, np.newaxis]

    inv_mass = invariant_mass_pairwise(
        ak.fill_none(ak.pad_none(events["Muon_pt"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(),
        ak.fill_none(ak.pad_none(events["Muon_eta"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(),
        ak.fill_none(ak.pad_none(events["Muon_phi"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy(),
        ak.fill_none(ak.pad_none(events["Jet_pt"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy(), 
        ak.fill_none(ak.pad_none(events["Jet_eta"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy(), 
        ak.fill_none(ak.pad_none(events["Jet_phi"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy(), 
    )
    X[:, :, :, :, 9] = inv_mass[:, :, np.newaxis, :]
    X[:, :, :, :, 10] = inv_mass[:, np.newaxis, :, :]
    # Add the b tagging information!
    X[:, :, :, :, 11] = ak.fill_none(ak.pad_none(events["Jet_btagDeepFlavB"][cand_jets], njetcand, axis=1, clip=True), 0).to_numpy()[:, np.newaxis, np.newaxis, :]

    # Add eta, phi information
    # jet_etas = ak.fill_none(ak.pad_none(events["Jet_eta"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy()
    # X[:, :, :, 6] = jet_etas[:, np.newaxis, :]
    # jet_phis = ak.fill_none(ak.pad_none(events["Jet_phi"][cand_jets], njetcand, axis=1, clip=True), -1).to_numpy()
    # X[:, :, :, 7] = jet_phis[:, np.newaxis, :]
    # muon_etas = ak.fill_none(ak.pad_none(events["Muon_eta"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy()
    # X[:, :, :, 8] = muon_etas[:, :, np.newaxis]
    # muon_phis = ak.fill_none(ak.pad_none(events["Muon_phi"][cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy()
    # X[:, :, :, 9] = muon_phis[:, :, np.newaxis]
    # # Include other muon information
    # other_muon_pt = np.flip(muon_pts[:, :, np.newaxis], axis=1)
    # X[:, :, :, 10] = other_muon_pt
    # other_muon_eta = np.flip(muon_etas[:, :, np.newaxis], axis=1)
    # X[:, :, :, 11] = other_muon_eta
    # other_muon_phi = np.flip(muon_phis[:, :, np.newaxis], axis=1)
    # X[:, :, :, 12] = other_muon_phi


    padded_cand_muons = ak.fill_none(ak.pad_none(ak.local_index(cand_muons)[cand_muons], nmuoncand, axis=1, clip=True), -1).to_numpy()
    is_correct_muon = np.broadcast_to((padded_cand_muons == true_muon[:, None])[:,:,None,None], X.shape[:-1])
    is_correct_creation_muon = np.broadcast_to((padded_cand_muons == true_creation_muon[:, None])[:,None,:,None], X.shape[:-1])
    padded_cand_jets = ak.fill_none(ak.pad_none(ak.local_index(cand_jets)[cand_jets], njetcand, axis=1, clip=True), -1).to_numpy()
    is_correct_jet = np.broadcast_to((padded_cand_jets == true_jet[:, None])[:,None,None,:], X.shape[:-1])
    is_true_comb = is_correct_muon & is_correct_creation_muon & is_correct_jet

    # Sanity check: If you include this, the NN should perform perfectly
    # X[:, :, :, 13] = is_true_pair.astype(np.float32)

    # Apply mask and collapse muon-pair axes
    X = X[:, mask, :, :]
    is_true_comb_masked = is_true_comb[:, mask, :]

    processed[i] = {
        "X": X,
        "is_true_pair": is_true_comb_masked,
        "is_true_pair_full": is_true_comb,
        "is_true_muon": is_correct_muon,
        "is_true_jet": is_correct_jet,
        "is_true_creation_muon": is_correct_creation_muon,
        "cand_muons": padded_cand_muons,
        "cand_jets": padded_cand_jets,
    }

# Selection algorithm.
train_events = np.arange(8000)
test_events = np.arange(8000, 10000)

train_data = np.concatenate([processed[i]["X"][train_events].reshape(-1, processed[i]["X"].shape[-1]) for i in range(len(masspoints))], axis=0)
is_true_pair_train = np.concatenate([processed[i]["is_true_pair"][train_events].reshape(-1) for i in range(len(masspoints))], axis=0)
# weights = np.ones(len(train_data), dtype=np.float32)
# weights[is_true_pair_train] = 6.0

mlp = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(16, 16),   # small, fast
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=1024,
        max_iter=10000,                 # bump if you see ConvergenceWarning
        random_state=42,
        verbose=False,
    ))
])

mlp.fit(train_data, is_true_pair_train)#, clf__sample_weight=weights)

pt_selection_rates = np.zeros(len(masspoints))
NN_selection_rates = np.zeros(len(masspoints))
NN_muon_rates = np.zeros(len(masspoints))
NN_creation_muon_rates = np.zeros(len(masspoints))
NN_jet_rates = np.zeros(len(masspoints))

for i, mass in enumerate(masspoints):
    print("\nEvaluating masspoint: ", mass)
    test_data = processed[i]["X"][test_events].reshape(-1, processed[i]["X"].shape[-1])
    scores = mlp.predict_proba(test_data)[:,1]
    scores = scores.reshape(len(test_events), nmuoncand * (nmuoncand - 1), njetcand)
    # Send shape back to original, adding -1000 where signal muon = creation muon
    scores_full = np.full(
        (len(test_events), nmuoncand, nmuoncand, njetcand),
        -1000,
        dtype=scores.dtype
    )
    scores_full[:, mask, :] = scores
    scores = scores_full
    selected = scores == np.max(scores, axis=(1,2,3))[:, None, None, None] # confirmed choses one per event
    is_true_pair = processed[i]["is_true_pair_full"][test_events]

    NN_selection_rates[i] = np.sum((is_true_pair & selected)) / len(test_events)
    pt_selection_rates[i] = np.sum((is_true_pair[:, 0, 1, 0])) / len(test_events)
    NN_muon_rates[i] = np.sum((processed[i]["is_true_muon"][test_events] & selected)) / len(test_events)
    NN_creation_muon_rates[i] = np.sum((processed[i]["is_true_creation_muon"][test_events] & selected)) / len(test_events)
    NN_jet_rates[i] = np.sum((processed[i]["is_true_jet"][test_events] & selected)) / len(test_events)

this_run = {
    "masspoints": masspoints[0],
    "nmuoncand": nmuoncand,
    "njetcand": njetcand,
    "retention_rates": retention_rates,
    "selection_rates": NN_selection_rates,
    "pt_selection_rates": pt_selection_rates,
    "NN_muon_rates": NN_muon_rates,
    "NN_creation_muon_rates": NN_creation_muon_rates,
    "NN_jet_rates": NN_jet_rates,
    "model": mlp,
    "run_name": run_name,
}

# Validation
sel_inds = np.where(selected)[1:]

with open(f"validation/{run_name}.pkl", "wb") as f:
    pickle.dump(selected, f)

classifier_object = FinalModel(mlp, nmuoncand, njetcand)
with open(model_save_path, "wb") as f:
    pickle.dump(classifier_object, f)

if os.path.exists(results_file):
    with open(results_file, "rb") as f:
        existing_results = pickle.load(f)
else:
    existing_results = {}
if run_name not in list(existing_results.keys()):
    existing_results[run_name] = {key: [] for key in list(this_run.keys())}

# append this_run's values
for key, value in this_run.items():
    # in case new keys show up later:
    if key not in existing_results[run_name]:
        existing_results[run_name][key] = []
    existing_results[run_name][key].append(value)

with open(results_file, "wb") as f:
    pickle.dump(existing_results, f)

fig, ax = plt.subplots()
ax.set_title(f"Top {nmuoncand} muons and top {njetcand} jets")
ax.plot(masspoints, retention_rates, marker='o', color="black", linestyle="--", label=f"Best possible (after selecting candidates)")
ax.plot(masspoints, pt_selection_rates, marker='o', label="pt-based selection")
ax.plot(masspoints, NN_selection_rates, marker='o', label="neural network selection")
ax.set_xlabel("LQ Mass (GeV)")
ax.set_ylabel("Correct selection rate")
ax.legend()
plt.savefig(save_fig)

fig, ax = plt.subplots()
ax.set_title(f"Top {nmuoncand} muons and top {njetcand} jets")
ax.plot(masspoints, NN_selection_rates, marker='o', label="All three correct")
ax.plot(masspoints, NN_muon_rates, marker='o', label="Correct muon")
ax.plot(masspoints, NN_creation_muon_rates, marker='o', label="Correct initial state muon")
ax.plot(masspoints, NN_jet_rates, marker='o', label="Correct jet")
ax.set_xlabel("LQ Mass (GeV)")
ax.set_ylabel("Correct selection rate")
ax.legend()
plt.savefig("algo_figures/separate_selection_rates")