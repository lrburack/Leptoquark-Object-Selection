import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import awkward as ak
from Classifier import FinalModel, pt
from config import PROCESSED_SAMPLES_PATH, MODEL_PERFORMANCE, MODELS_DIRECTORY, MODEL_FILENAME, MASSPOINTS

masspoints = MASSPOINTS
# masspoints = [sys.argv[1]]

model = FinalModel(nmuons=4, njets=5)
# model = pt()
# ------ Should only need to change above this line ------ 

model_name = f"{model}/M{masspoints[0]}-{masspoints[-1]}"
model_dir = os.path.join(MODELS_DIRECTORY, model_name)
os.makedirs(model_dir, exist_ok=True)

# Train the model on all masses simultaneously
train_events = np.arange(8000)
test_events = np.arange(8000, 10000)

if model.needs_training:
    # Build the features and labels for each masspoint.
    features = np.empty(len(masspoints), dtype=object)
    labels = {
        "all_correct": np.empty(len(masspoints), dtype=object),
        "lq_mu": np.empty(len(masspoints), dtype=object),
        "lq_crea": np.empty(len(masspoints), dtype=object),
        "lq_jet": np.empty(len(masspoints), dtype=object),
    }
    for i, mass in enumerate(masspoints):
        file = f"processed_Signal_LQToBMu_M_{mass}_single.root"
        with open(os.path.join(PROCESSED_SAMPLES_PATH, file), "rb") as f:
            events = pickle.load(f)

        features[i] = model.build_features(events)
        labels["lq_mu"][i], labels["lq_crea"][i], labels["lq_jet"][i] = model.ground_truth(events)
        labels["all_correct"][i] = labels["lq_mu"][i] & labels["lq_crea"][i] & labels["lq_jet"][i]

    training_data = np.concatenate([features[i][train_events] for i in range(len(masspoints))], axis=0)
    training_labels = np.concatenate([labels["all_correct"][i][train_events] for i in range(len(masspoints))], axis=0)
    # Remove bad events
    goodevents_mask = np.any(training_labels, axis=1) & ~np.any(np.isnan(training_data), axis=(1,2))
    training_data = training_data[goodevents_mask]
    training_labels = training_labels[goodevents_mask]
    training_data = training_data.reshape(-1, training_data.shape[-1])
    print(type(training_labels))

    training_labels = training_labels.reshape(-1)
    model.train(training_data, training_labels)

# Save the model
with open(os.path.join(model_dir, MODEL_FILENAME), "wb") as f:
    pickle.dump(model, f)

# Test the model
# For each masspoint, store the retention rate, all correct rate, mu1, mu2, jet selection rates
performance = {
    "masspoints": masspoints,

    "retention_rate": np.zeros(len(masspoints)),
    "selection_rate": np.zeros(len(masspoints)),

    "muon_rate": np.zeros(len(masspoints)),
    "creation_muon_rate": np.zeros(len(masspoints)),
    "jet_rate": np.zeros(len(masspoints)),

    "muon_retention_rate": np.zeros(len(masspoints)),
    "creation_muon_retention_rate": np.zeros(len(masspoints)),
    "jet_retention_rate": np.zeros(len(masspoints)),
}

for i, mass in enumerate(masspoints):
    file = f"processed_Signal_LQToBMu_M_{mass}_single.root"
    with open(os.path.join(PROCESSED_SAMPLES_PATH, file), "rb") as f:
        events = pickle.load(f)
    mu1_idx, jet_idx, mu2_idx = model.predict(events, usetest=test_events)
    print(events["reco_lq_jet"][test_events][10:20])
    print(jet_idx[10:20])

    # Retention rates (i.e. fraction of events where the correct objects are in the candidate list and thus a correct selection is possible)
    # performance["retention_rate"][i]        = np.sum(np.any(labels["all_correct"][i][test_events], axis=1)) / len(test_events)
    # performance["muon_retention_rate"][i]   = np.sum(np.any(labels["lq_mu"][i][test_events], axis=1)) / len(test_events)
    # performance["creation_muon_retention_rate"][i] = np.sum(np.any(labels["lq_crea"][i][test_events], axis=1)) / len(test_events)
    # performance["jet_retention_rate"][i]    = np.sum(np.any(labels["lq_jet"][i][test_events], axis=1)) / len(test_events)
    # performance["muon_retention_rate"][i]   = np.sum(~np.isnan(events["reco_lq_muon"][test_events]) & (events["reco_lq_muon"][test_events] >= 0)) / len(test_events)
    # performance["creation_muon_retention_rate"][i] = np.sum(~np.isnan(events["reco_creation_muon"][test_events]) & (events["reco_creation_muon"][test_events] >= 0)) / len(test_events)
    # performance["jet_retention_rate"][i]    = np.sum(~np.isnan(events["reco_lq_jet"][test_events]) & (events["reco_lq_jet"][test_events] >= 0)) / len(test_events)
    # performance["retention_rate"][i]        = np.sum(
    #     (~np.isnan(events["reco_lq_muon"][test_events]) & (events["reco_lq_muon"][test_events] >= 0)) &
    #     (~np.isnan(events["reco_creation_muon"][test_events]) & (events["reco_creation_muon"][test_events] >= 0)) &
    #     (~np.isnan(events["reco_lq_jet"][test_events]) & (events["reco_lq_jet"][test_events] >= 0))
    # ) / len(test_events)

    # Selection rates (i.e. fraction of events where the model's selected objects are correct)
    performance["muon_rate"][i] = np.sum(events["reco_lq_muon"][test_events] == mu1_idx) / len(test_events)
    performance["creation_muon_rate"][i] = np.sum(events["reco_creation_muon"][test_events] == mu2_idx) / len(test_events)
    performance["jet_rate"][i] = np.sum(events["reco_lq_jet"][test_events] == jet_idx) / len(test_events)
    performance["selection_rate"][i] = np.sum((events["reco_lq_muon"][test_events] == mu1_idx) & (events["reco_creation_muon"][test_events] == mu2_idx) & (events["reco_lq_jet"][test_events] == jet_idx)) / len(test_events)

# Save the performance
with open(os.path.join(model_dir, MODEL_PERFORMANCE), "wb") as f:
    pickle.dump(performance, f)

# Everything below is just to plot performance! ----------------------------------------------------
# Note there is a separate script to plot several selection algorithms against eachother.

# Make a bar graph for each mass point showing the retention rates and selection rates for the muon, creation muon, and jet, as well as the overall selection rate.
for i, mass in enumerate(masspoints):
    figname = f"selection_rates_M{mass}.png"
    fig, ax = plt.subplots()
    ax.set_title(f"Selection rates for M={mass} GeV")
    categories = ["muon", "creation muon", "jet", "total"]
    # retention_rates = [performance["muon_retention_rate"][i], performance["creation_muon_retention_rate"][i], performance["jet_retention_rate"][i], performance["retention_rate"][i]]
    selection_rates = [performance["muon_rate"][i], performance["creation_muon_rate"][i], performance["jet_rate"][i], performance["selection_rate"][i]]
    x = np.arange(len(categories))
    width = 0.35
    # ax.bar(x - width/2, retention_rates, width, label="Retention rate")
    ax.bar(x + width/2, selection_rates, width, label="Selection rate")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.savefig(os.path.join(model_dir, figname))