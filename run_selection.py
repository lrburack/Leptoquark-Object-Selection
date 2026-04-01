import numpy as np
import uproot
import matplotlib.pyplot as plt
import os
import sys
import pickle
import ROOT
import ctypes
from config import MODELS_DIRECTORY, MODEL_FILENAME, USE_BRANCHES, INITIAL_STATE_MUON_BRANCH, LQ_MUON_BRANCH, LQ_JET_BRANCH
from array import array

# This script operates on pre-slimming signal and background files. 
# Load the unlabeled root files. For each event, compute the inputs to the model and run the model to get the predicted labels.
# There will be several models for various mass ranges. All models will compute their predictions for each event.
# Each prediction will require three branches to store the indexes of the predicted lq_muon, lq_jet, initial_state_muon.

models = [
    "NN_4mu_5jet/M500-4000", 
]
file_path = sys.argv[1]


loaded_models = {}
for model_name in models:
    with open(os.path.join(MODELS_DIRECTORY, model_name, MODEL_FILENAME), "rb") as f:
        loaded_models[model_name] = pickle.load(f)

print("Adding prediction labels from models: ", list(loaded_models.keys()))
print("To the root file: ", file_path)

# -------- Compute the model predictions vectorized --------
uproot_file = uproot.open(file_path)
events = {
    branch: uproot_file["Events"][branch].array() for branch in USE_BRANCHES
}
classifications = {model_name: [] for model_name in loaded_models.keys()}
for model_name in loaded_models.keys():
    classifications[model_name] = loaded_models[model_name].predict(events)
    
# ------- Now actually write these to the ROOT file. Loop through the events with ROOT to avoid loading everything into memory at once with uproot. -------
# Thank you Claude. I hope never to deal with ROOT's C++ interface
# ------- Set up branches and buffers -------
file = ROOT.TFile(file_path, "UPDATE")
tree = file.Get("Events")
n_events = tree.GetEntries()

branch_buffers = {}
for model_name in loaded_models.keys():
    lq_mu, lq_jet, lq_cre = classifications[model_name]
    safe_name = model_name.replace("/", "_")
    for suffix, arr in [
        (INITIAL_STATE_MUON_BRANCH, lq_cre),
        (LQ_MUON_BRANCH, lq_mu),
        (LQ_JET_BRANCH, lq_jet),
    ]:
        branch_name = f"{safe_name}_{suffix}"
        buf = array('i', [0])  # int buffer
        branch_buffers[branch_name] = (buf, arr)
        tree.Branch(branch_name, buf, f"{branch_name}/I")

# ------- Fill the new branches -------
for i in range(n_events):
    for branch_name, (buf, arr) in branch_buffers.items():
        buf[0] = int(arr[i])
    tree.GetEntry(i)  # not strictly needed but keeps the tree in sync
    for branch_name, (buf, arr) in branch_buffers.items():
        tree.GetBranch(branch_name).Fill()

    print(f"\rProcessing events: {i+1}/{n_events} [{int((i+1)/n_events*50)*'█'}{(50-int((i+1)/n_events*50))*'░'}]", end="", flush=True)

tree.Write("", ROOT.TObject.kOverwrite)
file.Close()