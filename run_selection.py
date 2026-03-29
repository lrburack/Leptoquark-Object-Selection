import numpy as np
import uproot
import matplotlib.pyplot as plt
import os
import sys
import pickle
import ROOT
import ctypes

# This script operates on pre-slimming signal and background files. 
# Load the unlabeled root files. For each event, compute the inputs to the model and run the model to get the predicted labels.
# There will be several models for various mass ranges. All models will compute their predictions for each event.
# Each prediction will require three branches to store the indexes of the predicted lq_muon, lq_jet, initial_state_muon.

# The naming convention for branches will be "modelname_lq_muon_index", "modelname_lq_jet_index", "modelname_initial_state_muon_index"
# I have decided to use the lazy loading in ROOT rather than loading the entire file with uproot

models = {
    "test4_M500": "/Users/lrburack/Documents/CERN/Leptoquark/lq_object_selection/models/test4_M500.pkl",
}
loaded_models = {}
for model_name, model_path in models.items():
    with open(model_path, "rb") as f:
        loaded_models[model_name] = pickle.load(f)


file_path = sys.argv[1]

print("Adding prediction labels from models: ", list(loaded_models.keys()))
print("To the root file: ", file_path)

# Open the root file with the standard root library to be looped through event by event
file = ROOT.TFile(file_path, "UPDATE")
tree = file.Get("Events")
# Loop through the events and compute the model predictions
# Create a friend tree to hold the new branches separately
friend_tree = ROOT.TTree("predictions", "predictions")
buffers = {}
for model_name in loaded_models.keys():
    buffers[model_name] = {
        "lq_muon_index":            ctypes.c_int(0),
        "lq_jet_index":             ctypes.c_int(0),
        "initial_state_muon_index": ctypes.c_int(0),
    }
    friend_tree.Branch(model_name + "_lq_muon_index",
                       buffers[model_name]["lq_muon_index"],
                       model_name + "_lq_muon_index/I")
    friend_tree.Branch(model_name + "_lq_jet_index",
                       buffers[model_name]["lq_jet_index"],
                       model_name + "_lq_jet_index/I")
    friend_tree.Branch(model_name + "_initial_state_muon_index",
                       buffers[model_name]["initial_state_muon_index"],
                       model_name + "_initial_state_muon_index/I")

n_events = tree.GetEntries()
for i, event in enumerate(tree):
    for model_name, model in loaded_models.items():
        lq_muon_index, lq_jet_index, initial_state_muon_index = model.predict(event)
        buffers[model_name]["lq_muon_index"].value            = lq_muon_index
        buffers[model_name]["lq_jet_index"].value             = lq_jet_index
        buffers[model_name]["initial_state_muon_index"].value = initial_state_muon_index

    friend_tree.Fill()  # Fill the FRIEND, not the original tree
    print(f"\rProcessing events: {i+1}/{n_events} [{int((i+1)/n_events*50)*'█'}{(50-int((i+1)/n_events*50))*'░'}]", end="", flush=True)

friend_tree.Write("", ROOT.TObject.kOverwrite)
tree.AddFriend("predictions")
tree.Write("", ROOT.TObject.kOverwrite)
file.Close()