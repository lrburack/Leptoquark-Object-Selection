# Validate that sim_select_objects.py correctly applies the trained model to classify objects


import ROOT
import pickle
import numpy as np

# Load the annotated root file
file = "/Users/lrburack/Documents/CERN/Leptoquark/samples/LQ_2017merged/Signal_LQToBMu_M_500_single.root"
root_file = ROOT.TFile(file, "READ")
tree = root_file.Get("Events")
friend = tree.GetFriend("predictions")
print([b.GetName() for b in friend.GetListOfBranches()])
# n_events = tree.GetEntries()
# print([branch.GetName() for branch in tree.GetListOfBranches()])

file = f"validation/test4_M500.pkl"
with open(file, "rb") as f:
    validation_data = pickle.load(f)

best = np.where(validation_data)[1:]
print(best)
# sig_mu_cand, cre_mu_cand, jet_cand = np.unravel_index(best, (4, 4, 3))
# print(sig_mu_cand)

events_to_compare = np.arange(8000, 10000, dtype=int)  # Compare the last 2000 events

# Compare the predicted labels in the root file with the validation data
for i in events_to_compare:
    tree.GetEntry(int(i))
    friend.GetEntry(int(i))
    root_lq_muon_index = friend.GetBranch("test4_M500_lq_muon_index").GetLeaf("test4_M500_lq_muon_index").GetValue()
    root_lq_jet_index = friend.GetBranch("test4_M500_lq_jet_index").GetLeaf("test4_M500_lq_jet_index").GetValue()
    root_initial_state_muon_index = friend.GetBranch("test4_M500_initial_state_muon_index").GetLeaf("test4_M500_initial_state_muon_index").GetValue()

    print(root_lq_muon_index, root_lq_jet_index, root_initial_state_muon_index)

    val_lq_muon_index = best[0][i - events_to_compare[0]]
    val_initial_state_muon_index = best[1][i - events_to_compare[0]]
    val_lq_jet_index = best[2][i - events_to_compare[0]]

    print(val_lq_muon_index, val_lq_jet_index, val_initial_state_muon_index)

    assert root_lq_muon_index == val_lq_muon_index, f"Mismatch in lq muon index for event {i}"
    # assert root_lq_jet_index == val_lq_jet_index, f"Mismatch in lq jet index for event {i}"
    assert root_initial_state_muon_index == val_initial_state_muon_index, f"Mismatch in initial state muon index for event {i}"