import numpy as np
import awkward as ak
from helpers import dr, invariant_mass
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class Classifier:
    def __init__(self, needs_training=False):
        self.needs_training = needs_training

    def predict(self, event):
        raise NotImplementedError("Subclasses should implement this method.")

class FinalModel(Classifier):
    def __init__(self, nmuons, njets):
        super().__init__(needs_training=True)
        self.nmuons = nmuons
        self.njets = njets
        self.trained_model = None

        triplets = [(m1, m2, j) for m1, m2, j in product(range(nmuons), range(nmuons), range(njets)) if m1 != m2]
        # This has length ncombinations. Each row tells you which objects are in the row
        # For example row 0 is (0 1 0), meaning it corresponds to signal mu rank 1, creation mu rank 2, jet rank 1.
        self.triplet_indices = np.array(triplets)
        # These tell you which objects are in each row.
        self.m1_idx = self.triplet_indices[:, 0] # i.e. in the i'th combination, the signal muon's pt rank is self.m1_idx[i]
        self.m2_idx = self.triplet_indices[:, 1]
        self.j_idx  = self.triplet_indices[:, 2]

    def jet_cands(self, events):
        jet_conditions = events["Jet_jetId"] >= 6
        jetranks   = ak.argsort(ak.where(jet_conditions, events["Jet_pt"], -1e9), axis=1, ascending=False)
        cand_jets = ak.pad_none(jetranks, self.njets, axis=1, clip=True)
        return ak.fill_none(cand_jets, -1)
    
    def muon_cands(self, events):
        muonranks = ak.argsort(events["Muon_pt"], axis=1, ascending=False)
        cand_muons = ak.pad_none(muonranks, self.nmuons, axis=1, clip=True)
        return ak.fill_none(cand_muons, -1)
    
    def build_features(self, events):
        nevents = len(events["Muon_pt"])
        cand_jets = self.jet_cands(events)
        cand_muons = self.muon_cands(events)

        # cand_muons is an array of shape nevents x nmuons
        # For each event, it should contain indices to muons, with indices listed in decending order of pT

        # --- Helper to pad, fill, and convert to numpy ---
        def get_candidate_properties(arr, cands):
            maxlength = ak.max(ak.num(arr, axis=1))
            padded = ak.pad_none(arr, maxlength + 1, axis=1)
            return padded[np.arange(len(padded))[:, None], cands]

        # --- Extract padded candidate arrays once ---
        mu_pt  = get_candidate_properties(events["Muon_pt"], cand_muons)
        mu_eta = get_candidate_properties(events["Muon_eta"], cand_muons)
        mu_phi = get_candidate_properties(events["Muon_phi"], cand_muons)

        j_pt   = get_candidate_properties(events["Jet_pt"], cand_jets)
        j_eta  = get_candidate_properties(events["Jet_eta"], cand_jets)
        j_phi  = get_candidate_properties(events["Jet_phi"], cand_jets)
        j_btag = get_candidate_properties(events["Jet_btagDeepFlavB"], cand_jets)

        # --- Compute features for each triplet ---
        # Ranks are just candidate-list position + 1 (since candidates are
        # already sorted by pt), broadcast to (nevents, ncomb).
        ones = np.ones((nevents, 1), dtype=np.float32)

        # Build the features array
        X = np.stack([
            # 0: signal muon pt
            mu_pt[:, self.m1_idx],
            # 1: creation muon pt
            mu_pt[:, self.m2_idx],
            # 2: jet pt
            j_pt[:, self.j_idx],
            # 3: signal muon pt rank
            ones * (self.m1_idx + 1).astype(np.float32),
            # 4: creation muon pt rank
            ones * (self.m2_idx + 1).astype(np.float32),
            # 5: jet pt rank
            ones * (self.j_idx + 1).astype(np.float32),
            # 6: dR(signal muon, jet)
            dr(mu_eta[:, self.m1_idx], mu_phi[:, self.m1_idx],
                j_eta[:, self.j_idx],   j_phi[:, self.j_idx]),
            # 7: dR(creation muon, jet)
            dr(mu_eta[:, self.m2_idx], mu_phi[:, self.m2_idx],
                j_eta[:, self.j_idx],   j_phi[:, self.j_idx]),
            # 8: dR(signal muon, creation muon)
            dr(mu_eta[:, self.m1_idx], mu_phi[:, self.m1_idx],
                mu_eta[:, self.m2_idx], mu_phi[:, self.m2_idx]),
            # 9: invariant mass(signal muon, jet)
            invariant_mass(mu_pt[:, self.m1_idx], mu_eta[:, self.m1_idx], mu_phi[:, self.m1_idx],
                        j_pt[:, self.j_idx],   j_eta[:, self.j_idx],   j_phi[:, self.j_idx]),
            # 10: invariant mass(creation muon, jet)
            invariant_mass(mu_pt[:, self.m2_idx], mu_eta[:, self.m2_idx], mu_phi[:, self.m2_idx],
                        j_pt[:, self.j_idx],   j_eta[:, self.j_idx],   j_phi[:, self.j_idx]),
            # 11: jet b-tag score
            j_btag[:, self.j_idx],
        ], axis=-1)  # shape: (nevents, ncomb, nfeatures)

        return np.array(X)

    def ground_truth(self, events):
        cand_jets = self.jet_cands(events)
        cand_muons = self.muon_cands(events)

        # Map candidate ranks to actual event-level indices
        actual_m1 = cand_muons[:, self.m1_idx]  # shape (nevents, ncomb)
        actual_m2 = cand_muons[:, self.m2_idx]
        actual_j  = cand_jets[:, self.j_idx]

        lq_mu  = events["reco_lq_muon"][:, np.newaxis] == actual_m1
        lq_cre = events["reco_creation_muon"][:, np.newaxis] == actual_m2
        lq_jet = events["reco_lq_jet"][:, np.newaxis] == actual_j

        return np.array(lq_mu), np.array(lq_cre), np.array(lq_jet)


    def train(self, train_data, train_labels):
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

        mlp.fit(train_data, train_labels)
        self.trained_model = mlp

    def predict(self, events, usetest=None):
        features = self.build_features(events)
        if usetest is None:
            usetest = np.arange(len(events["Muon_pt"]))
        features = features[usetest]  # shape (nevents, ncomb, nfeatures)

        cand_jets = self.jet_cands(events)
        cand_muons = self.muon_cands(events)
        
        scores = np.full(len(usetest) * np.shape(features)[1], -1e9, dtype=np.float32) # length (nevents * ncomb)
        # Need to remove the cominations with nans (i.e. in events with less than nmuon muons or njet jets)
        combinations_flattened = features.reshape(-1, features.shape[-1]) # shape (nevents * ncomb, nfeatures)
        good_rows = ~np.any(np.isnan(combinations_flattened), axis=1) # boolean array to mask combinations_flattened

        scores[good_rows] = self.trained_model.predict_proba(combinations_flattened[good_rows])[:,1]
        scores = scores.reshape(len(usetest), features.shape[1]) # reshape back to (nevents, ncomb)
        best_combination_indices = np.argmax(scores, axis=1) # shape (nevents,)
        best_m1_idx = self.m1_idx[best_combination_indices]
        best_m2_idx = self.m2_idx[best_combination_indices]
        best_j_idx  = self.j_idx[best_combination_indices]

        return cand_muons[usetest, best_m1_idx], cand_jets[usetest, best_j_idx], cand_muons[usetest, best_m2_idx]
    
    def __str__(self):
        return "NN_{}mu_{}jet".format(self.nmuons, self.njets)

class pt(Classifier):
    def __init__(self):
        super().__init__()

    def predict(self, events, usetest=None):
        if usetest is None:
            usetest = np.arange(len(events["Muon_pt"]))
        lq_muon_index = np.argmax(events["Muon_pt"][usetest], axis=1)
        
        jet_conditions = events["Jet_jetId"][usetest] >= 6
        jetranks   = ak.argsort(ak.where(jet_conditions, events["Jet_pt"][usetest], -1e9), axis=1, ascending=False)
        lq_jet_index = jetranks[:, 0]  # take the leading jet that passes the jet ID

        has_two_muons = ak.num(events["Muon_pt"][usetest]) > 1
        initial_state_muon_index = np.zeros(len(usetest), dtype=int)
        initial_state_muon_index[has_two_muons] = np.argsort(events["Muon_pt"][usetest], axis=1)[has_two_muons][: , -2]

        return lq_muon_index, lq_jet_index, initial_state_muon_index
    
    def __str__(self):
        return "pt"