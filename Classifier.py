import numpy as np
import awkward as ak
from helpers import dr_ak, invariant_mass_pairwise

class Classifier:
    def __init__(self):
        pass

    def predict(self, event):
        raise NotImplementedError("Subclasses should implement this method.")


class FinalModel(Classifier):
    def __init__(self, trained_model, nmuons, njets):
        super().__init__()
        self.nmuons = nmuons
        self.njets = njets
        self.trained_model = trained_model

    def predict(self, event):
        nmuoncand = self.nmuons
        njetcand = self.njets

        # ── 1. Wrap scalar branches into awkward arrays (one-event "batch") ──
        def aw(branch):
            return ak.Array([list(branch)])

        # ── 2. Select candidate muons (top-pt, same as training) ──
        muon_pt   = aw(event.Muon_pt)
        muon_eta  = aw(event.Muon_eta)
        muon_phi  = aw(event.Muon_phi)

        muonranks  = ak.argsort(muon_pt, axis=1, ascending=False)
        cand_muons = (ak.local_index(muonranks, axis=1) < nmuoncand)[ak.argsort(muonranks, axis=1)]

        # ── 3. Select candidate jets (jetId >= 6, top-pt, same as training) ──
        jet_pt    = aw(event.Jet_pt)
        jet_eta   = aw(event.Jet_eta)
        jet_phi   = aw(event.Jet_phi)
        jet_jetId = aw(event.Jet_jetId)

        jet_conditions = jet_jetId >= 6
        jetranks   = ak.argsort(ak.where(jet_conditions, jet_pt, -1e9), axis=1, ascending=False)
        cand_jets  = ((ak.local_index(jetranks, axis=1) < njetcand)[ak.argsort(jetranks, axis=1)]) & jet_conditions

        # ── 4. Helper: pad & fill a masked array to a fixed length ──
        def pad(arr, mask, n):
            return ak.fill_none(
                ak.pad_none(arr[mask], n, axis=1, clip=True), -1
            ).to_numpy()  # shape (1, n)

        m_pt   = pad(muon_pt,  cand_muons, nmuoncand)   # (1, nmuoncand)
        m_eta  = pad(muon_eta, cand_muons, nmuoncand)
        m_phi  = pad(muon_phi, cand_muons, nmuoncand)
        j_pt   = pad(jet_pt,   cand_jets,  njetcand)    # (1, njetcand)
        j_eta  = pad(jet_eta,  cand_jets,  njetcand)
        j_phi  = pad(jet_phi,  cand_jets,  njetcand)

        m_rank = (ak.local_index(
            ak.fill_none(ak.pad_none(muon_pt[cand_muons], nmuoncand, axis=1, clip=True), -1),
            axis=1
        ) + 1).to_numpy()
        j_rank = (ak.local_index(
            ak.fill_none(ak.pad_none(jet_pt[cand_jets], njetcand, axis=1, clip=True), -1),
            axis=1
        ) + 1).to_numpy()

        # ── 5. Compute ΔR matrices ──
        dr_muon_jet = dr_ak(m_eta, m_phi, j_eta, j_phi)   # (1, nmuoncand, njetcand)
        dr_muon_jet[dr_muon_jet == 0] = -1

        dr_muon_muon = dr_ak(m_eta, m_phi, m_eta, m_phi)  # (1, nmuoncand, nmuoncand)
        dr_muon_muon[dr_muon_muon == 0] = -1

        # ── 6. Compute invariant-mass matrix ──
        inv_mass = invariant_mass_pairwise(
            m_pt, m_eta, m_phi,
            j_pt, j_eta, j_phi,
        )  # (1, nmuoncand, njetcand)

        # ── 7. Build feature tensor X  (1, nmu, nmu, njet, 12) ──
        nfeatures = 12
        X = np.zeros((1, nmuoncand, nmuoncand, njetcand, nfeatures), dtype=np.float32)
        X[:, :, :, :, 0]  = m_pt[:, :, np.newaxis, np.newaxis]       # signal muon pt
        X[:, :, :, :, 1]  = m_pt[:, np.newaxis, :, np.newaxis]       # creation muon pt
        X[:, :, :, :, 2]  = j_pt[:, np.newaxis, np.newaxis, :]       # jet pt
        X[:, :, :, :, 3]  = m_rank[:, :, np.newaxis, np.newaxis]     # signal muon pt rank
        X[:, :, :, :, 4]  = m_rank[:, np.newaxis, :, np.newaxis]     # creation muon pt rank
        X[:, :, :, :, 5]  = j_rank[:, np.newaxis, np.newaxis, :]     # jet pt rank
        X[:, :, :, :, 6]  = dr_muon_jet[:, :, np.newaxis, :]         # ΔR(signal μ, jet)
        X[:, :, :, :, 7]  = dr_muon_jet[:, np.newaxis, :, :]         # ΔR(creation μ, jet)
        X[:, :, :, :, 8]  = dr_muon_muon[:, :, :, np.newaxis]        # ΔR(signal μ, creation μ)
        X[:, :, :, :, 9]  = inv_mass[:, :, np.newaxis, :]            # m(signal μ, jet)
        X[:, :, :, :, 10] = inv_mass[:, np.newaxis, :, :]            # m(creation μ, jet)
        X[:, :, :, :, 11] = ak.fill_none(ak.pad_none(ak.local_index(aw(event.Jet_btagDeepFlavB)[cand_jets]), self.njets, axis=1, clip=True), 0).to_numpy()[:, np.newaxis, np.newaxis, :]

        # ── 8. Mask diagonal (signal μ == creation μ is unphysical) ──
        mu_idx = np.arange(nmuoncand)
        mask   = mu_idx[:, None] != mu_idx[None, :]                   # (nmu, nmu) bool

        X_masked = X[:, mask, :, :]                                   # (1, nmu*(nmu-1), njet, 11)
        X_flat   = X_masked.reshape(-1, nfeatures)                    # flatten all combos

        # ── 9. Score every combination, restore full shape ──
        scores_masked = self.trained_model.predict_proba(X_flat)[:, 1]
        scores_masked = scores_masked.reshape(1, nmuoncand * (nmuoncand - 1), njetcand)

        scores_full = np.full((1, nmuoncand, nmuoncand, njetcand), -1000.0, dtype=np.float32)
        scores_full[:, mask, :] = scores_masked

        # ── 10. Pick the best (signal_mu, creation_mu, jet) combo ──
        best = np.argmax(scores_full[0].ravel())
        sig_mu_cand, cre_mu_cand, jet_cand = np.unravel_index(best, (nmuoncand, nmuoncand, njetcand))

        # ── 11. Map candidate-local indices back to original event indices ──
        padded_cand_muons = ak.fill_none(
            ak.pad_none(ak.local_index(cand_muons)[cand_muons], nmuoncand, axis=1, clip=True), -1
        ).to_numpy()[0]  # (nmuoncand,)

        padded_cand_jets = ak.fill_none(
            ak.pad_none(ak.local_index(cand_jets)[cand_jets], njetcand, axis=1, clip=True), -1
        ).to_numpy()[0]  # (njetcand,)

        lq_muon_index            = int(padded_cand_muons[sig_mu_cand])
        initial_state_muon_index = int(padded_cand_muons[cre_mu_cand])
        lq_jet_index             = int(padded_cand_jets[jet_cand])

        return lq_muon_index, lq_jet_index, initial_state_muon_index