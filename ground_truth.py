import numpy as np
import uproot
import os
import sys
import pickle
from config import SIGNAL_SAMPLES_PATH, USE_BRANCHES, PROCESSED_SAMPLES_PATH

# Ground Truth: This script processes samples into a format usable for training a classification model
# It takes a root file and outputs a pickled dictionary with relevant kinematics and ground truth classification.
# There are three objects which should exist in both gen and reco: the leptoquark muon, the leptoquark jet, and the creation muon.

# Ground truth definitions: 
# Gen LQ muon: The highest pt final state muon (id==13, state==1) that can be traced back to the LQ only through muons.
# Gen creation muon: The highest pt final state muon (id==13, state==1) that can be traced back to the quark or gluon that created the LQ.
# Gen jet: Find the b quark that comes directly from the LQ, then find all the b quark children. Choose the highest pt one.
#   From what I've seen the LQ quark almost always has one pt change, and then that b quark has no children.
# Reco LQ muon: any reco muon that dr matches to a gen muon from the LQ decay is a candidate. the one that matches to the highest pt gen muon is the reco muon.
# 
# Reco jet: Must pass lepton veto (jetid>=6). Any reco jet that dr matches to a b quark from the LQ decay is a candidate. the one that matches to the highest pt gen b quark is the reco jet.
# It should output a pkl object with the relevant gen reco kinematics, and the indices of each object.
# Okay these gen definitions turn out to be pretty bad because the particle directly from the leptoquark goes through
# several pt changes before the final state, so I should definitely add final state as a condition.

# Notes:
# PartIdxMother -1 means no parent. Two of these in every event (the quark and gluon)
# Pdg id key: https://twiki.cern.ch/twiki/bin/view/Main/PdgId
# sample file: Signal_LQToBMu_M_500_single.root

def smart_format(x):
    # If value is "close" to an integer, print as int
    if np.isclose(x, int(x)):
        return f"{int(x):6d}"      # 6-wide field, integer format
    else:
        return f"{x:8.3f}"         # 8-wide field, 3 decimal places

np.set_printoptions(
    threshold=np.inf,
    linewidth=np.inf,
    formatter={"all": smart_format}
)

file = sys.argv[1]
nevents = 10000
print_every = 100

signal_path = SIGNAL_SAMPLES_PATH
outpath = PROCESSED_SAMPLES_PATH
os.makedirs(outpath, exist_ok=True)

uproot_file = uproot.open(os.path.join(signal_path, file))
branches = uproot_file["Events"].keys()

# There are a lot of branches and we dont need them all. 
events = {
    branch: uproot_file["Events"][branch].array()[:nevents] for branch in USE_BRANCHES
}
# Add extra branches for the ground truth objects that we will fill in
events.update({
    branch: np.zeros(nevents, dtype=np.int32) - 1 for branch in ["gen_lq_muon", "gen_creation_muon", "gen_lq_quark", "reco_lq_muon", "reco_creation_muon", "reco_lq_jet"]
})

LQ_id = 9000007
muon_id = 13
bquark_id = 5
jet_match_dr_cut = 0.2

def dr(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi  # wrap into [-π, π]
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def print_update():
    print("", flush=True)
    print(f"Processing event {i + 1}/{nevents}. Percentage of events with unidentified objects thus far:")
    branches = ["gen_lq_muon", "gen_creation_muon", "gen_lq_quark", "reco_lq_muon", "reco_creation_muon", "reco_lq_jet"]
    print(" | ".join(f"{b:>15}" for b in branches))
    row = []
    for branch in branches:
        frac = 100*np.sum(events[branch][:i] == -1) / (i+1)
        row.append(f"{frac:15.3f}%")
    print(" | ".join(row))


def findmother(part_index, stopat = [-1], gothrough=None):
    # This is a function to step back through the parent history of a gen particle
    # stopat defines the parent particle type to stop at, and gothrough defines the types of particles allowed as parents.
    current_particle = part_index
    while (events["GenPart_pdgId"][i][current_particle] not in stopat) and (gothrough is None or events["GenPart_pdgId"][i][current_particle] in gothrough) and (events["GenPart_genPartIdxMother"][i][current_particle] != -1):
        current_particle = events["GenPart_genPartIdxMother"][i][current_particle]
    return current_particle


for i in range(nevents):
    if (i + 1) % print_every == 0:
        print_update()

    ngenpart = len(events["GenPart_pdgId"][i])

    # This is a little silly but I wanted it to be compact. Basically, you identify the gen muon candidates, then find
    # their parents, and then find the highest pt particle that matches the selection criteria
    cands_gen_muon = (np.abs(events["GenPart_pdgId"][i]) == muon_id) & (events["GenPart_status"][i] == 1)
    muon_sources = np.array([findmother(index, stopat=[-1, LQ_id, -1*LQ_id], gothrough=[muon_id, -1*muon_id]) for index in np.where(cands_gen_muon)[0]])
    gen_lq_muon = np.zeros(len(cands_gen_muon), dtype=bool)
    gen_lq_muon[cands_gen_muon] = np.abs(events["GenPart_pdgId"][i][muon_sources]) == LQ_id
    # gen_lq_muon = np.where(cands_gen_muon, np.abs(events["GenPart_pdgId"][i][muon_sources]) == LQ_id, False)
    if np.any(gen_lq_muon): events["gen_lq_muon"][i] = np.argmax(np.where(gen_lq_muon, events["GenPart_pt"][i], -1))

    gen_creation_muon = np.zeros(len(cands_gen_muon), dtype=bool)
    gen_creation_muon[cands_gen_muon] = (muon_sources <= 1) & (muon_sources != -1)
    if np.any(gen_creation_muon): events["gen_creation_muon"][i] = np.argmax(np.where(gen_creation_muon, events["GenPart_pt"][i], -1))

    # Find all the bquarks with no child bquarks
    # childless_bquarks = (np.abs(events["GenPart_pdgId"][i]) == bquark_id) & (~np.isin(np.arange(ngenpart), events["GenPart_genPartIdxMother"][i][np.abs(events["GenPart_pdgId"][i]) == bquark_id]))
    # gen_lq_quark = np.array([
    #     childless_bquarks[index] and np.abs(events["GenPart_pdgId"][i][findmother(index, stopat=[-1, LQ_id, -1*LQ_id], gothrough=[bquark_id, -1*bquark_id])]) == LQ_id
    #     for index in range(ngenpart)
    # ])
    gen_lq_quark = (np.abs(events["GenPart_pdgId"][i]) == bquark_id) & (np.abs(events["GenPart_pdgId"][i][events["GenPart_genPartIdxMother"][i]]) == LQ_id)
    if np.any(gen_lq_quark): events["gen_lq_quark"][i] = np.argmax(np.where(gen_lq_quark, events["GenPart_pt"][i], -1))

    # Reco muons (there are two real muons in these events)
    lq_candidates = np.zeros(events["nMuon"][i], dtype=np.int32) - 1
    creation_candidates = np.zeros(events["nMuon"][i], dtype=np.int32) - 1
    for candidate in np.arange(events["nMuon"][i]):
        # The corresponding gen muon
        current_particle = events["Muon_genPartIdx"][i][candidate]

        # Follow back the gen muon until something thats not a muon
        while current_particle != -1 and np.abs(events["GenPart_pdgId"][i][current_particle]) == muon_id:
            current_particle = events["GenPart_genPartIdxMother"][i][current_particle]
        
        # Check what type of particle the muon came from
        if np.abs(events["GenPart_pdgId"][i][current_particle]) == LQ_id:
            lq_candidates[candidate] = events["Muon_genPartIdx"][i][candidate]
        elif current_particle <= 1:# and events["Muon_genPartIdx"][i][candidate] != -1: # This means the muon was created by the quark or gluon when the LQ was created.
            creation_candidates[candidate] = events["Muon_genPartIdx"][i][candidate]

    # Find the highest pt candidates
    if np.any(lq_candidates):
        events["reco_lq_muon"][i] = np.argmax(np.where(lq_candidates != -1, events["GenPart_pt"][i][lq_candidates], -1))
    if np.any(creation_candidates):
        events["reco_creation_muon"][i] = np.argmax(np.where(creation_candidates != -1, events["GenPart_pt"][i][creation_candidates], -1))

    jetsources = np.zeros(events["nJet"][i], dtype=np.int32) - 1
    isquark = np.abs(events["GenPart_pdgId"][i]) == bquark_id
    for recojet in range(events["nJet"][i]):
        if events["Jet_jetId"][i][recojet] < 6: # Lepton veto
            continue
        # Pairwise dr to all gen particles
        DR = dr(events["Jet_eta"][i][recojet], events["Jet_phi"][i][recojet],
                events["GenPart_eta"][i], events["GenPart_phi"][i])
        DR_isquark = np.array(DR)
        DR_isquark[~isquark] = 10000 # Only match to quarks

        if np.min(DR_isquark) > jet_match_dr_cut and np.argmin(DR) != np.argmin(DR_isquark):
            continue # No good quark match, and the closest match isnt a quark
        # Jet source should be the closest dr matched quark
        jet_source = np.argmin(DR_isquark)

        current_particle = jet_source # Follow the quark back to see if it came from the LQ
        while current_particle != -1 and np.abs(events["GenPart_pdgId"][i][current_particle]) != LQ_id:
            current_particle = events["GenPart_genPartIdxMother"][i][current_particle]
        if np.abs(events["GenPart_pdgId"][i][current_particle]) == LQ_id: # Child jet
            jetsources[recojet] = jet_source

    if np.any(jetsources != -1):
        jetsource_pts = np.where(jetsources != -1, events["GenPart_pt"][i][jetsources], -1)
        events["reco_lq_jet"][i] = np.argmax(jetsource_pts)

print_update()

with open(os.path.join(outpath, f"processed_{file}"), "wb") as f:
    pickle.dump(events, f)