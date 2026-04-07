# Needed for running classification with a pretrained model 
MODELS_DIRECTORY = "models"
MODEL_FILENAME = "model.pkl"
MODEL_PERFORMANCE = "performance.pkl"
PERFORMANCE_PLOT_PATH = "performance_plots"

MASSPOINTS = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3500, 4000]
USE_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_genPartIdxMother", "GenPart_pdgId", "GenPart_status", "nGenPart"]
USE_BRANCHES += ["Muon_eta", "Muon_phi", "Muon_pt", "Muon_genPartIdx", "nMuon"]
USE_BRANCHES += ["Jet_eta", "Jet_phi", "Jet_pt", "nJet", "Jet_genJetIdx", "Jet_jetId", "Jet_btagDeepFlavB"]

# Branch names to add to the root file for the model predictions.
LQ_MUON_BRANCH = "lq_muon"
INITIAL_STATE_MUON_BRANCH = "is_muon"
LQ_JET_BRANCH = "lq_jet"

# Needed only for training the models
SIGNAL_SAMPLES_PATH = "/Users/lrburack/Documents/CERN/Leptoquark/samples/LQ_2017merged"
PROCESSED_SAMPLES_PATH = "/Users/lrburack/Documents/CERN/Leptoquark/processed_withbtag"