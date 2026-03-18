Photon-induced leptoquark production events have three final state objects.

Feynmann diagram

To have a sensitive search, we need to have a robust way of identifying these objects in signal events. There are often many more jets and muons in each signal event than these three final state objects, which presents a challenge. Furthermore, it is important to distinguish the initial state muon from the LQ decay muon. The purpose of this codebase is to train and test algorithms to make these object selections

get_samples.py: download samples if you are doing work locally on your machine

For efficiency, this codebase works with event data in numpy arrays rather than root files. 
The desired branches of the root files are loaded in step 1 and saved to numpy arrays. These are used for the analysis and training of ML models.
The script select.py takes paths to root files, and uses the ML models trained in this codebase to tag the events.

Unfortunately, the best discriminating power comes when a separate NN is trained to select objects for each mass point. This is largely because the best kinematic variable for identifying the correct LQ decay muon/jet pair is the invariant mass of the pair, which is muddled in training if there are multiple mass points with multiple "correct" invariant masses. If we train a NN on only the 400GeV LQ signal mass, the network essentially just learns to choose the muon/jet pair with invariant mass closest to 400GeV. If we train an NN from 300-1000GeV, then on a given event with many muon-jet pairs with invariant mass in this range, the choice is much harder.

Step 1: ground_truth.py: tags objects the objects with ground truth information
analysis.py: makes some basic plots of the kinematics for each object to

Storage structure for models and predictions: A full set of trainings for all masspoints gets its own pickled dictionary. The dictionary contains a list of masspoints Each entry in this list corresponds to a training.
Each "training" key contains a list of masspoints, a trained model, and then under the "performance" key, the selection rates.

In the final stage of the LQ analysis, we suppose the data has a particular signal mass, and then run the classifer, and compare this to simulation. This is repeated for each signal mass. This requires that we can run the classifier on the simulation in a comparable way. Every background event must be given a fake LQ mass so that we can do this classification, and it should also meanin at the object selection stage, we need to suppose each background event has a 