"""AUDIT item 6: stage36 Arm B apples-to-apples + 431/3967 counts."""
import json
from collections import defaultdict
from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from icm.experiments.q01_can_voting_label_data.consensus_loop_15lang import metrics
import inspect
print("metrics() signature:", inspect.signature(metrics))
print(inspect.getsource(metrics))
