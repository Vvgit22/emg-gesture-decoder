# Why only these npz are committed

protocol_v1 checkpoints (.pt) were overwritten by later experiments and are
unrecoverable — these 10 files (cnn + lda, both, cross_subject) are the only
remaining record of v1 model predictions. Everything under
artifacts/experiments/*/predictions/ is regenerable from checkpoints via
`make predictions EXP=<name>` and stays local.
