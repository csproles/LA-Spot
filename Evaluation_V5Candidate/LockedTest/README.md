# Locked-test evaluation artifacts

`_vendored_v4_frozen_model.pkl` -- read-only, byte-identical copy of
`pipeline_v4/frozen_model.pkl`, extracted via
`git show user-shree:pipeline_v4/frozen_model.pkl` without checking out or
modifying `user-shree`. Used here only to re-derive V4's own per-image
locked-test predictions (needed for a paired comparison; the existing
`Evaluation_V4/LockedTest/locked_test_v2_v3_v4.json` only stores aggregate
confusion-matrix counts, not per-image scores). Not refit, not modified.

See `run_locked_test_evaluation.py` and `experiment_log.md` for the full
one-time locked-test run.
