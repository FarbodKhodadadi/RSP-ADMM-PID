# Legacy starting point

`Bergamncontrol_original.mlx` is an unmodified copy of the MATLAB live script
supplied at the start of this work. It is retained only for provenance.

The published results do **not** come from cached outputs in that live script.
They were regenerated using the auditable source in `python/` and independently
mirrored in `matlab/`. Among other corrections, the new benchmark evaluates
tracking error against the declared glucose reference, separates
training/validation/test seeds, adds matched stochastic cohorts, and reports
subject-level paired statistics.
