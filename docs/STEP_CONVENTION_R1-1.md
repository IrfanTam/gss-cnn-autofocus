# Step convention (reviewer comment R1-1)
One step = one position to which the stage moves and at which an image is captured and evaluated.
When the angular criterion ends the search, the returned midpoint (a+b)/2 has not been captured, so reaching it counts as one further step.
`gss_simulator.py` now adds `steps += 1` before returning the midpoint (GSS-only and GSS+classifier). The previous version is `gss_simulator_v_steps_old.py`; the previous outputs are in `_old_step_convention/`.
Classifier-triggered stops return an already captured frame and are unchanged. Success/failure results are unchanged; only step counts and times change.
Re-run order: select_on_val.py, rerun_unitfix.py, report_train_replay.py, nonlearning_stop.py, then ../04_supporting_analysis/make_figs_steps.py.
Cost model: n = 3 + ceil(log_phi(L0 / theta_min)). Predicted = observed: 11, 12, 12, 13, 15 steps for the five sessions.
