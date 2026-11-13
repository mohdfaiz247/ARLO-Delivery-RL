# Repeated-seed research findings

Executed protocol: 5 independent training seeds [11, 22, 33, 44, 55]; 1,000 matched orders in each evaluation. Validation seed 82026; fresh final test seed 92026; economic sensitivity seed 102026.

## Predefined training study and coverage

Only validation means select the training setting. No final test result or favoured training seed is used for selection.

- reference_15k: 15,000 episodes, learning rate 0.1.
- longer_60k: 60,000 episodes, learning rate 0.1.
- longer_lower_alpha: 60,000 episodes, learning rate 0.03.

Allocated table size and distribution-supported size are different:
- q_learning, reference_15k: validation cost R$58.468; 2,844 supported states out of 44,064 allocated; state coverage 90.8%, feasible-action coverage 88.6%; 66.3% of supported feasible pairs have fewer than 10 visits. Unvisited chosen actions on validation: 0.85%.
- q_learning, longer_60k: validation cost R$57.293; 2,844 supported states out of 44,064 allocated; state coverage 97.7%, feasible-action coverage 97.2%; 28.0% of supported feasible pairs have fewer than 10 visits. Unvisited chosen actions on validation: 0.01%.
- q_learning, longer_lower_alpha: validation cost R$58.156; 2,844 supported states out of 44,064 allocated; state coverage 97.0%, feasible-action coverage 96.9%; 29.0% of supported feasible pairs have fewer than 10 visits. Unvisited chosen actions on validation: 0.08%.
- sarsa, reference_15k: validation cost R$58.423; 2,844 supported states out of 44,064 allocated; state coverage 91.0%, feasible-action coverage 88.9%; 66.4% of supported feasible pairs have fewer than 10 visits. Unvisited chosen actions on validation: 0.76%.
- sarsa, longer_60k: validation cost R$57.457; 2,844 supported states out of 44,064 allocated; state coverage 97.7%, feasible-action coverage 97.4%; 28.0% of supported feasible pairs have fewer than 10 visits. Unvisited chosen actions on validation: 0.01%.
- sarsa, longer_lower_alpha: validation cost R$58.329; 2,844 supported states out of 44,064 allocated; state coverage 97.3%, feasible-action coverage 97.1%; 28.7% of supported feasible pairs have fewer than 10 visits. Unvisited chosen actions on validation: 0.04%.

The support count is specific to the default generator and configured costs. Coverage is not a guarantee of accurate values. Lossless reindexing could remove unreachable allocated rows, but would not pool samples or improve statistical efficiency. The exact budget/service representation is retained to avoid hiding feasible decisions.

## Fresh final test

- q_learning: mean cost R$56.188; training-seed standard deviation R$0.227; approximate 95% seed-mean interval ±R$0.281; on-time 35.6%; saving versus no intervention R$3.357/order.
- sarsa: mean cost R$56.474; training-seed standard deviation R$0.282; approximate 95% seed-mean interval ±R$0.350; on-time 35.4%; saving versus no intervention R$3.071/order.
- No intervention: mean cost R$59.545; training-seed standard deviation R$0.000; approximate 95% seed-mean interval ±R$0.000; on-time 19.4%; saving versus no intervention R$0.000/order.
- Rule based: mean cost R$54.655; training-seed standard deviation R$0.000; approximate 95% seed-mean interval ±R$0.000; on-time 39.2%; saving versus no intervention R$4.890/order.

Lowest observed final-test mean: **Rule based**, R$54.655/order. This is a comparison of observed means, not a universal superiority claim.

- q_learning selected longer_60k on validation; final mean gap to the rule baseline is R$+1.533/order (positive means RL is more expensive).
- sarsa selected longer_60k on validation; final mean gap to the rule baseline is R$+1.819/order (positive means RL is more expensive).

Error bars use approximate Student-t intervals across training-seed means, conditional on the same final orders. The non-learning baselines are repeated on the identical order set and therefore have zero training-seed variability, not zero order uncertainty. Per-run cost and paired-savings Monte Carlo intervals are in final_runs.csv. These are separate uncertainty estimates; no combined real-world or hierarchical confidence claim is made.

## Economic sensitivity: frozen-policy stress tests

Selected policies are not retrained for changed prices or penalties. Budgets are already represented in mixed-budget training. Price changes can change both action feasibility and remaining-budget support; unseen-action fractions help distinguish coverage shift from the change in economics.

- q_learning, Budget: 0: R$+0.000 saved/order (unvisited chosen actions 0.0%); 8: R$+1.928 saved/order (unvisited chosen actions 0.0%); 25: R$+1.646 saved/order (unvisited chosen actions 0.0%); 32: R$+4.530 saved/order (unvisited chosen actions 0.0%); 40: R$+4.294 saved/order (unvisited chosen actions 0.1%); 50: R$+4.080 saved/order (unvisited chosen actions 0.0%).
- q_learning, Intervention price multiplier: 0.75: R$+6.172 saved/order (unvisited chosen actions 39.0%); 1: R$+2.731 saved/order (unvisited chosen actions 0.0%); 1.25: R$-0.045 saved/order (unvisited chosen actions 22.1%).
- q_learning, Late-day penalty: 7.5: R$-3.424 saved/order (unvisited chosen actions 0.0%); 15: R$+2.731 saved/order (unvisited chosen actions 0.0%); 30: R$+15.040 saved/order (unvisited chosen actions 0.0%).
- sarsa, Budget: 0: R$+0.000 saved/order (unvisited chosen actions 0.0%); 8: R$+1.535 saved/order (unvisited chosen actions 0.0%); 25: R$+1.145 saved/order (unvisited chosen actions 0.0%); 32: R$+4.503 saved/order (unvisited chosen actions 0.1%); 40: R$+4.025 saved/order (unvisited chosen actions 0.1%); 50: R$+3.975 saved/order (unvisited chosen actions 0.2%).
- sarsa, Intervention price multiplier: 0.75: R$+5.915 saved/order (unvisited chosen actions 39.4%); 1: R$+2.398 saved/order (unvisited chosen actions 0.0%); 1.25: R$-0.338 saved/order (unvisited chosen actions 20.7%).
- sarsa, Late-day penalty: 7.5: R$-3.793 saved/order (unvisited chosen actions 0.0%); 15: R$+2.398 saved/order (unvisited chosen actions 0.0%); 30: R$+14.779 saved/order (unvisited chosen actions 0.0%).

## Interpretation and limits

Use the validation learning-rate and episode comparisons to assess whether additional experience changes performance. Do not conclude that allocated table size alone caused the original RL losses. Even a visited action can have a noisy value; constant learning rates, exploration, finite sampling and a useful rule baseline remain plausible explanations. This experiment does not isolate every causal contributor.

All parameters and orders are synthetic. Five training seeds are a stronger repeatability check, not proof of convergence. Reported settings are selected within a small predefined grid. New environment prices/rewards need retraining for optimal adaptation; frozen stress tests demonstrate robustness limits. Real-world deployment requires calibration and external validation.

Reproduce with `python scripts/run_studies.py`. Original single-seed outputs remain separate in results/ and models/. Research model snapshots live in models/studies/. See PROFESSOR_DEMO_GUIDE.md for the presentation sequence.
