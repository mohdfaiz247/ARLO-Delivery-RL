# Professor demonstration and viva guide

## Latest result to present

Use the updated 10-slide presentation with the 200k training results. The current individual-policy five-seed means on 5,000 fresh matched orders are R$53.29616 for Q-learning, R$53.37316 for SARSA, R$53.45220 for the unchanged rule baseline, and R$58.04500 for no intervention. Both RL means are slightly lower than the rule, but the margins are small and do not prove a universal ranking. Cite `results/improved_training/REPORT.md` and the paired intervals in `final_runs.csv`.

The extended study selects a visit-based step size `alpha_n=1/n^0.6`, trains for 200,000 episodes and lowers final exploration to 0.01. Standard Q-learning and SARSA targets remain intact. No simulator or baseline rule changed. This is a training-package improvement, not an isolated causal test of one hyperparameter. In Research experiments, leave the extended study toggle on and load the first listed seed. The live simulator defaults to the first predeclared extended Q-learning seed when available. Additional mean-Q ensembles are labelled separately and do not supply the main individual-policy results.

The walkthrough below describes the original study as well. Use the extended view and its metadata for the current presentation, keeping the earlier 60k results as a transparent before/after comparison.

## A seven-minute demonstration

**0:00–0:45 — State the question.** Open Overview. “We want to minimise delivery intervention expense plus penalties for late delivery. The cheapest intervention is not always the cheapest final outcome.” State clearly that orders and effectiveness parameters are synthetic.

**0:45–1:45 — Explain the sequential problem.** Open RL environment. One order makes three decisions. Buying a service changes the delivery outlook, remaining budget and later valid actions. Show an unaffordable action disappearing and a purchased service being masked. A single-stage contextual bandit would miss these dependencies.

**1:45–2:45 — Show genuine learning.** Open Train the agents. Contrast Q-learning’s maximum-valid-action target with SARSA’s actually selected next action. Show exploration decay and real recorded returns. With gamma=1, accumulated reward equals negative total economic cost. Training return includes exploratory mistakes, so it is not the final evaluation metric.

**2:45–4:00 — Walk through an order.** In Research experiments → Fresh final test, load a selected research model using the first listed training seed, rather than choosing a seed based on its final cost. Open Live order simulator, select that model and start a seeded order. Advance stage by stage. Explain that action selection comes from learned Q-values, while success and disturbances are stochastic simulator outcomes. At delivery, reconcile intervention spending and late penalties and show the matched no-intervention comparison.

**4:00–5:30 — Present the evidence.** Open Research experiments. Show the three predefined training configurations, five training seeds and reachable-support coverage. Explain that settings were chosen on validation orders, then tested on a separate order set. Show actual final results, including the rule baseline, whether or not RL wins.

**5:30–6:30 — Explain when intervention pays.** Open Economic sensitivity. Inspect savings against zero as budget, prices or late-day penalties change. These are frozen-policy stress tests; changing the reward does not magically retrain the policy. New prices can produce unvisited exact-budget states, so interpret the coverage measurements too.

**6:30–7:00 — Close with limits.** The policy learns a synthetic MDP, not a proven real-world causal intervention effect. Mention integer days, fixed per-order load/distance, absent shared capacity and finite training. Explain what calibration and larger repeated studies would add.

## Questions to be ready for

### Why is this reinforcement learning rather than supervised learning?

There are no labelled optimal actions. The agent obtains rewards from interacting with the simulator and updates estimates of future return. Action choices affect later states, budgets and opportunities.

### What exactly is the state?

Stage, signed integer-day outlook, operational load, distance, exact remaining budget and purchased-service bits. Risk and coarse late-day bands are derived displays. Exact budget and services prevent states with different affordable actions or already purchased services from being confused.

### Why is the table so large? Does a smaller table fix learning?

The rectangular table allocates 44,064 states, but only a subset is supported by the default order distribution. Measure coverage on that subset. Losslessly removing unreachable rows saves memory; it does not create additional learning experience. Coarsening budget or service status can remove information and violate the intended MDP. Poorly sampled feasible alternatives, not allocated size alone, are a more useful diagnostic.

### What is the difference between Q-learning and SARSA?

Q-learning bootstraps from the best valid next action, even if behaviour explores something else. SARSA bootstraps from the actual next epsilon-greedy action. At the terminal transition neither uses future values. Both mask impossible actions.

### Why gamma=1?

Episodes have only three decisions and always terminate. Gamma=1 aligns return exactly with the sum of monetary costs. Lower gamma changes how later penalties are valued.

### What does a negative Q-value mean?

Expected future negative economic cost under the learning objective. A value of −60 estimates roughly R$60 of remaining expected cost with gamma=1, subject to estimation error. Q-values are not probabilities. Zero can indicate an unvisited action, not a genuinely free outcome.

### Why can a simple rule outperform RL?

The rule encodes useful domain assumptions immediately. RL must discover the trade-offs from finite, noisy interactions across many contexts. Sparse action coverage, learning rate, exploration and insufficient training can matter. A stronger baseline makes the comparison meaningful. Use the executed training study to support any explanation; do not claim a diagnosis is proven solely because it sounds plausible.

### How are the comparisons fair?

Policies receive the same initial orders and stage-indexed exogenous disturbances. They cannot observe future draws. Evaluation freezes the Q-table and uses independent seed namespaces. The five training seeds vary learned policies; a fresh final order set is separated from validation-only setting selection. No best-performing test seed is selected for reporting.

### What do the uncertainty bars cover?

Research bars use approximate Student-t intervals over independent training-seed means, conditional on one shared order set. Per-run CSVs also contain Monte Carlo cost and paired-savings intervals over orders. They address different sources of uncertainty. A non-learning baseline has zero training-seed variation, but still has order-sampling and real-world uncertainty. Five seeds are useful evidence, not a definitive convergence study.

### Does the risk display predict final delivery probability?

No. It is an explicitly defined logistic pressure proxy coupled to signed outlook, distance and load. Final lateness is determined by the accumulated simulated trajectory, without a contradictory independent terminal draw. Calibration against real logistics outcomes is future work.

### Does this prove interventions work in real logistics?

No. Intervention costs and effectiveness are assumed. The project demonstrates a reproducible sequential cost-minimisation method under those assumptions. Real deployment would need calibrated effects, operational constraints and external validation.

### When is intervention economically worthwhile?

When expected avoided delay penalties exceed intervention spending. With per-day penalty `p` and fixed late fee `F`, saving is `p*(E[D_no_action] - E[D_policy]) + F*(P(late_no_action) - P(late_policy)) - E[intervention_cost]`. Positive saving means the policy reduces expected economic cost. Preventing the last late day can avoid the fixed fee too, so the benefit need not be linear in days saved. Use the executed sensitivity curves for evidence under this simulator, not invented example performance.

## Before presenting

1. Open `D:\ARLO Project` and run `run_dashboard.cmd`.
2. Use `results/studies/REPORT.md` for the actual research findings and protocol.
3. Keep `results/studies/metadata.json` available for seeds and selected settings.
4. Use the same simulator seed when repeating an example; do not describe one successful episode as average performance.
5. Show the rule baseline and all five training seeds honestly. Optional smoothing is for readability, never a replacement for held-out costs.
