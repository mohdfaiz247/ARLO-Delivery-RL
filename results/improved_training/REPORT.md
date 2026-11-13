# Extended RL training: executed results

The simulator, costs, reward, order distribution and baseline rules are unchanged. The comparison explores 200,000-episode training, lower final exploration and an optional per-action visit-decay learning rate.

Pilot: two independent seeds [11, 22] per setting and algorithm, evaluated on 2,000 validation orders at seed 82026. Selection uses validation means only. Final evaluation freezes five policies per algorithm and uses 5,000 new matched orders at seed 192026.

## Validation selection

- q_learning, constant_200k: mean validation cost R$55.2880.
- q_learning, visit_decay_200k: mean validation cost R$53.9335.
- sarsa, constant_200k: mean validation cost R$55.3220.
- sarsa, visit_decay_200k: mean validation cost R$54.0495.

Selected configurations: {'q_learning': 'visit_decay_200k', 'sarsa': 'visit_decay_200k'}.

## Fresh final results

- q_learning: mean total cost R$53.2962, on-time 39.58%, savings versus no intervention R$4.7488/order. Approximate 95% interval across training-seed means: ±R$0.1120.
- sarsa: mean total cost R$53.3732, on-time 39.98%, savings versus no intervention R$4.6718/order. Approximate 95% interval across training-seed means: ±R$0.0568.
- No intervention: mean total cost R$58.0450, on-time 20.42%, savings versus no intervention R$0.0000/order. Approximate 95% interval across training-seed means: ±R$0.0000.
- Rule based: mean total cost R$53.4522, on-time 40.74%, savings versus no intervention R$4.5928/order. Approximate 95% interval across training-seed means: ±R$0.0000.

Lowest observed mean cost: **q_learning**, R$53.2962/order.

The intervals above quantify training-seed variability conditional on the same orders. Non-learning baselines have zero training variation, but nonzero order-sampling uncertainty. Per-run paired cost-gap intervals versus the rule baseline are in final_runs.csv. Neither a small mean difference nor these seed intervals alone proves universal superiority.

## Before and after on identical orders

- q_learning: original 60k policies cost R$54.9777; new selected 200k policies cost R$53.2962. Mean saving from the training change: R$1.6816/order.
- sarsa: original 60k policies cost R$55.1792; new selected 200k policies cost R$53.3732. Mean saving from the training change: R$1.8061/order.

## Scope and limitations

The optional step size is alpha_n(s,a) = alpha_0 / n(s,a)^p, with an exact per-pair visit count. The Q-learning and SARSA Bellman targets remain distinct and unchanged. Older saved models use p=0 by default and still load. The extended study also lowers final epsilon to 0.01, so any improvement compares a training package rather than isolating one causal hyperparameter. The two-seed pilot is a limited setting-selection sample. Real-world intervention effects remain synthetic and need calibration.

Earlier experiment results are preserved. Absolute costs from different test sets are not directly comparable. The before/after file evaluates old and new policies on identical fresh orders. No baseline has been weakened and no cost result has been replaced by an invented value.
