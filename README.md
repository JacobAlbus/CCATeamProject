# CCATeamProject

This project attempts to replicate the results obtained by [Counterfactual Credit Assignment in Model-Free Reinforcement Learning
](https://arxiv.org/pdf/2011.09464.pdf)

### Bandit Example

1. `bandit.py` defines the bandit environment and can be treated similarly to the openai gym environment: just instantiate, init, then step.
2. `train_benchmark.py` contains code for solving the bandit problem using VFA + Reinforce algorithm with value function baseline.
3. `train_cca.py` contains code for solving the bandit problem using VFA + Reinforce algorithm with CCA baseline from the paper.
4. `gru.py` contains PyTorch code for a GRU RNN.
