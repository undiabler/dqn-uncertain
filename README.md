# DQN-uncertain

DQN experiment with uncertain reward spaces to explore.
The idea is to convert Q values into probabilities and extract risk management from predicted actions.
Also, there are some extra tweaks inherited from probability spaces such as more smooth loss function.

The goals of the experiment is to give NN ability to control self exploration (by predicting Sigma) in environments where agent can't reproduce "win" strategy but need to balance between too risky and too conservative actions.

_100 episodes experience for both agents_
![Agent playing Pong](/videos/dqn.webp?raw=true)    | ![Agent playing Breakout](/videos/dqn-un.webp?raw=true)
----------------------------------------------------|---------------------------------------------------------
 DQN (Simple)                                       | DQN-uncertain (Probability)
