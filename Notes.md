# Notes

## Contribution Ideas

1. Time-Frequency Decoding Analysis:

- The original study uses raw EEG activation values for decoding. A valuable addition would be to perform decoding in the frequency domain.
  - Why: Different cognitive processes are reflected in specific frequency bands (e.g., Theta for memory/control, Alpha for attention, Beta for motor preparation).
  - How: Apply a wavelet transform or Multitaper analysis to the EEG data and use the power spectra as features for the classifier. You could then see which frequency bands carry the most information about the participant's decision or the previous trial's outcome.

2. Temporal Generalization (Cross-Temporal Decoding):

- The paper displays decoding accuracy over time but does not appear to show a Temporal Generalization Matrix (TGM).
  - Why: A TGM reveals whether the neural representation of a "decision" (e.g., choosing "Rock") is stable (the same pattern persists over time) or dynamic (the pattern evolves through different stages).
  - How: Train the classifier at time point $t$ and test it on all other time points $t'$. This would provide insight into the "life cycle" of a competitive decision in the brain.

3. Computational Strategy Modeling (RL Analysis):

- The authors noted that participants exhibited behavioral biases and were not perfectly random. You could add a Model-Based EEG Analysis.
  - Why: Instead of just decoding the "move," you could decode the internal strategy.
  - How: Fit a Reinforcement Learning (RL) model (like Win-Stay-Lose-Shift) or a Markov chain to the behavioral data found in the OSF repository. Use the model's trial-by-trial "prediction" or "surprise" (prediction error) as a regressor for the EEG signal to see where and when the brain computes these specific strategies.

4. Inter-Brain Representational Similarity Analysis (RSA):

- The paper mentions that they focused on within-brain representations because the number of response options (3) was too low for traditional inter-brain RSA. You could overcome this by using a Multi-Trial RSA.
  - Why: To see if "Information Alignment" occurs between winners and losers.
  - How: Create Representational Dissimilarity Matrices (RDMs) that include not just the move (Rock/Paper/Scissors) but also the outcome (Win/Loss/Draw) and the trial history. Comparing these RDMs between the two players would reveal if their "mental maps" of the game become synchronized over the course of the 480 games.

5. Source Localization of Decoding Topographies:

- The paper shows sensor-level decoding maps. Adding Source Reconstruction would strengthen the anatomical claims.
  - Why: To determine if the "loser's memory" of prior trials is localized in the hippocampus/prefrontal cortex or if it is a more general motor-planning bias.
  - How: Use tools like eLORETA or Beamforming (available in FieldTrip or MNE-Python) to project the EEG data into source space before running the decoding analysis.

6. Micro-Strategy Analysis (Win-Stay, Lose-Shift)

- The authors mention behavioural biases like the 'win-stay, lose-shift' strategy but perform a general decoding of the "current response".
  - The New Analysis: Perform a conditional decoding. Instead of decoding "Rock vs. Paper," decode the "Strategy" (e.g., did they just stay with their last move or shift?).
  - Why it's better: This specifically targets the "cognitive biases" mentioned in the abstract  and might reveal a much stronger neural signal than raw move decoding.

7. Markov Chain Correlation
- The authors used a Markov chain to measure how predictable a player's behavior was.
  - The New Analysis: Correlate the Markov Predictability Score (the behavior) with the Decoding Accuracy (the brain).
  - Class Project Angle: Does a more "predictable" player (high Markov score) also have a more "decodable" brain? This links the behavioral results in Figure 1 directly to the neural results in Figure 2, which the paper keeps somewhat separate.

## Felipe's Notes

- To test whether the EEG data tells us something about the decision-making process in a competitive setting, the researchers picked the game Rock-Paper-Scisors;
- In this game, the optimal strategy is to be completly random;
- In theory, if a player is not playing randomly, they must have some bias in their decision-making process/knowledge representation;
  - For example: remembering what the opponent played in the last few rounds, and trying to predict based on that;
- Question: can we "see" this "bias" in the player's EEG data? if we can "see" it, can we also predict the player's decision based on it?
- To answer this question, the researchers picked 4 metrics:
  1. Predict the player's decision;
  2. Predict the opponent's decision;
  3. Predict the player's decision in the previous round;
  4. Predict the opponent's decision in the previous round;
- Experiment goal: given the raw eeg data for a participant for a single game (Decision (2s), Response (2s), and Feedback (1s)), how accuratelly are we based on the 4 metrics?
  1. If the prediction accuracy always hovers around 33%, then the EEG data contains no information about the decision-making process;
  2. If the prediction accuracy does not always hover around 33%, then the EEG data contains some information about the decision-making process;

My own (Felipe) understanding of expected results:

- If the accuracy for the previous trial (metrics 3 and 4) is higher than 33%, then the player's strategy is not complete random --> the player is still thinking about the previous trial --> modeling previous trial to make current decision;
- If the accuracy for the player's decision in the current trial (metric 1) is higher than 33%, then the EEG data contains some information --> if it was around 33%, I think all the other metrics would become irrelevant (luck-based predictions)
- If the accuracy for the opponent's decision in the current tiral (metric 2) is higher than 33%, then the player can successfully predict the opponent's decision --> opponent is not random AND the player is ALSO not random --> both players are playing suboptimal strategies;
- if metrics 2, 3, 4 are all around 33% but metric 1 is above that, then the player is playing truly random;
