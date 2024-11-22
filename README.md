# CITS3001

**CITS3001 Project: Mario Bros AI Agent Analysis**

## Overview

This project explores the implementation and comparison of two AI approaches, **Deep Q-Network (DQN)** and **Rule-Based AI**, for playing the classic *Super Mario Bros* game. The primary goal is to analyze the performance, strengths, and limitations of these two methods across different levels of the game.

## Features

- **Deep Q-Network (DQN)**:
  - Uses a Convolutional Neural Network (CNN) for feature extraction.
  - Requires extensive training (~3200 iterations for "World 1-1").
  - Focuses on coin collection and defeating enemies to maximize scores.

- **Rule-Based AI**:
  - Operates without training.
  - Completes levels by following predefined accuracy rules.
  - Prioritizes moving forward while avoiding obstacles.

## Results Summary

| Metric                  | DQN Agent       | Rule-Based Agent |
|-------------------------|-----------------|------------------|
| Training Time (1 Level) | ~12 hours (GPU) | No training      |
| Score (*World 1-1*)     | 2900            | 1500            |
| Completion Speed        | Slower          | Faster          |

## Skills & Tools

- **Programming Languages**: Python
- **Libraries**: PyTorch, NumPy, Gym-Super-Mario-Bros

## Reference
The Baseline used for this DQN implementation is from https://github.com/xiaoyoubilibili/gym_super_mario.


