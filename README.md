# 🧠 Vacuum Cleaner Agent: Reflex, Goal-Based, and Utility-Based

This project demonstrates how a **simple vacuum cleaner agent** can behave under three types of AI models: **Reflex Agent**, **Goal-Based Agent**, and **Utility-Based Agent**.

Each version of the agent mimics how an intelligent system can work — from simply reacting to its environment to making smart, cost-effective decisions.

---

## ⚡ 1. Reflex Agent

The **Reflex Vacuum Agent** acts based purely on the **current perception**. It doesn’t remember anything and doesn’t plan ahead.

### 🔍 How it works:
- Checks current square (A or B).
- If the square is dirty → it **cleans**.
- If clean → it **moves** to the other square.
- No memory. No learning. Just reaction.

📌 Like how the **AND gate** directly gives output based on `[1,1]`, this vacuum agent checks `IsDirty?` and responds instantly.

### 🧾 Example Logic:
```python
if current_square == "A" and is_dirty:
    clean()
elif current_square == "A":
    move_right()

🧠 Vacuum Cleaner Agent: Goal-Based & Utility-Based Models

This project explores two intelligent versions of a vacuum cleaner agent:

1. **Goal-Based Agent**: Acts to achieve a defined goal — cleaning the entire environment.
2. **Utility-Based Agent**: Tries to achieve that goal in the most efficient way possible.

These agents demonstrate how AI can evolve from basic decision-making to more intelligent, optimized behaviors — just like your machine learning models improve from simple prediction to minimizing error effectively.

---

## 🎯 1. Goal-Based Agent

The **Goal-Based Vacuum Agent** doesn’t just react — it works towards a **clear goal**: **cleaning all rooms** (like A and B).

### 🧠 How it Works:
- Has a basic model of the world (knows about both rooms).
- Constantly checks if all rooms are clean.
- Continues to act until the **goal is fully achieved**.

📌 *Analogy:* Like your **training loop** that keeps updating weights until all outputs are correct, this vacuum keeps running until everything is clean.

### 🧾 Example Logic (Python):
```python
if A_dirty or B_dirty:
    plan_next_action_to_clean()
else:
    goal_achieved()

# 🤖 Utility-Based Vacuum Cleaner Agent

This is an implementation of a **Utility-Based Vacuum Agent**, an intelligent system that not only aims to **clean a room** but also does it in the **most efficient way possible** — saving time, energy, and unnecessary movement.

It behaves like a smart cleaner with a goal, a brain, and a strategy — just like how your ML model doesn’t stop after predicting, but tries to **minimize the error (MSE)** and perform better.

---

## 💡 What is a Utility-Based Agent?

A Utility-Based Agent doesn’t just act to complete a task — it **measures the usefulness of its actions** and picks the one with the **highest score** (or lowest cost).

In this case:
- The goal is to clean all rooms.
- But instead of doing it blindly, the agent:
  - Prioritizes **dirty rooms**.
  - Considers **battery/time cost**.
  - Chooses the **most optimal path** to clean.

📌 It’s like:
> A function-approximating ML model minimizing **MSE** — not just getting close, but **getting better efficiently**.

---

## ⚙️ How the Vacuum Agent Works

### 1. **Perceives** the environment:
- Sees if a room (A or B) is dirty or clean.

### 2. **Evaluates** each action:
- Cleaning dirty rooms scores high.
- Moving unnecessarily costs energy.

### 3. **Chooses** the best action:
- It picks the one that gives **maximum utility** (high score, low cost).

---

## 🧾 Example (Simplified Python Logic)

```python
def utility(state):
    score = 0
    if state["current"] == "dirty":
        score += 10  # cleaning gives positive score
    if state["battery"] < 20:
        score -= 5   # penalize low battery
    return score - state["energy_cost"]

# Agent chooses action with highest utility
best_action = max(possible_actions, key=utility)

# 🤝 K-Nearest Neighbors (KNN) Classifier

This project is a simple and intuitive implementation of the **K-Nearest Neighbors (KNN)** algorithm — one of the most popular and easy-to-understand machine learning methods.

> 📌 Think of KNN like this:
> If you're trying to decide your style, you might look at your 3 most fashionable friends — and copy what the majority of them wear.
> That’s exactly what KNN does — it checks the "closest friends" (neighbors) and lets them vote.

---

## 📚 What is KNN?

**KNN** is a **supervised learning algorithm** used for **classification** and **regression**. It:
1. Stores all training data.
2. Calculates the distance from the test point to all training points.
3. Picks the **K nearest points**.
4. Assigns the **most common label** (for classification) or **average value** (for regression).

---

## ⚙️ How It Works (Classification):

1. Choose value of **K** (number of neighbors to consider).
2. Compute **Euclidean distance** between the test point and all training data.
3. Select the **K points with the smallest distance**.
4. Predict the label based on **majority voting**.

---

## 🧠 Example (Pseudocode)

```python
for test_point in test_data:
    distances = []
    for train_point in train_data:
        distance = euclidean_distance(test_point, train_point)
        distances.append((distance, label))

    # Sort by distance and take top K
    top_k = sorted(distances)[:K]

    # Predict the most common label among the top_k
    prediction = most_common_label(top_k)

# 🧠 Artificial Neural Network (ANN) from Scratch

This project implements an **Artificial Neural Network (ANN)** — inspired by the way the human brain works — to learn and make predictions from data.

> 📌 Think of an ANN like a virtual brain:
> It takes input (like your senses), processes it (like your neurons), and makes decisions (like your brain)!

---

## 💡 What is an ANN?

An **Artificial Neural Network** is a computational model made up of layers of interconnected "neurons" that can learn complex patterns in data. It’s the foundation for **deep learning**.

---

## 🔁 How It Works

1. **Input Layer**: Takes in features (X).
2. **Hidden Layers**: Perform weighted sums and pass through activation functions.
3. **Output Layer**: Gives final prediction.
4. **Backpropagation**: Adjusts weights by calculating error and minimizing it (usually via gradient descent).

---

## ⚙️ Key Concepts Used

| Concept              | Description                                                  |
|----------------------|--------------------------------------------------------------|
| 🧮 Forward Propagation | Calculate outputs from inputs using weights and activations |
| 🔁 Backpropagation     | Learn from mistakes by adjusting weights                    |
| 🔢 Activation Function | Adds non-linearity (like sigmoid, ReLU, etc.)               |
| 🎯 Loss Function       | Measures how wrong the predictions are                      |
| 🧠 Learning Rate       | Controls how fast the model updates                         |

---

## ✍️ Sample Pseudocode

```python
# Forward pass
hidden_input = dot(X, weights_input_hidden)
hidden_output = sigmoid(hidden_input)

final_input = dot(hidden_output, weights_hidden_output)
final_output = sigmoid(final_input)

# Backward pass (error correction)
error = y - final_output
adjustment = error * sigmoid_derivative(final_output)
weights_hidden_output += dot(hidden_output.T, adjustment)
