# 🔥 2-2-1 Artificial Neural Network (ANN) with PyTorch for Heat Transfer Prediction

This repository provides a clean and well-documented implementation of a **2-2-1 Feedforward Artificial Neural Network** using **PyTorch**. 
It supports both **incremental (online)** and **batch training**, live **error visualization**, and flexible **error function selection** (`MSE`, `MAE`, or `BCE`). 
It's primarily used here for a **heat transfer prediction problem**, but the model is general enough to be adapted for other use cases.

---

## 📌 Features

* 🔢 2 Input → 2 Hidden → 1 Output architecture
* ✅ Supports MSE, MAE, and BCE loss functions
* 🔄 Incremental (sample-by-sample) and batch training modes
* 📉 Real-time error convergence visualization with Matplotlib
* 💾 Save and load model with learned weights and normalizers
* 📊 Built-in input/output normalization and denormalization
* 🧪 Simple test suite for evaluating model performance

---

## 🧠 Network Architecture

```
Input Layer (2 nodes)
      ↓
Hidden Layer (2 nodes, ReLU activation)
      ↓
Output Layer (1 node, Sigmoid activation)
```

---

## ⚙️ Setup

### Requirements

Install dependencies with:

```bash
pip install torch numpy matplotlib
```

---

## 🚀 Usage

### 1. Generate Heat Transfer Dataset

```python
inputs, outputs = generate_heat_transfer_data(n_samples=1000)
```

### 2. Normalize Data

```python
ann = ANN_2_2_1(learning_rate=0.01)
ann.input_normalizer.fit(inputs)
ann.output_normalizer.fit(outputs)

inputs_norm = ann.input_normalizer.normalize(inputs)
outputs_norm = ann.output_normalizer.normalize(outputs)
```

### 3. Train the Model

**Incremental Training:**

```python
ann.set_error_function('mse')
ann.train_incremental(inputs_norm, outputs_norm)
```

**Batch Training:**

```python
ann.train_batch(inputs_norm, outputs_norm)
```

### 4. Test the Model

```python
predictions, error = ann.test(inputs_norm, outputs_norm)
print("Test Error:", error)
```

### 5. Save/Load Model

```python
ann.save_model("ann_model.pt")
ann.load_model("ann_model.pt")
```

---

## 🧪 Heat Transfer Function

The dataset is generated using a simple nonlinear heat transfer formula:

```
final_temp = initial_temp + 0.5 * initial_temp * sin(0.1 * duration) + 0.3 * duration
```

---

## 📈 Visualization

Training error is visualized live using Matplotlib's interactive mode. This helps observe convergence trends and fine-tune learning rates or epoch counts effectively.

---

## 🗃 Project Structure

```
├── ann.py                    # Main ANN class and Normalizer
├── heat_transfer_data.py     # Data generation function
├── README.md                 # Project documentation
└── ann_model.pt              # Saved model (after training)
```

---

## 💡 Future Work

* Add support for custom activation functions
* Extend to n-n-m configurable networks
* Export training logs for advanced analysis
* Integrate with TensorBoard for rich visualization

---

## 👨‍💻 Author

**Shihab Mahmud Dhrobo**
MSc Student in Industrial Systems Analytics
University of Vaasa, Finland

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


