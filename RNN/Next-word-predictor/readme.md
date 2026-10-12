# 🧠 Next Word Predictor Using LSTM & PyTorch

### 🚀 Predict the Next Word Using Deep Learning

A deep learning project that predicts the next word in a sequence of text using a **Long Short-Term Memory (LSTM)** neural network built with **PyTorch**. The model learns patterns and relationships between words from a text dataset and generates a prediction based on the input provided by the user.

This project demonstrates the practical application of **Natural Language Processing (NLP), Recurrent Neural Networks (RNNs), LSTM architecture, and next-word prediction**.

---

## 📌 Table of Contents

- [✨ Project Overview](#-project-overview)
- [🎯 Objectives](#-objectives)
- [⚙️ How It Works](#️-how-it-works)
- [🧠 Model Architecture](#-model-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [📂 Project Structure](#-project-structure)
- [📋 Prerequisites](#-prerequisites)
- [🚀 Installation and Setup](#-installation-and-setup)
- [💻 How to Use](#-how-to-use)
- [📊 Example](#-example)
- [🔬 Key Concepts Learned](#-key-concepts-learned)
- [🌟 Features](#-features)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [👨‍💻 Author](#-author)
- [📜 License](#-license)

---

## ✨ Project Overview

Next Word Predictor is an NLP application designed to predict the most likely next word based on a given sequence of words.

For example, when the input is:

> Artificial intelligence is

The model attempts to predict a suitable next word based on the patterns it learned during training.

Unlike a rule-based approach, the model learns statistical patterns from training text using a neural network.

The project uses **LSTM (Long Short-Term Memory)**, a specialized type of Recurrent Neural Network designed to learn sequential dependencies and help address the vanishing gradient problem associated with traditional RNNs.

---

## 🎯 Objectives

The main objectives of this project are:

- 🧠 Understand how LSTM networks process sequential data.
- 📝 Perform basic text preprocessing and tokenization.
- 🔢 Convert words into numerical representations.
- 🔗 Prepare sequences for supervised learning.
- 🤖 Train a deep learning model using PyTorch.
- 🎯 Predict the next word from an input sequence.
- 📚 Explore the fundamentals of Natural Language Processing.

---

## ⚙️ How It Works

The project follows a standard NLP pipeline:

**1. 📄 Text Dataset**

The model uses a text file (`data.txt`) as its training dataset.

**2. ✂️ Text Preprocessing**

The text is cleaned and split into individual words or tokens.

**3. 🔢 Tokenization**

Each unique word is assigned a numerical index so that the text can be processed by a neural network.

**4. 🧩 Sequence Generation**

The tokenized text is divided into input sequences and corresponding target words.

For example:

| Input Sequence | Target Word |
|---|---|
| I love | Python |
| I love Python | programming |
| love Python | programming |

*These are illustrative examples of sequence preparation.*

**5. 🧠 LSTM Model Training**

The input sequences are passed through an embedding layer and an LSTM network. The model learns patterns in the text and predicts a probability distribution over the vocabulary.

**6. 🎯 Next Word Prediction**

When a user enters a sequence, the trained model predicts the next word based on the learned patterns.

---

## 🧠 Model Architecture

The model uses a neural network architecture designed for sequential text data.

Depending on the implementation, the pipeline consists of the following components:

| Component | Purpose |
|---|---|
| 🔢 Tokenization | Converts words into integer IDs |
| 🧮 Embedding Layer | Represents words as dense numerical vectors |
| 🔄 LSTM Layer | Learns sequential patterns and dependencies |
| 🔗 Fully Connected Layer | Maps the LSTM output to vocabulary scores |
| 🎯 Softmax / Probability Distribution | Represents the likelihood of possible next words |

### 🔄 Architecture Flow

```text
        Input Text
            ↓
         Tokenizer
            ↓
      Integer Sequences
            ↓
      Embedding Layer
            ↓
         LSTM Layer
            ↓
     Fully Connected Layer
            ↓
      Next Word Prediction
```

**Why LSTM?**

Traditional neural networks do not naturally preserve sequential context. LSTM networks use a cell state and three gates to regulate information flow:

- 🚪 **Forget Gate:** Determines which information to discard.
- 📥 **Input Gate:** Determines which new information to store.
- 📤 **Output Gate:** Determines what information to pass forward.

These mechanisms help the model learn relationships across sequences of text.

---

## 🛠️ Tech Stack

The project uses the following technologies:

- 🐍 **Python** — Core programming language.
- 🔥 **PyTorch** — Deep learning framework used to build and train the model.
- 🧠 **LSTM** — Recurrent neural network architecture for sequence learning.
- 📝 **NLP** — Text preprocessing, tokenization, and sequence modeling.
- 📊 **NumPy** — Numerical operations, if used in the implementation.

---

## 📂 Project Structure

```text
next-word-predictor/
│
├── data.txt             # Text dataset used for training
├── model.py             # LSTM model definition (if separated)
├── train.py             # Model training script (if separated)
├── predict.py           # Next-word prediction script (if separated)
├── main.py              # Main application entry point (if used)
├── requirements.txt     # Python dependencies
├── .gitignore           # Files excluded from Git
└── README.md            # Project documentation
```

**Note:** Adjust the filenames above to match your actual project. If your code is in a single Python file, you do not need to create separate scripts just to match this structure.

---

## 📋 Prerequisites

Before running the project, make sure you have:

- Python 3.10 or a compatible Python version.
- Basic knowledge of Python.
- Fundamental understanding of neural networks and LSTM.
- PyTorch installed in your environment.
- A text dataset for training.

---

## 🚀 Installation and Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/next-word-predictor.git
```

Navigate to the project directory:

```bash
cd next-word-predictor
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

Install PyTorch:

```bash
pip install torch
```

If your project uses NumPy:

```bash
pip install numpy
```

Alternatively, install all dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare the Dataset

Place your text dataset in the project directory and name it:

```text
data.txt
```

Ensure that your training script reads from the correct file path.

### 5️⃣ Train the Model

Run the Python script responsible for training.

For example, if your training code is in `train.py`:

```bash
python train.py
```

If training and prediction are implemented in the same script, execute that script instead.

### 6️⃣ Run the Predictor

```bash
python predict.py
```

Replace the filename with your actual prediction script.

---

## 💻 How to Use

1. Start the prediction script.
2. Enter a word or a short sequence of words.
3. The input is tokenized using the same vocabulary used during training.
4. The trained LSTM model processes the input sequence.
5. The model predicts the most likely next word.
6. Enter another sequence to test different predictions.

**Important:** Use the same preprocessing and vocabulary mapping during training and inference. Unknown words must be handled according to your implementation.

---

## 📊 Example

An illustrative interaction might look like this:

```text
🧠 Next Word Predictor
------------------------------

Enter a sequence: I love

Predicted next word: Python
```

*This is an example of the expected interaction, not a guaranteed prediction from the trained model.*

The actual output depends on the dataset, training process, vocabulary, and model parameters.

---

## 🔬 Key Concepts Learned

This project provides hands-on experience with:

- 📝 **Natural Language Processing:** Preparing text for machine learning.
- 🔢 **Tokenization:** Mapping words to integer IDs.
- 🧩 **Sequence Modeling:** Generating input-target pairs from text.
- 🧠 **LSTM Networks:** Learning sequential dependencies.
- 🔥 **PyTorch:** Defining neural networks and training models.
- 📉 **Loss Functions:** Measuring prediction errors during training.
- ⚙️ **Backpropagation:** Updating model parameters to improve predictions.
- 🎯 **Inference:** Using a trained model to predict the next token.

---

## 🌟 Features

- 🧠 LSTM-based next-word prediction.
- 🔥 Built using PyTorch.
- 📝 Text-based training dataset.
- 🔢 Tokenization and sequence preparation.
- 🔄 Learning from sequential text patterns.
- 🎯 Predicting the next word from user input.
- 📚 Practical introduction to NLP and deep learning.

---

## 🔮 Future Improvements

Potential improvements to make this project more advanced:

- 📚 Train on a larger and more diverse text corpus.
- 🎯 Predict the top 3 or top 5 candidate words.
- 📊 Display prediction probabilities.
- 💾 Save and reload trained model checkpoints.
- 🖥️ Build an interactive interface using Streamlit.
- ✍️ Add temperature-based sampling for more varied predictions.
- 🧹 Improve text cleaning and unknown-word handling.
- 📈 Track training loss and validation performance.
- 🧪 Evaluate prediction accuracy using a held-out dataset.
- 🚀 Compare LSTM performance with GRU and Transformer-based models.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

You can also open an issue to report bugs or suggest new features.

---

## 👨‍💻 Author

**Yash Singhal**

🎓 B.Tech in Electronics and Communication Engineering

💡 Interests: Artificial Intelligence, Machine Learning, Deep Learning, NLP, and Software Development.

🔗 **Connect with me:**

- 💼 LinkedIn: [yash-singhal-gbpiet](https://www.linkedin.com/in/yash-singhal-gbpiet/)
- 🐙 GitHub: [yashsinghal11](https://github.com/yashsinghal11)

---

## 📜 License

This project can be distributed under the MIT License. If you choose MIT, add a `LICENSE` file containing the appropriate license text to the repository.

---

⭐ **If you find this project interesting, consider giving the repository a star!**

💬 *Learning deep learning one project at a time.* 🚀


<img width="1022" height="577" alt="image" src="https://github.com/user-attachments/assets/9a577993-43e2-45ab-9aa9-e1afe03b123a" />
