import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model import NextWordLSTM

torch.manual_seed(42)

# 1. Training text
sentences = [
    "I love machine learning",
    "I love deep learning",
    "I love machine learning",
    "Deep learning uses neural networks",
    "I am learning pytorch",
    "Pytorch is a deep learning framework",
    "Language models predict the next word",
    "I love learning",
    "Machine learning is interesting",
]

# 2. Tokenization
tokenized_sentences = [
    re.findall(r"\b\w+\b", sentence.lower())
    for sentence in sentences
]

# 3. Build vocabulary
words = [
    word
    for sentence in tokenized_sentences
    for word in sentence
]

vocab = sorted(set(words))

# Reserve 0 for padding
word_to_idx = {
    word: idx + 1
    for idx, word in enumerate(vocab)
}

idx_to_word = {
    idx: word
    for word, idx in word_to_idx.items()
}

vocab_size = len(vocab) + 1

print("Vocabulary size:", len(vocab))
print("Word IDs:", word_to_idx)

# 4. Create input sequences and target words
X = []
y = []

for sentence in tokenized_sentences:
    for i in range(1, len(sentence)):
        sequence = sentence[:i]
        target = sentence[i]

        input_ids = [
            word_to_idx[word]
            for word in sequence
        ]

        target_id = word_to_idx[target]

        X.append(
            torch.tensor(input_ids, dtype=torch.long)
        )

        y.append(target_id)

# 5. Padding
lengths = torch.tensor(
    [len(sequence) for sequence in X],
    dtype=torch.long
)

X_padded = pad_sequence(
    X,
    batch_first=True,
    padding_value=0
)

y_tensor = torch.tensor(y, dtype=torch.long)

print("Input shape:", X_padded.shape)
print("Target shape:", y_tensor.shape)

# 6. Create LSTM model
embedding_dim = 32
hidden_dim = 64

model = NextWordLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# 7. Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# 8. Training
epochs = 200

for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()

    predictions = model(X_padded, lengths)

    loss = criterion(predictions, y_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )

# 9. Prediction function
def predict_next_word(sentence):

    model.eval()

    input_words = re.findall(
        r"\b\w+\b",
        sentence.lower()
    )

    if not input_words:
        return "Please enter a valid sentence."

    # Handle words not present in training vocabulary
    unknown_words = [
        word for word in input_words
        if word not in word_to_idx
    ]

    if unknown_words:
        return (
            "Unknown words: "
            + ", ".join(unknown_words)
            + ". Try words from the training vocabulary."
        )

    input_ids = [
        word_to_idx[word]
        for word in input_words
    ]

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long
    )

    input_length = torch.tensor(
        [len(input_ids)],
        dtype=torch.long
    )

    with torch.no_grad():

        predictions = model(
            input_tensor,
            input_length
        )

        predicted_id = predictions.argmax(dim=1).item()

    return idx_to_word[predicted_id]


# 10. Test the model
while True:

    sentence = input(
        "\nEnter a sentence (or type 'exit'): "
    )

    if sentence.lower().strip() == "exit":
        break

    next_word = predict_next_word(sentence)

    print("Predicted next word:", next_word)