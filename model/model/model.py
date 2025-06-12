
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Dane
df = pd.read_csv('data.csv')
X_raw, y = df['text'].tolist(), df['label'].values

# Tokenizacja (bag-of-words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_raw).toarray()
X = torch.tensor(X).float()
y = torch.tensor(y).long()

# Prosty model
class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.fc(x)

model = SimpleNet(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Trening
for epoch in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item()}")

# Inferencja
test = ["AI jest super", "Nie cierpię spamu"]
X_test = vectorizer.transform(test).toarray()
X_test = torch.tensor(X_test).float()
probs = model(X_test).softmax(dim=1)
print("Prawdopodobieństwa:", probs)
print("Predykcje:", probs.argmax(dim=1))
