import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import MyModel

EPOCHS = 20
BATCH_SIZE = 8
LR = 0.001

dataset = CustomDataset('../data/train.csv')
input_dim = dataset.X.shape[1]
num_classes = len(set(dataset.y))

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = MyModel(input_dim, num_classes)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), '../model.pth')
print("Model saved as model.pth")
