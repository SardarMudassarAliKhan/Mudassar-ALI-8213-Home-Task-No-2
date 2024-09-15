import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2, 1, bias=True)
        self.fc.weight.data = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        self.fc.bias.data = torch.tensor([0.0], dtype=torch.float32)
    
    def forward(self, x):
        return self.fc(x)

x_data = torch.tensor([[-2, 4], [7, -2]], dtype=torch.float32)
y_data = torch.tensor([5, -3], dtype=torch.float32).view(-1, 1)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 6
loss_values_auto = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()
    optimizer.step()
    
    loss_values_auto.append(loss.item())
    
    print(f"Epoch {epoch + 1}")
    print("Updated weights:", model.fc.weight.data)
    print("Updated bias:", model.fc.bias.data)
    print("Loss:", loss.item())
    print("----------------------")

print("Final weights:", model.fc.weight.data)
print("Final bias:", model.fc.bias.data)

def manual_update(weights, lr, x_data, y_data):
    w1, w2 = weights
    y_pred = x_data @ torch.tensor([w1, w2], dtype=torch.float32)
    loss = torch.mean((y_pred - y_data) ** 2).item()
    
    grad_w1 = 2 * torch.mean((y_pred - y_data) * x_data[:, 0]).item()
    grad_w2 = 2 * torch.mean((y_pred - y_data) * x_data[:, 1]).item()
    
    w1_new = w1 - lr * grad_w1
    w2_new = w2 - lr * grad_w2
    
    return w1_new, w2_new, loss

w1 = 0.1
w2 = 0.2
lr = 0.1

loss_values_manual = []

print("Manual Calculation Results:")
for epoch in range(1, epochs + 1):
    w1, w2, loss = manual_update((w1, w2), lr, x_data, y_data)
    loss_values_manual.append(loss)
    
    print(f"Epoch {epoch}")
    print(f"Weight 1: {w1:.4f}")
    print(f"Weight 2: {w2:.4f}")
    print(f"Loss: {loss:.4f}")
    print("----------------------")

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), loss_values_auto, label='Automatic Update (PyTorch)', marker='o')
plt.plot(range(1, epochs + 1), loss_values_manual, label='Manual Update', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison Between Manual and Automatic Update')
plt.legend()
plt.grid(True)
plt.show()
