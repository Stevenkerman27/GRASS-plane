import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
input = torch.randn(1, 3)
target = torch.tensor(1)

print('Input shape:', input.shape)
print('Target shape:', target.shape)

try:
    loss = criterion(input, target)
    print('Loss with scalar target:', loss)
except Exception as e:
    print('Error with scalar target:', e)

target_unsqueezed = target.unsqueeze(0)
print('Target unsqueezed shape:', target_unsqueezed.shape)
try:
    loss = criterion(input, target_unsqueezed)
    print('Loss with unsqueezed target:', loss)
    # Check torch.stack on scalars
    losses = [loss, loss]
    stack_loss = torch.stack(losses, 0)
    print('Stack loss shape:', stack_loss.shape)
    print('Stack loss:', stack_loss)
    # Check data[0] on scalar
    try:
        print('Scalar data[0]:', loss.data[0])
    except Exception as e:
        print('Error on scalar.data[0]:', e)
except Exception as e:
    print('Error during cat/loss:', e)
