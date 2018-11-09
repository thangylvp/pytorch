from __future__ import print_function
import torch
import copy

x = torch.tensor([5, 3])
# print(x)

y = x
print(y)

x.add_(1)
print(x)
print(y)


y = copy.deepcopy(x)
x.add_(1)
print(x)
print(y)

a = torch.ones(5)
print(a, type(a))

b = a.numpy()
print(b, type(b))


a = torch.ones(2, 2, requires_grad = True)

print(a)

b = a + 2
print(hex(id(a)))
print(b)
print(b.grad_fn)
print('de: ',a.grad)

out = b.mean()

print(out, type(out), out.grad_fn)

out.backward()

print('de: ',a.grad)