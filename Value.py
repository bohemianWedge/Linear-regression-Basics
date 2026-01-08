import math

class Value:
        def __init__(self, val, _children=(), op='', label=''):
                self.val = val
                self._backward = lambda : None
                self.grad = 0
                self.label = label
                self.op = op
                self.children = _children
                self._prev = set(_children)

        def __repr__(self):
               # return f"{self.label} = {self.children[0].label} {self.op} {self.children[1].label} = {self.children[0].val} {self.op} {self.children[1].val} = {self.val}"
               return f"Value(val={self.val})"

        def __add__(self, other):
                if not isinstance(other, Value) : other = Value(other)
                out =  Value(self.val + other.val, (self, other), '+')
                def _backward():
                     self.grad += 1.0 * out.grad
                     other.grad += 1.0 * out.grad
                out._backward = _backward
                return out
        
        def __radd__(self, other):
                return self + other
        
        def __pow__(self, other):
             assert isinstance(other, (int, float))
             out = Value(self.val ** other, (self,), f'**{other}')
             def _backward():
                  self.grad += other * self.val**(other - 1) * out.grad
             out._backward = _backward
             return out
        
        def __truediv__(self, other):
             return (self * (other ** -1))
        
        def __mul__(self, other):
                other = other if isinstance(other, Value) else Value(other)
                out = Value(self.val * other.val, (self, other), '*')
                def _backward():
                        self.grad += other.val * out.grad
                        other.grad += self.val * out.grad
                out._backward = _backward
                return out
                
        def __rmul__(self, other):
                return self * other
        
        def __neg__(self):
             return self * -1
        
        def __sub__(self, other):
             return self + (-other)
        
        def __rsub__(self, other):
             return other + (-self)
        
        def exp(self):
             out = Value(math.exp(self.val), (self, ), 'exp')
             def _backward():
                  self.grad += out.val * out.grad
             out._backward = _backward
             return out
        
        # Fonciton d'activation : y = (e^(2x) - 1) / (e^(2x) + 1)
        def tanh(self):
             x = self.val
             t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
             out = Value(t, (self, ), 'tanh')

             def _backward():
                self.grad += (1 - t**2) * out.grad
             out._backward = _backward
             return out
          
        # Fonciton d'activation : y = 1 / (1 + e^(-x)) --> : y' = y * (1 - y) 
        def sigmoid(self):
             x = (1 + math.exp(-1 * self.val))**-1
             out = Value(x, (self, ), 'sigmoid')
             def _backward():
                  self.grad += out.val * (1 - out.val) * out.grad
             out._backward = _backward
             return out
        
        # Fonciton d'activation : y = max(0, x)
        def relu(self):
             out = Value(max(0, self.val), (self, ), 'ReLu')
             def _backward():
                  self.grad += (out.val > 0) * out.grad
             out._backward = _backward
             return out
        
        def backward(self, zero_grad=True):
             # Tri topologique d'un graphe acyclique
             topo = []
             visited = set()
             def build_topo(v):
                if v not in visited:
                        visited.add(v)
                        for child in v._prev:
                                build_topo(child)
                        topo.append(v)
             build_topo(self)

             if zero_grad:
                  for node in topo:
                       node.grad = 0.0

             self.grad = 1.0
             for node in reversed(topo):
                  node._backward()
