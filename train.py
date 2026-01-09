import pandas as pd
import json
from Value import Value
import matplotlib.pyplot as plt

class linearReg:
        def __init__(self,learn_rate = 0.5 , target=0.1, max_iter=10000,a=Value(0), b=Value(0)):
                self.a = a
                self.b = b
                self.db = pd.read_csv('data.csv')

                # Sauvegarder les stats pour dénormaliser après
                self.km_mean = self.db['km'].mean()
                self.km_std = self.db['km'].std()
                self.price_mean = self.db['price'].mean()
                self.price_std = self.db['price'].std()

                self.kms = [Value((km - self.km_mean) / self.km_std) for km in self.db['km']]
                self.prices = [Value((p - self.price_mean) / self.price_std) for p in self.db['price']]

                self.lr = learn_rate
                self.target = target
                self.max_iter = max_iter

        def compute_model(self):
                out = [self.a * km + self.b for km in self.kms]
                return out

        def compute_cost(self, prediction):
                m = len(prediction)
                total_cost = Value(0)
                for pred, y in zip(prediction, self.prices):
                        error = pred - y
                        total_cost = total_cost + error**2  # carré, pas racine
                cost = total_cost / (2 * m)  # division après la somme
                return cost
        
        def plot_cost_history(self, costs):
                
                plt.figure(figsize=(10, 5))
                plt.plot(costs, color='blue', linewidth=1)
                plt.xlabel('Tteration')
                plt.ylabel('Cost (MSE)')
                plt.title('Cost evolution during training')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')  # Échelle log pour mieux voir la convergence
                # plt.savefig('graph/cost_history.png', dpi=150)
                plt.show()

        def fit(self):
                costs = []
                all_a = []
                all_b = []
                for i in range(self.max_iter):
                        act_y = self.compute_model()
                        cost = self.compute_cost(act_y)
                        costs.append(cost.val)
                        cost.backward()

                        self.a.val = self.a.val - self.lr * self.a.grad
                        self.b.val = self.b.val - self.lr * self.b.grad

                        all_a.append(self.a.val)
                        all_b.append(self.b.val)
                        
                        if i % 100 == 0: 
                                print(i, cost.val, self.a.grad, self.b.grad)

                        if cost.val < self.target:
                                break

                        if abs(self.a.grad) < 1e-9 and abs(self.b.grad) < 1e-9:
                                print(f"Convergence atteinte à l'itération {i}")
                                break
                self.save_training_history(all_a, all_b, costs)
                return all_a, all_b, costs

        def denormalize(self):
                a_real = self.a.val * (self.price_std / self.km_std)
                b_real = self.price_mean + self.price_std * self.b.val - a_real * self.km_mean
                return a_real, b_real

        def save_model(self, filepath='model.json'):
                a_real, b_real = self.denormalize()
                with open(filepath, 'w') as f:
                        json.dump({'theta0':b_real, 'theta1':a_real}, f)
                print(f"Params saved as {filepath}")

        def save_training_history(self, all_a, all_b, costs):
                all_a = [(a * self.price_std) / self.km_std for a in all_a]
                all_b = [
                self.price_mean + self.price_std * b - all_a[i] * self.km_mean 
                for i, b in enumerate(all_b)
                ]

                costs = [cost * (self.price_std ** 2) for cost in costs]
                history = {
                        'iterations': list(range(len(all_a))),
                        'a_values': all_a,
                        'b_values': all_b,
                        'costs': costs,
                }
                with open('training_history.json', 'w') as f:
                        json.dump(history, f)
                print("History saved as training_history.json")

if __name__ == '__main__':
        model = linearReg(learn_rate=0.1, target=0.1, max_iter=100)
        model.fit()
        model.save_model()
        
        