import pandas as pd
import math
import json
from Value import Value

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
                """Affiche l'évolution du coût pendant l'entraînement."""
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 5))
                plt.plot(costs, color='blue', linewidth=1)
                plt.xlabel('Itération')
                plt.ylabel('Coût (MSE)')
                plt.title('Évolution du coût pendant l\'entraînement')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')  # Échelle log pour mieux voir la convergence
                plt.savefig('cost_history.png', dpi=150)
                plt.show()

        def fit(self):
                costs = []
                for i in range(self.max_iter):
                        act_y = self.compute_model()
                        cost = self.compute_cost(act_y)
                        costs.append(cost.val)
                        cost.backward()

                        self.a.val = self.a.val - self.lr * self.a.grad
                        self.b.val = self.b.val - self.lr * self.b.grad

                        if i % 100 == 0: 
                                print(i, cost.val, self.a.grad, self.b.grad)

                        if cost.val < self.target:
                                break

                        if abs(self.a.grad) < 1e-11 and abs(self.b.grad) < 1e-11:
                                print(f"Convergence atteinte à l'itération {i}")
                                break
                self.plot_cost_history(costs)

        def denormalize(self):
                """Convertit les paramètres normalisés en paramètres réels"""
                # Si y_norm = a_norm * x_norm + b_norm
                # Alors y = a_real * x + b_real
                # Avec: a_real = a_norm * (price_std / km_std)
                #       b_real = price_mean + price_std * (b_norm - a_norm * km_mean / km_std)
                a_real = self.a.val * (self.price_std / self.km_std)
                b_real = self.price_mean + self.price_std * self.b.val - a_real * self.km_mean
                return a_real, b_real

        def save_model(self, filepath='model.json'):
                a_real, b_real = self.denormalize()
                with open(filepath, 'w') as f:
                        json.dump({'theta0':b_real, 'theta1':a_real}, f)
                print(f"Parametres sauvegardés dans {filepath}")

if __name__ == '__main__':
        model = linearReg(learn_rate=0.1, target=0.001, max_iter=10000)
        model.fit()
        model.save_model()
        
        