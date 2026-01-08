import json
import pandas as pd
import matplotlib.pyplot as plt


def predict():
        try:
                with open('model.json', 'r') as f:
                        params = json.load(f)
        except FileNotFoundError:
                params = {'theta0': 0, 'theta1': 0}
        
        km = float(input("Kilométrage : "))
        price = params['theta1'] * km + params['theta0']
        print(f"Prix estimé : {price:.2f} €")

if __name__ == '__main__':
        predict()