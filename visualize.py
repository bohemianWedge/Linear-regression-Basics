import pandas as pd
import matplotlib.pyplot as plt
import json

def load_model(filepath='model.json'):
        """Charge les paramètres du modèle."""
        try:
                with open(filepath, 'r') as f:
                        params = json.load(f)
                return params['theta1'], params['theta0']
        except FileNotFoundError:
                print("⚠ Modèle non trouvé. Lancez d'abord train.py")
                return 0, 0

def plot_data_and_regression():
        """Affiche les données et la droite de régression."""
        # Charger les données
        df = pd.read_csv('data.csv')

        theta0, theta1 = load_model('model.json')


        # Créer la figure
        plt.figure(figsize=(10, 6))
        
        # 1. Scatter plot des données
        plt.scatter(df['km'], df['price'], color='blue', alpha=0.7, label='Données réelles', s=100)
        
        # 2. Droite de régression
        km_range = range(0, int(df['km'].max()) + 10000, 1000)
        prices_pred = [theta1 * km + theta0 for km in km_range]
        plt.plot(km_range, prices_pred, color='red', linewidth=2, label=f'Régression: y = {theta1:.4f}x + {theta0:.0f}')
        
        # Mise en forme
        plt.xlabel('Kilométrage (km)', fontsize=12)
        plt.ylabel('Prix (€)', fontsize=12)
        plt.title('Régression Linéaire - Prix vs Kilométrage', fontsize=14)
        plt.legend(fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k€'))
        
        plt.tight_layout()
        plt.savefig('regression_plot.png', dpi=150)
        plt.show()


if __name__ == '__main__':
        plot_data_and_regression()