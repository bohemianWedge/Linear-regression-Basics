import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

def calculate_precision_metrics(y_true, y_pred):
    """
    Calculate various precision metrics for regression
    
    Args:
        y_true: pandas Series or list of actual values
        y_pred: pandas Series or list of predicted values
    
    Returns:
        dict: dictionary of metrics
    """
    n = len(y_true)
    
    # Mean Squared Error (MSE)
    squared_errors = [(y_true.iloc[i] - y_pred[i]) ** 2 for i in range(n)]
    mse = sum(squared_errors) / n
    
    # Root Mean Squared Error (RMSE)
    rmse = mse ** 0.5
    
    # Mean Absolute Error (MAE)
    absolute_errors = [abs(y_true.iloc[i] - y_pred[i]) for i in range(n)]
    mae = sum(absolute_errors) / n
    
    # R² Score (Coefficient of Determination)
    y_mean = y_true.mean()
    ss_res = sum([(y_true.iloc[i] - y_pred[i]) ** 2 for i in range(n)])  # Residual sum of squares
    ss_tot = sum([(y_true.iloc[i] - y_mean) ** 2 for i in range(n)])  # Total sum of squares
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Percentage Error (MAPE)
    percentage_errors = [abs((y_true.iloc[i] - y_pred[i]) / y_true.iloc[i]) for i in range(n) if y_true.iloc[i] != 0]
    mape = (sum(percentage_errors) / len(percentage_errors)) * 100 if percentage_errors else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2_score,
        'MAPE': mape
    }

def load_training_history(filepath='training_history.json'):
    """Charge l'historique d'entraînement depuis un fichier JSON."""
    try:
        with open(filepath, 'r') as f:
            history = json.load(f)
        return history
    except FileNotFoundError:
        print(f"⚠ Fichier {filepath} non trouvé. Lancez d'abord train.py")
        return None
    
db = pd.read_csv('data.csv')

history = load_training_history()
# history = None
if history == None:
        st.title("You have to train model first :")
        st.code("pip install -r requierement\npython train.py")
        st.stop()
costs = history['costs']
a_hist = history['a_values']
b_hist = history['b_values']
iterations = history['iterations']

def predict():
        st.header("Make Predictions")
        
        st.markdown("""
        Now that our model is trained, let's use it to predict car prices based on mileage!
        Enter a mileage value below to see what price the model predicts.
        """)
        
        # Récupérer les paramètres finaux du modèle
        final_a = a_hist[-1]
        final_b = b_hist[-1]

        st.latex(f"price = mileage * ({final_a:.4f}) + {final_b:.2f}")

        col1, col2 = st.columns(2)
        
        with col1:
                st.subheader("Input Parameters")
                
                # Input pour le kilométrage
                km_input = st.number_input(
                        label="Enter mileage (km):",
                        min_value=0,
                        max_value=300000,
                        value=100000,
                        step=1000
                )
                
        with col2:
                # Faire la prédiction
                predicted_price = final_b + final_a * km_input
                
                # st.markdown("---")
                st.subheader("Prediction Result")
                st.metric(
                        label="Predicted Price",
                        value=f"€{predicted_price:,.2f}"
                )

        # Créer le graphique avec tous les points de données
        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
        
        # Scatter plot des données originales
        ax_pred.scatter(db['km'], db['price'], alpha=1, label='Training data')
        
        # Tracer la ligne de régression
        km_line = [db['km'].min(), db['km'].max()]
        price_line = [final_b + final_a * km for km in km_line]
        ax_pred.plot(km_line, price_line, 'r-', linewidth=2, label='Regression line')
        
        # Marquer la prédiction
        ax_pred.scatter([km_input], [predicted_price], 
                        color='green', s=100, marker='o',  linewidths=1,
                        label='Your prediction', zorder=2)
        
        # Ajouter des lignes pointillées pour mieux visualiser
        ax_pred.plot([km_input, km_input], 
                        [ax_pred.get_ylim()[0], predicted_price],
                        'g--', alpha=0.5, linewidth=1)
        ax_pred.plot([ax_pred.get_xlim()[0], km_input], 
                        [predicted_price, predicted_price],
                        'g--', alpha=0.5, linewidth=1)
        
        ax_pred.set_xlabel('Mileage (km)', fontsize=12)
        ax_pred.set_ylabel('Price (€)', fontsize=12)
        ax_pred.set_title('Price Prediction Visualization', fontsize=14)
        ax_pred.legend()
        ax_pred.grid(True, alpha=0.3)
        st.pyplot(fig_pred)


        

        st.markdown("---")
        st.subheader("Model Performance Metrics")

        # Calculer les prédictions pour toutes les données d'entraînement
        y_pred_all = [final_b + final_a * km for km in db['km']]

        # Calculer les métriques
        metrics = calculate_precision_metrics(db['price'], y_pred_all)

        # Afficher les métriques
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
                st.metric("MSE", f"{metrics['MSE']:,.0f}")
        with col2:
                st.metric("RMSE", f"{metrics['RMSE']:,.0f}")
        with col3:
                st.metric("MAE", f"{metrics['MAE']:,.0f}")
        with col4:
                st.metric("R²", f"{metrics['R²']:.4f}")
        with col5:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
predict()