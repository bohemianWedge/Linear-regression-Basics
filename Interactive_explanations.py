import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json

        
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

def part_one():
        st.markdown("""
        Welcome to this interactive tutorial on linear regression! 
        Explore how machine learning models learn to fit data by adjusting parameters.
        """)

        st.header("I - Raw Dataset Exploration :" )

        col1, col2 = st.columns([7, 3])
        with col1:
                fig3, ax3 = plt.subplots()
                ax3.scatter(db['km'], db['price'], alpha=1)
                ax3.set_xlabel('Mileage (km)')
                ax3.set_ylabel('Price (€)')
                ax3.set_title('Prix en fonction du kilométrage')
                st.pyplot(fig3)
        with col2:
                st.dataframe(db)

def part_two():
        st.header("II - Understanding Linear Regression")

        st.markdown("""
        ### The Goal
        Linear regression aims to find the best-fitting straight line through our data points. 
        This line is described by the equation:
        """)
        col1, col2 = st.columns([1, 1])
        with col1:
                st.write("")
                st.write("")
                st.write("")
                st.latex(r"y = ax + b")

        with col2:
                st.markdown("""
        Where:
        - **a** is the slope (how steep the line is)
        - **b** is the y-intercept (where the line crosses the y-axis)
        - **x** is our input variable
        - **y** is our predicted output
        """)

        st.markdown("""
        ### The Strategy
        We want to position a line on our scatter plot so that it's as close as possible 
        to all the data points. To do this, we adjust parameters **a** and **b** to minimize 
        the error between our predictions and the actual values.

        ### Measuring Error
        We use the **Mean Squared Error (MSE)** to measure how well our line fits the data:
        """)
        st.latex(r"MSE = \frac{1}{2n} \sum_{i=1}^{n} (y_{actual} - y_{predicted})^2")
        st.markdown("""
        The smaller the MSE, the better our line fits the data!
        """)

def part_three():
        st.header("III - Interactive Parameter Tuning")
        st.markdown("""
        Now it's your turn! Use the sliders below to manually adjust the line parameters 
        and try to fit the data as best as you can.

        **Goal:** Find values of **a** and **b** that minimize the error.
        """)
        st.subheader("Adjust Parameters")

        col1, col2 = st.columns([2, 1])
        with col2:
                # dynamicly chosse a and b values
                a_slide = st.slider(label="Slope (a):", max_value=0.040, min_value=-0.040, value=0.0, step=0.001)
                b_slide = st.slider(label="Intercept (b):", max_value=10000, min_value=4000,value=7000, step=10)

                # Calcul predictions
                y_pred = b_slide + a_slide * db['km']
                
                # Calcul MSE
                mse = ((db['price'] - y_pred) ** 2).mean() / 2
                
                # print error (MSE)
                st.metric(label="Your Mean Squared Error (MSE)", value=f"{mse:,.0f}")
                st.write(f"**Lower is better!**")
        with col1:
                # Create graph
                fig_interactive, ax_interactive = plt.subplots()
                ax_interactive.scatter(db['km'], db['price'], alpha=1)
                ax_interactive.set_xlabel('Mileage (km)')
                ax_interactive.set_ylabel('Price (€)')
                ax_interactive.set_title('Interactive Fitting')

                # Fix limite of axes
                ax_interactive.set_xlim(db['km'].min() - 5000, db['km'].max() + 5000)
                ax_interactive.set_ylim(db['price'].min() - 500, db['price'].max() + 500)
                
                # Add line to regression
                km_line = [db['km'].min(), db['km'].max()]
                price_line = [b_slide + a_slide* km for km in km_line]
                ax_interactive.plot(km_line, price_line, 'r-', linewidth=2, label="Your predictions")
                ax_interactive.legend()
                
                st.pyplot(fig_interactive)

        st.info("💡 **Tip:** Adjust the slope first to match the general trend, then fine-tune the intercept.")

def part_foor():
        st.header("IV - Gradient Descent Convergence")
        st.markdown("""
        Manual tuning is tedious! Let's see how a machine learning algorithm can 
        automatically find the optimal parameters.

        ### How It Works
        **Gradient Descent** is an optimization algorithm that:
        1. Starts with random parameter values
        2. Calculates the error
        3. Adjusts parameters in the direction that reduces error
        4. Repeats until it finds the minimum error

        The graph below shows how the error decreases over training iterations.
        """)
        fig_cost, ax_cost = plt.subplots(figsize=(10, 5))
        ax_cost.plot(costs, color='blue', linewidth=1)
        ax_cost.set_xlabel('Iteration')
        ax_cost.set_ylabel('Cost (MSE)')
        ax_cost.set_title('Cost evolution during training')
        ax_cost.grid(True, alpha=0.3)
        ax_cost.set_yscale('log')  # log scale to better obsserve convergeance
        
        st.pyplot(fig_cost)
        final_mse = costs[-1]
        final_a, final_b = a_hist[-1], b_hist[-1]
        n_iterations = iterations[-1]

        st.subheader("Training Loss Over Time")
        st.markdown("Watch how the algorithm progressively reduces the error:")

        st.markdown("""
        Notice how the error drops rapidly at first, then gradually levels off as 
        the algorithm approaches the optimal solution.
        """)

        st.success(f"""
        #### Training Complete!
        - **Final slope (a):** {final_a:.4f}
        - **Final intercept (b):** {final_b:.4f}
        - **Final MSE:** {final_mse:.4f}
        - **Training iterations:** {n_iterations}
        """)

def part_five():
        st.header("V - Watch the Model Learn")

        st.markdown("""
        Now let's visualize the entire learning process! Use the slider below to 
        step through the training iterations and see how the regression line evolves 
        to fit the data.
        """)

        st.subheader("Training Progress")
        col1, col2 = st.columns([2, 1])

        with col2:
                st.markdown("Slide to see how the model improves over time:")
                curr_iter = st.slider(label="iterations", min_value=iterations[0], max_value=iterations[-1])

                a = history['a_values'][curr_iter]
                b = history['b_values'][curr_iter]

                st.metric(label='a', value=f"{a:.5f}")
                st.metric(label='b', value=f"{b:.2f}")
        

        with col1:

                fig_learn, ax_learn = plt.subplots()
                ax_learn.scatter(db['km'], db['price'], alpha=1)
                ax_learn.set_ylabel('Price (€)')
                ax_learn.set_title('Interactive Fitting')

                ax_learn.set_xlim(db['km'].min() - 5000, db['km'].max() + 5000)
                ax_learn.set_ylim(db['price'].min() - 500, db['price'].max() + 500)

                km_line = [db['km'].min(), db['km'].max()]
                price_line = [b + a * km for km in km_line]
                ax_learn.plot(km_line, price_line, 'r-', linewidth=2, label="Your predictions")

                # Ajouter les lignes d'erreur pour chaque point
                y_pred_points = [b + a * km for km in db['km']]
                for i in range(len(db)):
                        ax_learn.plot([db['km'].iloc[i], db['km'].iloc[i]], 
                                    [db['price'].iloc[i], y_pred_points[i]], 
                                    'g-', alpha=0.5, linewidth=0.5)
                
                # Ajouter une ligne fictive pour la légende
                ax_learn.plot([], [], 'g-', alpha=0.5, linewidth=1, label='Error (residuals)')

                ax_learn.legend()

                

                st.pyplot(fig_learn)

        st.metric(label='MSE', value=f"{costs[curr_iter]:,.0f}")


def main():
        st.title("Interactive Linear Regression Explorer")
        part_one()
        part_two()
        part_three()
        part_foor()
        part_five()
main()