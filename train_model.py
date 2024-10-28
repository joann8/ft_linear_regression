from load_csv import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_values(lst: list):
    """Check data values"""
    if not all(isinstance(value, (int, float)) for value in lst):
        raise ValueError("Values in csv must be all int or float")
    if not all(value >= 0 for value in lst):
        raise ValueError("Values in csv must be all >= 0")


def predict_price(data: np.array, theta0: int | float, theta1: int | float) \
                  -> int | float:
    """Apply simple linear function to predict price"""
    return theta0 + theta1 * data


def normalize(data) -> {float, float, np.array}:
    """Normalize data before iterations"""
    # Pourquoi normaliser?
    #   > Convergence plus rapide : Si vos données comportent des échelles très
    #     différentes (par exemple, km en milliers et prix en unités plus
    #     grandes), la descente de gradient mettra + de temps à trouver les
    #     valeurs optimales de θθ. La normalisation place les caractéristiques
    #     sur une même échelle, permettant une convergence plus rapide.
    #   > Stabilité de l’algorithme : Sans normalisation, le gradient pourrait
    #     osciller et être instable, surtout si l’une des variables a une large
    #     amplitude de valeurs. En réduisant les valeurs à une échelle commune,
    #     on diminue le risque de variations dans les mises à jour de θ
    #   > Interprétabilité des résultats: La normalisation permet d’interpréter
    #     + facilement les poids θ, car chaque variable contribue de manière
    #     similaire aux calculs.
    # Comment?
    # Une méthode courante pour normaliser les données est la normalisation
    # Z-score ou standardisation. Cela revient à centrer chaque variable en
    # lui soustrayant sa moyenne et en la divisant par son écart-type.
    d_mean = np.mean(data)
    d_std = np.std(data)
    d_norm = np.array([float((value - d_mean) / d_std) for value in data])
    return d_mean, d_std, d_norm


def training_model(data: pd.DataFrame, l_rate=0.01, iterations=400):
    """Train the model of linear regression with gradient descent"""

    theta0, theta1 = 0, 0
    m = len(data['km'])

    # Normaliser les données est essentiel en régression linéaire avec descente
    # de gradient, car cela aide l’algorithme à converger + efficacement.
    price_mean, price_std, price_data = normalize(data['price'])
    km_mean, km_std, km_data = normalize(data['km'])

    for i in range(iterations):
        pred = predict_price(km_data, theta0, theta1)
        tmp_theta0 = np.sum(pred - price_data) * (l_rate / m)
        tmp_theta1 = np.sum((pred - price_data) * km_data) * (l_rate / m)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    # On denormalize les resultats pour recuperer l'echelle de depart
    denormalized_theta1 = theta1 * (price_std / km_std)
    denormalized_theta0 = price_mean - denormalized_theta1 * km_mean

    return denormalized_theta0, denormalized_theta1


def plot_result(data, theta0, theta1):
    """Display the plot result"""
    y_line = [(theta1 * x + theta0) for x in data["km"]]
    plt.figure("Linear regression results")
    plt.scatter(data["km"], data["price"])
    plt.plot(data["km"], y_line, color="red",
             label=f"y = {theta1}x + {theta0}")
    plt.xlabel("Km driven")
    plt.ylabel("Price")
    plt.grid()
    plt.show()


def main():
    try:
        data = load("data.csv")
        if data is None:
            return
        check_values(data['km'].tolist())
        check_values(data['price'].tolist())

        theta0, theta1 = training_model(data)
        plot_result(data, theta0, theta1)
        with open('model_parameters.txt', 'w') as f:
            f.write(f"{theta0}\n{theta1}")

    except Exception as e:
        print(type(e).__name__ + ":", e)


if __name__ == "__main__":
    main()
