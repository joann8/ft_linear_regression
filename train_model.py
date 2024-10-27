from load_csv import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def check_values(lst: list):
    """Check data values"""
    if not all(isinstance(l, (int, float)) for l in lst):
        raise ValueError("Values in csv must be all int or float")
    if not all(l >= 0 for l in lst):
        raise ValueError("Values in csv must be all >= 0")


# def linear_regression(km : list[int, float], price : list[int,float]) -> list[(int, float), (int, float)]:
#     """Linear regression function to calculate coefficients (pente & interception)"""
#     # Fonction qui calcule les coefficients de la droite de régression
#     # price = theta1 * km + theta0 

#     k_mean = sum(km) / len(km)
#     p_mean = sum(price) / len(price)

#     # Calcul de la pente theta1
#     numerator = sum((k - k_mean) * (p - p_mean) for k, p in zip(km, price))
#     denominator = sum((k - k_mean) ** 2 for k in km)
#     theta1 = numerator / denominator

#     # Calcul de l'interception tehta0
#     theta0 = p_mean -  theta1 * k_mean

#     return [theta0, theta1]

# Normaliser les données est essentiel en régression linéaire avec descente de gradient,
# car cela aide l’algorithme à converger plus rapidement et efficacement.
# --> Convergence plus rapide : Si vos données comportent des échelles très différentes
# (par exemple, kilométrage en milliers et prix en unités plus grandes), la descente de
# gradient mettra plus de temps à trouver les valeurs optimales de θθ. La normalisation
# place les caractéristiques sur une même échelle, permettant une convergence plus rapide.
# =--> Stabilité de l’algorithme : Sans normalisation, le gradient pourrait osciller et être instable, surtout si l’une des variables a une large amplitude de valeurs. En réduisant les valeurs à une échelle commune, on diminue le risque de grandes variations dans les mises à jour de θθ.
# --> Interprétabilité des résultats : La normalisation permet d’interpréter plus facilement les poids θθ, car chaque variable contribue de manière similaire aux calculs.

# Comment normaliser ?
# Une méthode courante pour normaliser les données est la normalisation Z-score ou
# standardisation. Cela revient à centrer chaque variable en lui soustrayant sa moyenne
# et en la divisant par son écart-type.



def predict_price(data: pd.DataFrame, theta0: int | float, theta1 : int | float) -> int | float: 
    return theta0 + theta1 * data['km']

def normalize(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_norm = [float((value - data_mean) / data_std) for value in data]
    return data_mean, data_std, data_norm


def training_model(data: pd.DataFrame, learning_rate=0.01, iterations=100):
    
    # comment controler le depacement de theta? (eviter valeur infinies?)
    # borner les km?
    
    theta0, theta1 = 0, 0
    m = len(data['km'])
    # print(f"--> m = {m}")

    price_mean, price_std, price_data = normalize(data['price'])
    print(f"price_mean = {price_mean} | price_std = {price_std}")
    km_mean, km_std, km_data = normalize(data['km'])
    print(f"km_mean = {km_mean} | km_std = {km_std}")

    new_data = pd.DataFrame({'km': km_data, 'price': price_data})
    print(new_data)
   # price = np.array()
   # print(price)

    # boucle pour le nombre d'ierations donnees en param 
    for i in range(iterations):
        # Calculate linear regression for each km in the km base
        predictions = predict_price(new_data, theta0, theta1)  # Y 
        #print(f"predictions =\n {predictions}")
        #print(f"price =\n {data['price']}")
        #print(f"km =\n {data['km']}")

        # Calculate error for each km value in the km base ( fonction cout)
        #errors = np.sum((predictions - new_data['price']) ** 2) / (2 * m)
        #print(f"errors =\n {errors}")

        # calcul gradient
        #errors42 = np.sum(predictions - data['price'])
        tmp_theta0 = np.sum(predictions - new_data['price']) * (learning_rate / m)
        #print(f"errors42 =\n {errors42}")
        # errors42bis = np.sum((predictions - data['price']) * data['km'])
        tmp_theta1 = np.sum((predictions - new_data['price']) * new_data['km']) * (learning_rate / m)

        #tmp_theta0 = theta0 - (learning_rate / m) * np.sum(errors)
        #tmp_theta1 = theta1 - (learning_rate / m) * np.sum(errors * data['km'])
        # Simultaneously update theta0 and theta1
        # if tmp_theta0 != tmp_theta0 or tmp_theta1 != tmp_theta1:
        #     print("----->break")
        #     break
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        # print(f"Theta0 = {theta0} | Theta1 = {theta1}")

    print(f"Theta0 = {theta0} | Theta1 = {theta1}")

    denormalized_theta1 = theta1 * (price_std / km_std)
    denormalized_theta0 = price_mean - denormalized_theta1 * km_mean
    return denormalized_theta0, denormalized_theta1

def plot_result(data, theta0, theta1):
    y_line = [(theta1 * x + theta0) for x in data["km"]]
    plt.figure("Linear regression results", figsize=(10, 5))
    plt.scatter(data["km"], data["price"], color="blue", alpha=0.7)
    plt.plot(data["km"], y_line, color="red", label=f"y = {theta1}x + {theta0}")
    plt.xlabel("Kilometers Driven (km)")
    plt.ylabel("Price")
    plt.axhline(0, color="black", lw=0.5, ls="solid")
    plt.axvline(0, color="black", lw=0.5, ls="solid")
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
        print(f"___FINAL___\nTheta0 = {theta0} | Theta1 = {theta1}")

        plot_result(data, theta0, theta1)

        theta0, theta1 = training_model(data, iterations=50000)
        plot_result(data, theta0, theta1)

        with open('model_parameters.txt', 'w') as f:
            f.write(f"{theta0}\n{theta1}")

    except Exception as e:
        print(type(e).__name__ + ":", e)
   

if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser(description="Linear regression training program")
    # parser.add_argument("-o", "--output", type=open_thetafile, default="theta.csv", help="Output data file")

