from load_csv import load
import pandas as pd
import numpy as np

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

def predict_price(data: pd.DataFrame, theta0: int | float, theta1 : int | float) -> int | float: 
    return theta0 + theta1 * data['km']

def training_model(data: pd.DataFrame, learning_rate=0.01, iterations=100):
    
    theta0, theta1 = 0, 0
    m = len(data['km'])

    # boucle pour le nombre d'ierations donnees en param 
    for _ in range(iterations):
    
        # Calculate linear regression for each km in the km base
        predictions = predict_price(data, theta0, theta1)
        # Calculate error for each km value in the km base
        errors = predictions - data['price']
        tmp_theta0 = theta0 - (learning_rate / m) * np.sum(errors)
        tmp_theta1 = theta1 - (learning_rate / m) * np.sum(errors * data['km'])
        # Simultaneously update theta0 and theta1
        if tmp_theta0 != tmp_theta0 or tmp_theta1 != tmp_theta1:
            print("----->break")
            break
        theta0, theta1 = tmp_theta0, tmp_theta1
        print(f"Theta0 = {theta0} | Theta1 = {theta1}")

    return theta0, theta1

def main():
    try:
        data = load("data.csv")
        if data is None:
            return
        check_values(data['km'].tolist())
        check_values(data['price'].tolist())

        theta0, theta1 = training_model(data)
        with open('model_parameters.txt', 'w') as f:
            f.write(f"{theta0}\n{theta1}")

    except Exception as e:
        print(type(e).__name__ + ":", e)
   

if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser(description="Linear regression training program")
    # parser.add_argument("-o", "--output", type=open_thetafile, default="theta.csv", help="Output data file")

