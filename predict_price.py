import sys
import os


def get_km() -> int | float:
    """Get user input for mileage to check"""
    print("What is the car mileage to check? "
          "(please enter an int or a float >= 0)")
    input = sys.stdin.readline()
    km = float(input)
    if km < 0:
        raise ValueError("invalid input - mileage should be >= 0")
    return km


def predict_price(km: int | float, theta0: int | float, theta1: int | float) \
                  -> int | float:
    """Apply simple linear function to predict price"""
    value = theta0 + theta1 * km
    return 0 if value < 0 else value
    # for negative value in our model we consider the price is 0


def get_theta() -> list[int | float]:
    """Check if model parameters file exists, load if available"""
    theta0, theta1 = 0, 0
    if os.path.exists('model_parameters.txt'):
        # -______________________-Que faire en cas de string vides? fichier vide?
        with open('model_parameters.txt', 'r') as f:
            theta0 = float(f.readline().strip())
            theta1 = float(f.readline().strip())
            if theta0 != theta0 or theta1 != theta1:
                raise ValueError("The price cannot be estimated "
                                 "(theta0 or theta1 is nan)")
            if theta0 == float('inf') or theta0 == float('-inf') or \
               theta1 == float('inf') or theta1 == float('-inf'):
                raise ValueError("The price cannot be estimated "
                                 "(theta0 or theta1 inf)")
    return [theta0, theta1]


def main():
    try:
        km = get_km()
        theta = get_theta()
        print(f"Estimated price for {km} km: "
              f"{predict_price(km, theta[0], theta[1]):.2f} euros")

    except Exception as e:
        print(type(e).__name__ + ":", e)


if __name__ == "__main__":
    main()
