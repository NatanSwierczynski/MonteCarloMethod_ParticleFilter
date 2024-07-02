import random

HOW_MANY_SIMULATIONS = 1000000


def main():
    within = 0
    overall = 0
    for _ in range(HOW_MANY_SIMULATIONS):
        (x, y) = (random.uniform(-1, 1), random.uniform(-1, 1))
        if x**2 + y**2 <= 1:
            within += 1
        overall += 1
    #Wynik mnożony jest przez 4, jako że 4 to pole kwadratu w który wpisany jest okrąg o R = 1
    print(f"Approximation of PI using Monte Carlo method is: {4*within/overall:.5f}")

if __name__ == "__main__":
    main()