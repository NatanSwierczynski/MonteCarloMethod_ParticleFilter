import random
from typing import Tuple

HOW_MANY_STEPS = 31
HOW_MANY_SIMULATIONS = 20000
CONDITION = 4

def random_walk(n_steps: int) -> Tuple[int]:
    x, y = 0, 0
    directions = ((0,1), (1,0), (0, -1), (-1, 0))
    for _ in range(n_steps):
        (dx, dy) = random.choice(directions)
        x += dx
        y += dy
    return (x, y)


def main():
    for iterations in range(HOW_MANY_STEPS):
        succesful, overall = 0, 0
        for _ in range(HOW_MANY_SIMULATIONS):
            (x, y) = random_walk(iterations)
            if abs(x) + abs(y) <=  CONDITION:
                succesful += 1
            overall += 1
        print(f"For random walk of {iterations} steps, chance of being further "
        f"than {CONDITION} blocks away is {succesful/overall*100:.2f}%")


if __name__ == "__main__":
    main()