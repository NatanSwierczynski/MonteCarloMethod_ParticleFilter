import numpy as np
import plotly.express as px
from random import random, gauss
import plotly.graph_objects as go
from copy import copy

N = 300
SIGMA = 1
VARIANCE = SIGMA**2
ID = 0

def resample(weights, samples):
    particles_resampled = []
    index = int(random() * N)
    beta = 0.
 
    for _ in range(N):
        beta += random() * 2. * max(weights)
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        particles_resampled.append(copy(samples[index]))
    return particles_resampled


def particle_filter(samples: np.ndarray, draw_graphs: bool = False) -> np.ndarray:
    weights = np.apply_along_axis(likelyhood, 1, samples)
    weights += 1.e-300
    predictions = np.asarray([move(sample, move_angle, move_forward) for sample in samples])
    if draw_graphs:
        graph(predictions, ID + 1, weights)
    samples = resample(weights, samples)
    # sample_indices = np.random.choice(len(samples), p=weights / np.sum(weights), size=N)
    # samples = predictions[sample_indices]
    return samples


def graph(samples: np.ndarray, id: int, weight: np.ndarray = None) -> None:
    if weight is None:
        weight = np.ones(N, dtype=int)
    fig1 = px.scatter(x=samples[:, 0], y=samples[:, 1], size=weight, color_discrete_sequence=["red"])
    fig2 = px.scatter(x=[robot_position[0]], y=[robot_position[1]], size=[1])
    fig = go.Figure(data = fig1.data + fig2.data)
    fig.update_layout(width=800, height=600)
    fig.update_yaxes(range=[-2, 3], scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[-2, 8])
    id += 1
    fig.show()


def likelyhood(location: list[int]) -> float:
    return np.exp(-0.5 * ((location[0] - robot_position[0]) ** 2 + (location[1] - robot_position[1])**2)/ VARIANCE)


def move(sample: list[float, float], angle: float, distance: float):
    turn = angle + gauss(0., 1.)
    turn %= 2 * np.pi

    gain = distance + gauss(0., 0.3)
    sample[0] += np.cos(angle) * gain
    sample[1] += np.sin(angle) * gain

    return sample

robot_position = [0, 0]
mean = np.array([0, 0])
convariance = np.diag([9, 9])
samples = np.random.multivariate_normal(mean, convariance, size=N)

# graph(samples, ID + 1)
# graph(samples, ID + 1, np.apply_along_axis(likelyhood, 1, samples))

move_angle = 0.1
move_forward = 1

for _ in range(5):
    robot_position = move(robot_position, move_angle, move_forward)
    print(robot_position)
    samples = particle_filter(samples, draw_graphs=True)
