from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

# Crear distribuciones de probabilidad
rain_distribution = Categorical([[
    0.7,
    0.2,
    0.1
]])

maintenance_distribution = ConditionalCategorical([[
    [0.4, 0.6],
    [0.2, 0.8],
    [0.1, 0.9]
]], [rain_distribution])

train_distribution = ConditionalCategorical([[
    [
                [0.8, 0.2],
                [0.9, 0.1],
            ],
    [
                [0.6, 0.4],
                [0.7, 0.3],
            ],
    [
                [0.4, 0.6],
                [0.5, 0.5],
            ]
]], [rain_distribution, maintenance_distribution])

appointment_distribution = ConditionalCategorical([[
    [0.9, 0.1],
    [0.6, 0.4],
]], [train_distribution])

# Crear modelo y añadir estados
model = BayesianNetwork()
model.add_distributions([rain_distribution, maintenance_distribution, train_distribution, appointment_distribution])

# Añadir aristas entre nodos
model.add_edge(rain_distribution, maintenance_distribution)
model.add_edge(rain_distribution, train_distribution)
model.add_edge(maintenance_distribution, train_distribution)
model.add_edge(train_distribution, appointment_distribution)

