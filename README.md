# Freight Route Optimization Demo

## Overview
This project implements a Gurobi-based multi-truck Vehicle Routing Problem (VRP) solver for freight delivery. The model determines the optimal assignment of delivery orders to trucks, minimizing the number of trucks used and the total delivery distance, subject to operational constraints.

## Optimization Model
The VRP model is formulated as a Mixed Integer Program (MIP) with the following key features:

- **Objective:**
  - Minimize the number of trucks used (primary objective).
  - Minimize the total one-way delivery distance (secondary objective).

- **Variables:**
  - `x[i, j, k]`: Binary variable; truck `k` travels from node `i` to node `j`.
  - `y[i, k]`: Binary variable; order `i` is served by truck `k`.
  - `z[k]`: Binary variable; truck `k` is activated (used).
  - `u[i, k]`: Continuous variable; position of node `i` in truck `k`'s route (MTZ).

- **Constraints:**
  - Each order is delivered by exactly one truck.
  - Each truck visits at most a set number of delivery locations (MAX_STOPS).
  - Each truck's total load does not exceed the maximum allowed (TL_MAX_LBS).
  - Trucks depart and return to the depot only if activated.
  - Orders can only be delivered by active trucks.
  - Flow conservation at delivery stops.
  - Subtour elimination (MTZ constraints).
  - Symmetry breaking (use lower-indexed trucks first).

## Mathematical Formulation

### Sets and Indices
- $i, j$: Delivery stops ($i, j \in \{1, \ldots, n\}$), depot is node $0$
- $k$: Trucks ($k \in \{1, \ldots, K_{\max}\}$)

### Parameters
- $\mathrm{MAX\,STOPS}$: Maximum stops per truck
- $\mathrm{TL\,MAX\,LBS}$: Maximum load per truck
- $w_i$: Weight of order $i$
- $d_{ij}$: Distance from node $i$ to node $j$

### Decision Variables
- $x_{i,j,k} \in \{0,1\}$: Truck $k$ travels from node $i$ to node $j$
- $y_{i,k} \in \{0,1\}$: Order $i$ is served by truck $k$
- $z_k \in \{0,1\}$: Truck $k$ is activated
- $u_{i,k} \geq 0$: Position of node $i$ in truck $k$'s route

### Objective
Minimize:
$$
\mathrm{BIG\,M} \cdot \sum_k z_k + \sum_{i,j,k} d_{ij} \cdot x_{i,j,k}
$$

### Constraints
1. **Each order served once:**
$$
\sum_k y_{i,k} = 1 \quad \forall i
$$
2. **Truck departs depot iff activated:**
$$
\sum_j x_{0,j,k} = z_k \quad \forall k
$$
3. **Truck returns to depot iff activated:**
$$
\sum_i x_{i,0,k} = z_k \quad \forall k
$$
4. **Stop limit:**
$$
\sum_i y_{i,k} \leq \mathrm{MAX\,STOPS} \quad \forall k
$$
5. **Weight limit:**
$$
\sum_i w_i y_{i,k} \leq \mathrm{TL\,MAX\,LBS} \quad \forall k
$$
6. **Only deliver on active trucks:**
$$
y_{i,k} \leq z_k \quad \forall i, k
$$
7. **Flow conservation at stops:**
$$
\sum_j x_{j,i,k} = y_{i,k} \quad \forall i, k
$$
$$
\sum_j x_{i,j,k} = y_{i,k} \quad \forall i, k
$$
8. **MTZ subtour elimination:**
$$
u_{i,k} - u_{j,k} + n x_{i,j,k} \leq n-1 \quad \forall i \neq j, k
$$
9. **Depot position anchor:**
$$
u_{0,k} = 0 \quad \forall k
$$
10. **Symmetry breaking:**
$$
z_{k} \geq z_{k+1} \quad \forall k < K_{\max}
$$

## Case Description
The demo case involves a set of delivery orders, each with a destination and weight. The depot is the starting point for all trucks. The model assigns orders to trucks and determines the delivery sequence for each truck, ensuring:

- No truck exceeds its stop or weight limits.
- Only active trucks are used for deliveries.
- The solution is optimal with respect to truck count and delivery distance.

## Files
- `main.py`: Entry point for running the optimization.
- `preprocess.py`: Preprocessing utilities for order data.
- `src/`: Contains the core optimization logic and supporting modules.
- `data/`: Sample order datasets.
- `requirements.txt`: Python dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`

## Requirements
- Python 3.8+
- Gurobi

## License
MIT License
