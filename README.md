# Freight Route Optimization Demo

## Overview
This project implements a Gurobi-based multi-truck Vehicle Routing Problem (VRP) solver for freight delivery. The model determines the optimal assignment of delivery orders to trucks, minimizing the number of trucks used and the total delivery distance, subject to operational constraints.

## Optimization Model Description

### Objective
The model aims to minimize two things: first, the total number of trucks used for deliveries, and second, the total distance traveled by all trucks. The solver prioritizes using fewer trucks, and then, among those solutions, selects the one with the shortest delivery routes.

### Variables
- Truck route variable: Indicates whether a truck travels between two locations.
- Order assignment variable: Indicates whether a truck is assigned to deliver a specific order.
- Truck activation variable: Indicates whether a truck is used in the solution.
- Route position variable: Represents the position of a location in a truck's delivery sequence.

### Parameters
- Maximum stops per truck: The highest number of delivery locations a truck can visit.
- Maximum load per truck: The maximum weight a truck can carry.
- Order weights: The weight of each delivery order.
- Distance matrix: The distance between every pair of locations (including the depot).

### Constraints
1. Each order must be delivered by exactly one truck.
2. A truck can only depart from the depot if it is used (activated).
3. A truck can only return to the depot if it is used.
4. No truck can visit more delivery locations than the maximum allowed.
5. No truck can carry more weight than its maximum load capacity.
6. Orders can only be delivered by trucks that are active (used in the solution).
7. For each delivery stop, the number of trucks arriving must equal the number of trucks departing, ensuring proper flow.
8. Subtour elimination constraints prevent disconnected loops in the delivery sequence.
9. The depot is always the starting point for each truck's route.
10. Lower-indexed trucks are used before higher-indexed trucks, to break symmetry and improve solver efficiency.

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
