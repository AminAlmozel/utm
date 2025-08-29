# Safety-Aware Path Planning for UAVs in Urban Airspaces

This repository contains the implementation and experimental results of our **safety-aware pre-flight path planning framework** for Unmanned Aerial Vehicles (UAVs). The framework is designed to enhance the robustness and safety of drone operations in urban and mixed-use airspaces by incorporating **emergency landing contingencies**, **GPS fallback strategies**, and **communication-aware planning** into trajectory optimization.  

The work is motivated by real-world failure scenarios—such as GPS spoofing, communication loss, and propulsion failures—that threaten the safe operation of drone delivery systems. Our planner proactively integrates environmental knowledge (e.g., designated emergency landing zones, communication tower locations, and No-Fly Zones) to generate safer 3D flight paths.

---

## Features
- **Safety-Aware Path Planner**: Optimizes UAV trajectories with respect to emergency landing accessibility.  
- **GPS Fallback via Communication Towers**: Incorporates proximity to cellular towers to maximize coverage and support localization through signal triangulation.  
- **Airspace Constraints**: Models permanent and temporary No-Fly Zones (NFZs).  
- **Urban Environment Modeling**: Uses real-world geographic data (OpenStreetMap, KAUST campus data) to create realistic flight environments.  
- **Mission Profiles**: Simulates diverse operations, including food delivery, security patrols, research, inspection, and recreational flights.  
- **Emergency Scenario Evaluation**: Tests response times and landing behavior under injected emergencies.  
- **Extensible Simulator**: Modular implementation with environment, mission planner, physics simulator, event generator, and data logger.  

---

## Repository Structure
.
├── data/ # Geographic data, POIs, NFZs, tower locations
├── figs/ # Figures used in the paper
├── planner/ # Safety-aware path planning implementation
├── simulator/ # Core simulator modules (physics, environment, missions)
├── experiments/ # Scripts and configs for running experiments
├── results/ # Generated experimental results
├── docs/ # Documentation and supporting material
└── README.md # This file

---

## Getting Started

### Requirements
- Python 3.9+  
- Dependencies (see `requirements.txt`):  
  - numpy  
  - scipy  
  - gurobipy (for MILP optimization)  
  - shapely, geopandas (for geospatial data handling)  

### Installation and Running the Experiment
There is no installation, as the code runs using Python. To run the code, you can optionally generate a new mission profile by running the mission.py file. After that, you can choose parameter values for the pathplanner in pathplanner.py. Lastly, run the simulator using simulator.py
python3 mission.py
python3 simulator.py

The results will be saved in both plot/simulations/last and plot/simulations/run_YY-MM-DD-HHmmSS

To analyze the results we copy the results into full_runs/mission# and run data_analysis.py
python3 data_analysis.py
This will generate stats.csv files, which can be read by the matlab script
data_analytics.m 	for analyzing a single run
data_analytics_group 	for analyzing a group of experiments (emergency landing) with the same mission profile
data_analytics_comm.m 	for analyzing a group of experiments (communication) with the same mission profile

### Results

Our experiments demonstrate that:

Increasing safety weighting improves proximity to emergency landing areas, reducing response times under failures.

Communication-aware planning reduces out-of-coverage trajectory segments by over 40% with modest increases in path length.

These results show the value of integrating pre-flight contingency planning into UAV traffic management systems.

### Acknowledgments

This research was funded by King Abdullah University of Science and Technology's baseline support (BAS/1/1682-01-01).
We thank Christian Cloiseau, Product Development Manager at Thales, for his support and guidance, as well as the KAUST Security team and Campus and Community team for operational input and data support.


