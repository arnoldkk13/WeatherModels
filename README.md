# WeatherModels
Houses the code for the Chaos project for Math435. This code was written by Kyle Arnold and William Miller
for the purpose of understanding Chaotic Systems.

Commands to run each chaotic system to generate the images found in the images folder:
Lorenz Attractor:
 
    python3 Simulator.py --system LorenzAttractor --seconds 200 --x 5.0 --y 5.0 --z 5.0 --visualize --animate
    
Rossler Attractor:

    python3 Simulator.py --system RosslerAttractor --seconds 1000 --visualize --animate
   
Chua's Circuit:

    python3 Simulator.py --system ChuaCircuit --seconds 1000 --x 0.0 --y 0.0 --z 1.0 --visualize --animate
    
Aizawa Attractor: 

    python3 Simulator.py --system AizawaAttractor --x .1 --y 0.0 --z 0.0 --seconds 1000 --visualize --animate

Lyapunov Exponents Calculation:
    python3 Lyapunov.py 

