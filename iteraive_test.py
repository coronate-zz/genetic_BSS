

import pandas as pd 
import numpy as np 
import random 
from pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time
from utils_solver import mapaAleatorio, save_obj, load_obj, subsytem_distribution_iterativeOptimization
from utils_genetic import Map, solve_genetic_algorithm, Fleet, Car



N_CARS        = 16
N_STATIONS    = 128
AREA_SIZE     = 7
N_STATES      = 10
MAX_COST      = N_STATES
MAX_COBERTURE = N_STATES
BITS_PER_STATION = N_STATIONS - N_CARS

ITERATIONS = [50 , 100, 500, 1000]


def cerate_test_enviroment():
    TEST_MAPS = dict()
    TEST_POSITIONS = dict()
    for i in range(10):
        TEST_MAPS[i] = mapaAleatorio(N_STATIONS, N_STATES)

    for i in range(10):
        positions_list = random.sample(range(1, N_STATIONS), N_CARS+1)
        positions = list()
        for station in positions_list:
            positions.append("estacion_" + str(station))
        TEST_POSITIONS[i] = positions
    save_obj(TEST_POSITIONS, "TEST_POSITIONS")
    save_obj(TEST_MAPS, "TEST_MAPS")



for N_ITERATION in ITERATIONS:
    name = "RESULT_ITERATIVE+" + str(N_ITERATION)
    TEST_POSITIONS = load_obj(name)
    TEST_MAPS = load_obj("TEST_MAPS")
    try:
    	RESULT_ITERATIVE = load_obj("RESULT_ITERATIVE_100")
    except Exception as e:
    	RESULT_ITERATIVE = dict()

    for i in TEST_MAPS.keys():
        MAP_df = TEST_MAPS[i]
        print("Executing MAP: ",i)
        for j in TEST_POSITIONS.keys():
            SOLUCIONES    = dict()
            MAP = Map(N_STATIONS, N_STATES, MAX_COST, MAP_df)

            name = "MAP" + str(i) + "_POSITION" + str(j)
            try:
                print(RESULT_ITERATIVE[name])

            except Exception as e:
                POSITIONS = TEST_POSITIONS[j]
                print("\n\tExecuting POSITIONS: ", POSITIONS)

                FLEET = Fleet(AREA_SIZE, MAX_COBERTURE, N_STATIONS)
                # We assign a random station for each car

                for j in range(N_CARS):
                    id_car = j
                    position = POSITIONS[id_car]
                    capacity = random.randint(5,20) #TO DO: each car have their on capcity
                    car = Car( id_car, position, capacity )
                    FLEET.insertCar(car)

                FLEET.update_carsPosition()          #Save car positions in FLEET.positions
                MAP.update_available_stations(FLEET) #Remove car positio and car subsytem from available_stations
                FLEET.assignArea(MAP)                #Assign subsystem to each car
                FLEET.solve_subsystems(MAP, SOLUCIONES, N_STATIONS, N_STATES)

                ts = time.time()
                MIN_COST, FLEET_MIN  = subsytem_distribution_iterativeOptimization(N_ITERATION,FLEET, MAP, SOLUCIONES, N_STATIONS, N_STATES)
                time_f = time.time() - ts

                RESULT_ITERATIVE[name] = dict()
                RESULT_ITERATIVE[name]["FLEET"]  = FLEET_MIN
                RESULT_ITERATIVE[name]["POSITIONS"] = POSITIONS
                RESULT_ITERATIVE[name]["TIME"]      = time_f
                RESULT_ITERATIVE[name]["SCORE"]     = MIN_COST
                save_obj(RESULT_ITERATIVE, name)







