

import pandas as pd 
import numpy as np 
import random 
from pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time
from utils_solver import mapaAleatorio, save_obj, load_obj
from utils_genetic import Map, solve_genetic_algorithm



N_CARS        = 16
N_STATIONS    = 128
AREA_SIZE     = 7
N_STATES      = 10
SOLUCIONES    = dict()
MAX_COST      = N_STATES
MAX_COBERTURE = N_STATES
BITS_PER_STATION = N_STATIONS - N_CARS

N = 8
PC = .135
PM = .185
MAX_ITERATIONS = 10


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



TEST_POSITIONS = load_obj("TEST_POSITIONS")
TEST_MAPS = load_obj("TEST_MAPS")

try:
	RESULTS_GENETIC = load_obj("RESULTS_GENETIC")
	print("SOLUTIONS FOUND")
except Exception as e:
	RESULTS_GENETIC = dict()

for i in TEST_MAPS.keys():
	MAP_df = TEST_MAPS[i]
	print("Executing MAP: ",i)
	for j in TEST_POSITIONS.keys():
		MAP = Map(N_STATIONS, N_STATES, MAX_COST, MAP_df)
		name = "MAP" + str(i) + "_POSITION" + str(j)
		SOLUCIONES = dict()
		try:
			print("RESULTS for {} -> {} ".format(name, RESULTS_GENETIC[name]))
		except Exception as e:
			POSITIONS = TEST_POSITIONS[j]
			print("\n\tExecuting POSITIONS: ", POSITIONS)
			ts = time.time()
			POPULATION_X, solution = solve_genetic_algorithm( N, PC, PM, MAX_ITERATIONS, MAP, N_STATIONS, AREA_SIZE, MAX_COBERTURE, N_CARS, N_STATES, SOLUCIONES, POSITIONS )
			time_f = time.time() - ts

			RESULTS_GENETIC[name] = dict()
			RESULTS_GENETIC[name]["POPULATION_X"]  = POPULATION_X
			RESULTS_GENETIC[name]["POSITIONS"] = POSITIONS
			RESULTS_GENETIC[name]["TIME"]      = time_f

			min_cost = 10000
			for key in POPULATION_X.keys():
				cost_i = POPULATION_X[key]["FLEET"].accumalated_cost
				if cost_i < min_cost:
					min_cost = cost_i

			RESULTS_GENETIC[name]["SCORE"] = min_cost
			print("\n\n\n ---------------END ITERATION  === ", min_cost)

			save_obj(RESULTS_GENETIC, "RESULTS_GENETIC")


