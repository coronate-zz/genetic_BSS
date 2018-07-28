

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
import utils_genetic



N_CARS        = 16
N_STATIONS    = 128
AREA_SIZE     = 7
N_STATES      = 10
SOLUCIONES    = dict()
MAX_COST      = N_STATES
MAX_COBERTURE = N_STATES
BITS_PER_STATION = N_STATIONS - N_CARS

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

RESULTS_GENETIC = dict()
for i in TEST_MAPS.keys():
	MAP = TEST_MAPS[i]
	print("Executing MAP: ",i)
	for j in TEST_POSITIONS.keys():
		name = "MAP" + i + "_POSITION" +j
		try:
			print(RESULTS_GENETIC[name])
		except Exception as e:
			POSITIONS = TEST_POSITIONS[j]
			print("\n\tExecuting POSITIONS: ", POSITIONS)
			ts = time.time()
			POPULATION_X, solution = utils_genetic.solve_genetic_algorithm( N, PC, PM, MAX_ITERATIONS, MAP, N_STATIONS, AREA_SIZE, MAX_COBERTURE, N_CARS, N_STATES, SOLUCIONES, POSITIONS )
			time = time.time() - ts

			RESULTS_GENETIC[name] = dict()
			RESULTS_GENETIC[name]["FLEET"]     = POPULATION_X["FLEET"]
			RESULTS_GENETIC[name]["POSITIONS"] = POSITIONS
			RESULTS_GENETIC[name]["TIME"]      = time
			RESULTS_GENETIC[name]["SCORE"]     = POPULATION_X["FLEET"].accumulated_cost
			save_obj(RESULTS_GENETIC, "RESULTS_GENETIC")



"""
RESULT_ITERATIVE = dict()
for i in TEST_MAPS.keys():
	MAP = TEST_MAPS[i]
	print("Executing MAP: ",i)
	for j in TEST_POSITIONS.keys():
		POSITIONS = TEST_POSITIONS[j]
		print("\n\tExecuting POSITIONS: ", POSITIONS)
		ts = time.time()
		subsytem_distribution_iterativeOptimization(N_ITERATION,FLEET, MAP, SOLUCIONES)
		time = time.time() - ts
		name = "MAP" + i + "_POSITION" +j

		RESULT_ITERATIVE[name] = dict()
		RESULT_ITERATIVE[name]["FLEET"]     = POPULATION_X["FLEET"]
		RESULT_ITERATIVE[name]["POSITIONS"] = POSITIONS
		RESULT_ITERATIVE[name]["TIME"]      = time
		RESULT_ITERATIVE[name]["SCORE"]     = POPULATION_X["FLEET"].accumulated_cost
"""

