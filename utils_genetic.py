import pandas as pd 
import numpy as np 
import random 
from pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time
import utils_solver
import operator


#MODEL= "test"; len_population = 0; len_pc = 0; len_pm= 0 ; N_WORKERS =16




def solve_genetic_algorithm(N, PC, PM, MAX_ITERATIONS, MAP, N_STATIONS, AREA_SIZE, MAX_COBERTURE, N_CARS, N_STATES, SOLUCIONES, POSITIONS = False ):
    """
    Solve a Genetic Algorithm with 
        len(genoma)               == GENLONG, 
        POPULATION                == N individuals
        Mutation probability      == PM
        Crossover probability     == PC
        Max number of iterations  == MAX_ITERATIONS
        Cost function             == MODEL

    """
    STILL_CHANGE = True 
    still_change_count = 0 
    if POSITIONS:
        car_position = POSITIONS
    else:
        car_position = random.sample(list(MAP.weights.columns), N_CARS)

    SOLUTIONS_GENETIC = dict()
    POPULATION_X = generate_population( N, MAP, N_STATIONS, AREA_SIZE, MAX_COBERTURE, N_CARS, car_position)
    POPULATION_X = score_chromosome(POPULATION_X, SOLUTIONS_GENETIC, N_STATES, N_STATIONS, SOLUCIONES, MAP  )
    POPULATION_X = sort_population(POPULATION_X)
    GENLONG = len(POPULATION_X[0]["GENOMA"])

    n_iter = 0 
    while (n_iter <= MAX_ITERATIONS) & STILL_CHANGE:

        POPULATION_Y = cross_mutate(POPULATION_X, N, PC, PM, GENLONG) #<--------------NOT CHANGE
        #HELP--------------------------------------------------------------------------


        #RESELECT score_chromosome
        POPULATION_X = score_chromosome(POPULATION_X, SOLUTIONS_GENETIC, N_STATES, N_STATIONS, SOLUCIONES, MAP )
        POPULATION_Y_DEBUG = POPULATION_Y
        POPULATION_Y = score_chromosome(POPULATION_Y, SOLUTIONS_GENETIC, N_STATES, N_STATIONS, SOLUCIONES, MAP )

        POPULATION_X = sort_population(POPULATION_X)
        POPULATION_Y = sort_population(POPULATION_Y)

        POPULATION_X = select_topn(POPULATION_X, POPULATION_Y, N)
        
        equal_individuals, max_score = population_summary(POPULATION_X)

        n_iter += 1
        if equal_individuals >= len(POPULATION_X.keys())*.8:
            still_change_count +=1 
            #print("SAME ** {} ".format(still_change_count))
            if still_change_count >= 50:
                print("\n\n\nGA Solved: \n\tGENLONG: {} \n\tN: {}\n\tPC:{}, \n\tPM: {}\n\tN_WORKERS: {} \n\tMAX_ITERATIONS: {}\n\tMODE: {}".format( GENLONG, N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL ))
                STILL_CHANGE = False

    return POPULATION_X, SOLUTIONS_GENETIC



def score_chromosome(POPULATION_X, SOLUTIONS_GENETIC, N_STATES, N_STATIONS, SOLUCIONES, MAP):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function and parameters saved individual_score
    MODEL dictionary.

    """
    s = 0
    POPULATION_SIZE = len(POPULATION_X)
    for s in POPULATION_X.keys():
        print("\n\nSolving {} out of {}".format(s, POPULATION_SIZE))

        if POPULATION_X[s]["GENOMA"] in SOLUTIONS_GENETIC.keys():
            score = SOLUTIONS_GENETIC[POPULATION_X[s]["GENOMA"]]
            POPULATION_X[s]["SCORE"] = score
            print("Modelo encontrado")
            s+=1
        else:
            assignation = POPULATION_X[s]["FLEET"].decode_chromosome(POPULATION_X[s]["GENOMA"])
            if POPULATION_X[s]["FLEET"].valid_assignation(assignation, MAP):
                POPULATION_X[s]["FLEET"].reasignArea(assignation, MAP)
                POPULATION_X[s]["FLEET"].solve_subsystems(POPULATION_X[s]["MAP"], SOLUCIONES,  N_STATIONS, N_STATES)
                score = -1*POPULATION_X[s]["FLEET"].accumalated_cost 
                genoma = POPULATION_X[s]["GENOMA"]
                SOLUTIONS_GENETIC[genoma] = score
                POPULATION_X[s]["SCORE"]  = score
                s+=1

            else:
                #print("Invalid asignation: \n{}".format(assignation)) 
                score = (POPULATION_X[s]["FLEET"].WORST_SCORE)
                POPULATION_X[s]["SCORE"]  = score
        print("\n\n\t\tSCORE at s {}: {}".format(s, score))

    return POPULATION_X

def generate_population(N , MAP, N_STATIONS, AREA_SIZE, MAX_COBERTURE, N_CARS, car_position): #<------------- Esta parte debe tener la parte de asignacion de alatatoria
    """
    Each individual is represented by a FLEET
    The Fleet is created using the Class Fleet and follows it's own proedures.
    After creating a FLEET, this is saved in a dictionary linked to its chromosome representation.
    """
    POPULATION_X = dict()

    for i in range(N):
        print("GENERATE POPULATION N: ", i)
        POPULATION_X[i] = dict()
        POPULATION_X[i]["MAP"]  =  copy.deepcopy(MAP)
        FLEET = Fleet(AREA_SIZE, MAX_COBERTURE, N_STATIONS)
        # We assign a random station for each car

        for j in range(N_CARS):
            id_car = j
            position = car_position[id_car]
            capacity = random.randint(5,20) #TO DO: each car have their on capcity
            car = Car( id_car, position, capacity )
            FLEET.insertCar(car)

        FLEET.update_carsPosition()          #Save car positions in FLEET.positions
        MAP.update_available_stations(FLEET) #Remove car positio and car subsytem from available_stations
        FLEET.assignArea(MAP)                #Assign subsystem to each car

        genoma = FLEET.get_fleet_chromosome(N_STATIONS)
        POPULATION_X[i]["GENOMA"] = genoma
        POPULATION_X[i]["SCORE"]  = np.nan 
        POPULATION_X[i]["FLEET"]  = FLEET

        #print(POPULATION_X)
        #print("New individual generated: {}".format(genoma))
    return POPULATION_X


def look_solutions(genoma, SOLUTIONS_GENETIC):
    try:
        return SOLUTIONS_GENETIC[genoma]
    except Exception as e:
        print("genoma is not save in SOLUTIONS_GENETIC")
        return False


def save_solutions(genoma, new_score, SOLUTIONS_GENETIC):
    SOLUTIONS_GENETIC[genoma] = new_score



def sort_population(POPULATION_X):
    POPULATION_Y = sorted(POPULATION_X.items(), key=lambda x: x[1]["SCORE"], reverse = True)
    POPULATION_NEW = dict()
    result = list()
    cont = 0
    for i in POPULATION_Y:
        POPULATION_NEW[cont] = i[1]
        cont += 1
    return POPULATION_NEW

def population_summary(POPULATION_X):
    suma = 0
    min_score = 10000
    max_score = -10000 
    lista_genomas = list()

    for key in POPULATION_X.keys():
        individual_score =  POPULATION_X[key]["SCORE"]
        lista_genomas.append(POPULATION_X[key]["GENOMA"])
        suma += individual_score
        if max_score < individual_score:
            max_score = individual_score
        if min_score > individual_score:
            min_score = individual_score
    promedio = suma/len(POPULATION_X.keys())
    equal_individuals = len(lista_genomas) - len(set(lista_genomas))
    #print("\n\nTOTAL SCORE: {} MEAN SCORE: {} \n\t MAX_SCORE: {} MIN_SCORE: {} ".format(suma, promedio, max_score, min_score))
    return equal_individuals, max_score


def select_topn( POPULATION_X, POPULATION_Y, N):
    for key in POPULATION_X.keys():
        new_key = key  + N
        POPULATION_Y[new_key] = POPULATION_X[key]

    POPULATION_Y = sort_population(POPULATION_Y)
    POPULATION_NEW =dict()

    for key in range(N):
        POPULATION_NEW[key] = POPULATION_Y[key]

    return POPULATION_NEW


def cross_mutate( POPULATION_X, N, PC, PM, GENLONG): #<----------NOT CHANGE
    POPULATION_Y = copy.deepcopy(POPULATION_X)
    #pprint(POPULATION_X)
    for j in range(int(N/2)):
        pc = random.uniform(1,0)

        #CROSSOVER
        if pc < PC:
            best = POPULATION_Y[j]["GENOMA"]
            worst = POPULATION_Y[N -j-1]["GENOMA"]
            startBest =  best
            startWorst = worst

            genoma_crossover = random.randint(0, GENLONG)
            best_partA  = best[:genoma_crossover]
            worst_partA = worst[:genoma_crossover]

            best_partB  =  best[genoma_crossover:]
            worst_partB = worst[genoma_crossover:]

            new_best    = best_partA  + worst_partB
            new_worst   = worst_partA + best_partB

            endBest = new_best
            endWorst =  new_worst

            POPULATION_Y[j]["GENOMA"]    = new_best
            POPULATION_Y[N-j-1]["GENOMA"]  = new_worst

            #print("\n\nCrossover Performed on individual {}-{}: \n\t StartBest: {} \n\t EndBest: {} \n\t StartWorst: {} \n\t EndWorst: {}".format(j,genoma_crossover, startBest, endBest, startWorst, endWorst))

    for j in range(N):
        #MUTATION

        pm = random.uniform(1,0)
        mutation_gen = random.randint(0, GENLONG-1)

        if pm < PM:
            #PERFORM MUTATION

            mutated_genoma = POPULATION_Y[j]["GENOMA"]
            start = mutated_genoma
            if mutated_genoma[mutation_gen] =="1":
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "0"
                mutated_genoma = "".join(mutated_genoma)
            else:
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "1"
                mutated_genoma = "".join(mutated_genoma)

            end = mutated_genoma

            POPULATION_Y[j]["GENOMA"] = mutated_genoma

            #print("\nMutation Performed on individual {}-{}: \n\t Start: {} \n\t End: {}".format(j, mutation_gen, start, end))

    return POPULATION_Y

def report_genetic_results(genoma, MODEL):
    end_population = MODEL["params"]["len_population"]
    end_pc         = end_population + MODEL["params"]["len_pc"]
    end_pm         = end_pc         + MODEL["params"]["len_pm"]

    POPULATION_SIZE  = int(genoma[:end_population], 2) +1
    PC               = 1/(int(genoma[end_population: end_pc], 2) +.01)
    PM               = 1/(int(genoma[end_pc: end_pm], 2) +.01)

    MAX_ITERATIONS = MODEL["params"]["max_iterations"]
    GENLONG        = MODEL["params"]["len_genoma"] + 1
    N_WORKERS      = MODEL["params"]["n_workers"]
    print("\n\n ** BEST GA **: \n\tgenlong: {}\n\tpc: {} \n\tPM: {}".format(POPULATION_SIZE, PC, PM ))





"""
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

"""

import random
import pandas as pd 
import numpy as np  
import random
import pandas as pd 
import numpy as np  
import operator
import copy
import utils_solver
import time

from utils_genetic import score_chromosome, generate_population, look_solutions, \
                          save_solutions, sort_population, population_summary, select_topn, \
                          cross_mutate, solve_genetic_algorithm, report_genetic_results



def countSOLUCIONES(SOLUCIONES):
    count  = 0

    for i in SOLUCIONES.keys():
        count+= len(SOLUCIONES[i])
    return count



        
class Fleet(object):
    """
    This class cointain all the vehicles that perform that allocate the bycles to
    each station.
    """

    def __init__(self, AREA_SIZE, MAX_COBERTURE, N_STATIONS):
        self.fleet = dict()
        self.accumalated_cost = 0
        self.positions = list()
        self.cost_distribution = dict()
        self.AREA_SIZE = AREA_SIZE
        self.MAX_COBERTURE = MAX_COBERTURE
        self.cars_rebalancing_time = pd.DataFrame()
        self.chromosome = ""
        self.size = len(self.fleet)
        self.N_STATIONS = N_STATIONS
        self.WORST_SCORE = -1000


    def bits_per_station(self,N_STATIONS):
        return int(np.ceil(np.log2(N_STATIONS)))


    def insertCar (self,car):
        self.fleet[car.id_car] = car
        self.size = len(self.fleet)

    def get_fleet_chromosome(self,N_STATIONS):
        """
        This function represent how the stations will be assiged to the vehicles in the Map. 
            1.Take one vehicles, transform the assiged stations to binary representation.
            2.append the  binary representation to the chromosome list.

        """
        chromosome_total = ""
        bits =  self.bits_per_station(N_STATIONS)
        for i in self.fleet:
            car = self.fleet[i]
            binary_stations   = [np.binary_repr(int(x.replace("estacion_", "")) -1, width= bits ) for x in car.subsystem_list]
            chromosome_i      = ''.join(binary_stations)
            chromosome_total += chromosome_i
        self.chromosome       = chromosome_total
        return chromosome_total

    def decode_chromosome(self,chromosome):
        """
        Divides a chromosome in blocks of size bits_per_station and decode them
        """
        start = 0 
        cut = self.bits_per_station(self.N_STATIONS)
        station_bits = cut
        stations_list = []
        while cut <=len(chromosome):
            gen = chromosome[start:cut]
            station  = "estacion_" + str(int(chromosome[start:cut],2)+1)
            stations_list.append( station)

            start=cut
            cut+= station_bits
        start = 0 
        cut = self.AREA_SIZE
        station_bits = cut
        assignation = list()
        while cut <=len(stations_list):
            assignation.append( stations_list[start:cut] )
            start=cut
            cut+= station_bits

        return assignation

    def valid_assignation(self, assignation_list, MAP):
        for v in assignation_list:
            if len(set(v))< self.AREA_SIZE:
                print("WARNING: Same station assignated two times")
                return False
        for v in assignation_list:
            if v in self.positions:
                print("WARNING: Car position assignated to car rebalancing area")
        for v in assignation_list:
            for station in v:
                if  station not in MAP.weights.columns:
                    print("WARNING: Station doesn't exist in MAP")
                    return False
  
        cont1 = 0
        for assignation1 in assignation_list:
            cont1 +=1
            cont2 = 0 
            for assignation2 in assignation_list:
                cont2 +=1 
                if cont2 == cont1:
                    continue
                else:
                    if len([x for x in assignation1 if x in assignation2])>0:
                        print(assignation1)
                        print("Warning:  Same station assignated two times for different vehicles")
                        print(assignation2)
                        return False
        return True




    def set_cost_distribution(self):
        cost_distribution  = dict()
        for i in self.fleet.keys():
            cost_distribution[i] = self.fleet[i].solution_cost
        #Sort dictionary by values:
        sorted_cost_distribution = pd.Series(cost_distribution).sort_values( ascending = False)
        self.cost_distribution = sorted_cost_distribution


    def __repr__(self):
        repr_str =  "\n\t ------------------FLEET---------------------\n"
        repr_str += "\n\t fleet: list of cars\n\t accumalated_cost: sum of cars solving subsystem\n\t positions: position of each car"
        return repr_str

    def __str__(self):
        print(FLEET.fleet)

    def update_carsPosition(self ):
        """
        Returns a list with the position of each car so this station can be deleated
        from available spots
        """
        self.positions = list()
        for car in self.fleet:
            self.positions.append(self.fleet[car].position)


    def reasignArea(self, assignation_list, MAP):
        """
        Use the chromosome to represent the stations that each car will explore.
        """
        #Reset asignation
        for i in self.fleet:
            car = self.fleet[i]
            car.subsystem = pd.DataFrame()
            car.subsystem_list = list()
            MAP.update_available_stations(self)
        cont  = 0

        for assignation in assignation_list:
            car = self.fleet[cont]
            for station in assignation:
                car.add_car_subsystem(station)
                MAP.update_available_stations(self)
            cont += 1
            car.set_subsystem( MAP)

    def reasignAreaGradient(self, MAP):
        """
        Select all the cars that dont have 
        """
        reasignCars = list()
        for key in self.fleet.keys():
            car = self.fleet[key]
            car_subsystem = car.subsystem_list
            if len(car.subsystem_list) < self.AREA_SIZE:
                reasignCars.append(car)
                n_reasignation = self.AREA_SIZE - len(car.subsystem_list)

        for i in range(n_reasignation): #number of statitions assigned to each car
            #print("---------------------------NEXT AREA--------------------------------".format(i))
            for car in random.sample(reasignCars, len(reasignCars)): #car are selected randomly
                #print("----------------------------FLEET KEY------------------------------".format(j))
                #For each car we want to know which stations it can visit
                car_stations = MAP.weights.loc[car.position] #row in weight matriz
                car_stations = car_stations[MAP.available_stations] #select just the available stations
                #print("TEST 1:  \n{}".format(car_stations))
                car_stations_weights = self.car_cost(car, car_stations) #for this particular car how expensive is to travel from it's possition to all available stations
                car.set_stations_weight( car_stations_weights )
                #print("TEST 1.1:  \n{}".format(car_stations_weights))
                car_possible_area = car_stations_weights[car_stations_weights <= self.MAX_COBERTURE]
                #print("TEST 2:  \n{}".format(car_possible_area))

                if len(car_possible_area) > 0:
                    car_select = random.choice(car_possible_area.index)
                    car.add_car_subsystem(car_select)
                    #print("TEST 3 car select: \n{}".format(car_select))
                    #We have to take out the selected station from the available list
                    MAP.update_available_stations(self)
                    #print("TEST 4 available_stations: \n{}".format(MAP.available_stations))
                else:
                    print("Not available stations for car {} ".format(j))

        #print("END Area assignation")


    def assignArea(self,MAP):
        """
        For each car we select randomly a station inside their coberture area
        to be taken in consideration for rebalancing. This process is repeated
        until each car has AREA_SIZE assiged stations.

        #TO DO: Delete print and print-commented
        """
        #Reset asignation
        for i in self.fleet:
            car = self.fleet[i]
            car.subsystem = pd.DataFrame()
            car.subsystem_list = list()
            
        MAP.update_available_stations(self)

        for i in range(self.AREA_SIZE): #number of statitions assigned to each car
            #print("---------------------------NEXT AREA--------------------------------".format(i))
            for j in random.sample(self.fleet.keys(), len(self.fleet.keys())): #car are selected randomly
                #print("----------------------------FLEET KEY------------------------------".format(j))
                car = self.fleet[j]
                #For each car we want to know which stations it can visit
                car_stations = MAP.weights.loc[car.position] #row in weight matriz
                
                car_stations = car_stations[MAP.available_stations] #select just the available stations
                #print("TEST 1:  \n{}".format(car_stations))
                car_stations_weights = self.car_cost(car, car_stations) #for this particular car how expensive is to travel from it's possition to all available stations
                car.set_stations_weight( car_stations_weights )
                #print("TEST 1.1:  \n{}".format(car_stations_weights))
                car_possible_area = car_stations_weights[car_stations_weights <= self.MAX_COBERTURE]
                #print("TEST 2:  \n{}".format(car_possible_area))

                if len(car_possible_area) > 0:
                    car_select = random.choice(car_possible_area.index)
                    car.add_car_subsystem(car_select)
                    #print("TEST 3 car select: \n{}".format(car_select))
                    #We have to take out the selected station from the available list
                    MAP.update_available_stations(self)
                    #print("TEST 4 available_stations: \n{}".format(MAP.available_stations))
                else:
                    print("Not available stations for car {} ".format(j))

        #print("END Area assignation")

    def solve_subsystems(self, MAP, SOLUCIONES, N_STATIONS, N_STATES):
        """
        After for each car an area under the MAX_COBERTURE was assiged, each car have
        a subsystem that needs to be solved. This means that we must find the
        route for each car that minimizes the weight of traveling for all the nodes in the
        system.

        """
        self.accumalated_cost =0 
        for j in random.sample(self.fleet.keys(), len(self.fleet.keys())): #car are selected randomly
            car           = self.fleet[j]
            car.set_subsystem(MAP)
            subsystem_S   = car.subsystem
            #print("TESTSUB: {}".format(subsystem_S))
            ts = time.time()
            seq, cost     = utils_solver.solveSystem(subsystem_S, SOLUCIONES, N_STATES, N_STATIONS, equivalent_systems =True, start_solutions = False )
            tt = time.time() - ts
            print("\n\nSolucion encontrada : \n\tSEQ: {} \n\tCOST: {}\n\tTIME: {}".format(seq,cost,tt))
            car.subsystem_solution  = seq
            car.solution_cost = cost
            self.accumalated_cost += cost
        self.chromosome = self.get_fleet_chromosome( N_STATIONS )
        self.set_cost_distribution()


    def make_rebalancing(self):
        """
        After finding the best area distribution each car must perform a rebalancing that allocate the correct number
        of bicicles to each station. This is represented by changing the  cost of this station to the MAX_COST.
        This way this station is not going to be taken into account for the next solve_subsystems iteration.
        Also the make_rebalancing function is going to take some time depending on the car that it's performing the
        rebalancing action. This time is added to the car.next_iteration.

        """

        for j in self.fleet.keys():
            car        = self.fleet[j]
            subsystem  = car.subsystem
            solution   = car.subsystem_solution
            movement   = solution[-1][1]
            solve_station = subsystem.columns[movement]
            car.position  = solve_station
            car.subsystem_list = list()
            car.subsystem = pd.DataFrame()

            #car.next_iteration_temp += self.cars_rebalancing_time[car]


    def car_cost(self,car,car_stations):
        """
        This function user the Car information to upadate the cost
        for a paricular car C to travel to station J. Taking into account 
        the size of the car.
        """
        #car_test_cost = car.capacity # Aqui le deberiamos aumentar el peso
        #de acuerdo a la capacidad del carro y la cantidad de lugares 
        #en la estacion

        return car_stations

class Car(object):
    """docstring for ClassName"""
    def __init__(self, id_car, position, capacity):
        #------------IDENTIFIERS---------------
        self.id_car = id_car
        self.position = position
        self.capacity = capacity
        self.car_stations_weights = np.nan

        #------------SOLUTION-----------------
        self.subsystem_list = list()
        self.subsystem = pd.DataFrame()
        self.subsystem_solution =list()

        #-------------COST---------------------
        self.solution_cost = 0
        self.next_iteration_temp = 0 

    def __repr__(self):
         return "CLASS Car: \n\tposition \n\tcapacity\n\tnumber\n\tsubsystem\n\tcar_stations_weight\n\tsubsystem_solution\n\n"
    def __str__(self):
         return "CLASS Car: \n\tid_car: {}\n\tposition: {} \n\tcapacity: {}\
         \n\n\tcar_stations_weight: {}\n\nsubsystem: \n\n{}\n\n".format(self.id_car, \
            self.position, self.capacity, self.car_stations_weights, self.subsystem)

    def set_subsystem(self, MAP):
        self.subsystem = MAP.weights.loc[ self.subsystem_list, self.subsystem_list].copy()

    def set_solution(self, solution):
        self.subsystem_solution = solution

    def set_stations_weight(self,car_stations_weights):
        """
        The weight from traveling from station A to B change depending on
        the car that is performing that trip.
        """
        self.car_stations_weights = car_stations_weights
    def add_car_subsystem(self,new_element_subsystem):
        if len(self.subsystem_list) ==0:
            self.subsystem_list = [new_element_subsystem]
        else:
            self.subsystem_list.append(new_element_subsystem)

    def get_mostExpensive_station(self, E):
        subsystem =  self.subsystem
        stations_cost = dict()
        most_expensive_station_cost = 0 
        for col in subsystem.columns:
            sum_station_col = subsystem[col].replace("X", 0 ).sum()
            sum_station_index = subsystem.loc[col].replace("X", 0 ).sum()
            sum_station =  sum_station_col + sum_station_index 
            stations_cost[col] = sum_station

        sorted_by_value = sorted(stations_cost.items(), key=lambda kv: kv[1])[:E]
        sorted_by_value = dict(sorted_by_value)
        #print("Most expensive station for car {}\n\t Station:{}\n\t Cost: {} ".format(self.id_car, most_expensive_station, most_expensive_station_cost))
        return sorted_by_value


class Map(object):
    def __init__(self, num_estaciones, num_estados, MAX_COST, df = pd.DataFrame(), df_bool = False):
        #TO DO: use bycicle_slots_cost and parking_slots_cost 
        #to upadte self.weights
        if df_bool:
            self.distances =  df
        else:
            self.distances = utils_solver.mapaAleatorio( num_estaciones, num_estados)
        self.available_stations = list(self.distances.columns)
        self.parking_slots_cost = list()
        self.bycicle_slots_cost = list()
        self.weights      = self.distances.copy()
        self.all_stations = list(self.distances.columns)
        self.MAX_COST = MAX_COST
        self.car_positions =  list()

    def __repr__(self):
        repr_str =  "\n\t ------------------MAP---------------------"
        repr_str += "\nMap: \n\tmatrix\n\t available_stations \
         \n\t parking_slots_cost \n\t bycicle_slots_cost\n\t distances,weights: matrix"
        return repr_str


    def update_available_stations( self, FLEET):
        # For all car it remove from available_stations list all the spots in which cars
        #are allocated the it remove the already-route-assigned stations.
        self.available_stations = self.all_stations.copy()
        self.weights      = self.calculateWeights(FLEET)
        self.car_positions = list()
        for i in FLEET.fleet:
            car = FLEET.fleet[i]
            if car.position in self.available_stations: #Take away the stations thata are ocupied by some car.
                self.available_stations.remove(car.position)
                self.car_positions.append(car.position)

            for occupied_area_stations in car.subsystem_list: #Take away the stations that are assigned to a car as part of the optimization problem.
                if occupied_area_stations in self.available_stations:
                    self.available_stations.remove(occupied_area_stations)

    def calculateWeights(self, FLEET, next_time = False):
        """
        In this function we use all the available infromation to represent the expected cost
        of traveling from station A to station B

        for i in self.distances.columns:
            for j in self.distances.columns:
                distance =  self.distance.loc[i,j]
                parking_cost = self.parking_slots_cost[j]
                bycicle_cost = self.bycicle_slots_cost[j]

                weight = bycicle_cost + parking_cost + distances
        """
        #When the time advance we expect new bycycles to arrive at different stations.
        #This can be represented as a new iteration where the weighs of the MAP change.
        if next_time:
            #MAKE SOME FUCNTION WITH BIKES
            self.weights = self.weights.replace("X", -100) -1
            self.weights = self.weights.replace(-101, "X")
            self.weights = self.weights.replace(-1, 0)

        #TWe block the stations we the cars in the Fleet are located using the MAX_COST
        for i in FLEET.fleet:
            car = FLEET.fleet[i]
            for ind in self.weights.index: #station_1, station_2 ...  station_n
                self.weights.loc[ind , car.position] = self.MAX_COST
            
        return  self.weights #En lo que encontramos una funcion que represente



    def change_station(self, car, MAP):
        min_cost = 1000000
        av_st_min = ""
        if len(self.available_stations)>0:
            for av_st in self.available_stations:
                car.subsystem_list.append(av_st)
                car.set_subsystem(MAP)
                subsystem =  car.subsystem.replace("X", 0)
                total_cost_with_new_station =  subsystem.sum().sum()
                #print("\tTEST COST: {} => {}".format(av_st, total_cost_with_new_station))
                if total_cost_with_new_station < min_cost:
                    min_cost = total_cost_with_new_station
                    av_st_min = av_st
                else:
                    #There is an other station with better performance un car.subsystem
                    car.subsystem_list.remove(av_st)
            #print("New station inserted on car {} subsystem => {}".format( car.id_car, av_st_min ))
        else:
            print("There isn't any available_station in Fleet")
            #TO DO: In this part we can take the second worst cart and take the 
            #worst station inside it's subsystem_list.

