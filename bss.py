import random
import pandas as pd 
import numpy as np  
import random
import pandas as pd 
import numpy as np  
import operator
import copy
import utils_solver
from tqdm import tqdm
import time

def countSOLUCIONES(SOLUCIONES):
    count  = 0

    for i in SOLUCIONES.keys():
        count+= len(SOLUCIONES[i])
    return count

class Map(object):
    def __init__(self, num_estaciones, num_estados, MAX_COST):
        #TO DO: use bycicle_slots_cost and parking_slots_cost 
        #to upadte self.weights
        self.distances = utils_solver.mapaAleatorio( num_estaciones, num_estados)
        self.available_stations = list(self.distances.columns)
        self.parking_slots_cost = list()
        self.bycicle_slots_cost = list()
        self.weights      = self.distances.copy()
        self.all_stations = list(self.distances.columns)
        self.MAX_COST = MAX_COST
    def __repr__(self):
        print("\nMap: \n\tmatrix\n\t available_stations \
         \n\t parking_slots_cost \n\t bycicle_slots_cost\n\t distances,weights: matrix")

    def update_available_stations( self, FLEET):
        # For all car it remove from available_stations list all the spots in which cars
        #are allocated the it remove the already-route-assigned stations.
        self.available_stations = self.all_stations.copy()
        self.weights      = self.calculateWeights(FLEET)

        for i in FLEET.fleet:
            car = FLEET.fleet[i]
            if car.position in self.available_stations: #Take away the stations thata are ocupied by some car.
                self.available_stations.remove(car.position)

            for occupied_area_stations in car.subsystem_list: #Take away the stations that are assigned to a car as part of the optimization problem.
                if occupied_area_stations in self.available_stations:
                    self.available_stations.remove(occupied_area_stations)

    def calculateWeights(self, FLEET, next_time = False):
        """
        In this function we use all the available infromation to represent the expected cost
        of traveling from stattion A to station B

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




        
class Fleet(object):
    """
    This class cointain all the vehicles that perform that allocate the bycles to
    each station.
    """

    def __init__(self, AREA_SIZE, MAX_COBERTURE):
        self.fleet = dict()
        self.accumalated_cost = 0
        self.positions = list()
        self.cost_distribution = dict()
        self.AREA_SIZE = AREA_SIZE
        self.MAX_COBERTURE = MAX_COBERTURE
        self.cars_rebalancing_time = pd.DataFrame()

    def insertCar (self,car):
        self.fleet[car.id_car] = car

    def get_fleet_chromosome():
        """
        This function represent how the stations will be assiged to the vehicles in the Map. 
            1.Take one vehicles, transform the assiged stations to binary representation.
            2.append the  binary representation to the chromosome list.

        """
        for i in self.fleet:
            car = self.fleet[i]
        return car


    def set_cost_distribution(self):
        cost_distribution  = dict()
        for i in self.fleet.keys():
            cost_distribution[i] = self.fleet[i].solution_cost
        #Sort dictionary by values:
        sorted_cost_distribution = pd.Series(cost_distribution).sort_values( ascending = False)
        self.cost_distribution = sorted_cost_distribution


    def __repr__(self):
        repr_str = "\n\t---------FLEET--------------"
        repr_str +=  "\t fleet: list of cars\n\t accumalated_cost: sum of cars solving subsystem\n\t positions: position of each car"
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
                car_possible_area = car_stations_weights[car_stations_weights <= MAX_COBERTURE]
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

    def solve_subsystems(self, MAP, SOLUCIONES):
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
            seq, cost     = utils_solver.solveSystem(subsystem_S, SOLUCIONES, N_STATES, N_STATIONS, equivalent_systems =True, start_solutions = False )
            car.subsystem_solution  = seq
            car.solution_cost = cost
            self.accumalated_cost += cost

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
         return "CLASS Car: \n\tposition \n\tcapacity\n\tnumber\n\tsubsystem\n\tcar_stations_weight\n\n "
    def __str__(self):
         return "CLASS Car: \n\tid_car: {}\n\tposition: {} \n\tcapacity: {}\
         \n\tcar_stations_weight: {}\n\nsubsystem: \n\t\t{}\n\n".format(self.id_car, \
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

    def get_mostExpensive_station(self):
        subsystem =  self.subsystem
        most_expensive_station_cost = 0 
        for col in subsystem.columns:
            sum_station_col = subsystem[col].replace("X", 0 ).sum()
            sum_station_index = subsystem.loc[col].replace("X", 0 ).sum()
            sum_station =  sum_station_col + sum_station_index 
            if most_expensive_station_cost < sum_station:
                most_expensive_station_cost = sum_station
                most_expensive_station = col

        #print("Most expensive station for car {}\n\t Station:{}\n\t Cost: {} ".format(self.id_car, most_expensive_station, most_expensive_station_cost))
        return most_expensive_station


"""
Note that   
[N_STATIONS - N_CARS][STATE<=MAX_COBERTURE] <= N_CARS * AREA_SIZE

All of this code must be executed to run test_optimization.py ....
"""

N_CARS        = 16
N_STATIONS    = 144
AREA_SIZE     = 8
MAX_COBERTURE = 5
N_STATES      = 10
SOLUCIONES    = dict()
MAX_COST      = N_STATES



MAP = Map(N_STATIONS, N_STATES, MAX_COST)
FLEET = Fleet(AREA_SIZE, MAX_COBERTURE)
# We assign a random station for each car
car_position = random.sample( list(MAP.weights.columns) , N_CARS)

for i in range(N_CARS):
    id_car = i
    position = car_position[i]
    capacity = random.randint(5,20) #TO DO: each car have their on capcity

    car = Car( id_car, position, capacity )
    FLEET.insertCar(car)

FLEET.update_carsPosition()          #Save car positions in FLEET.positions
MAP.update_available_stations(FLEET) #Remove car positio and car subsytem from available_stations
FLEET.assignArea(MAP)                #Assign subsystem to each car


car_test = FLEET.fleet[1]


#car_stations = car_stations[MAP.available_stations] #select just the available stations
#print("TEST 1:  \n{}".format(car_stations))
#car_stations_weights = MAP.car_cost(car, car_stations) #for this particular car how expensive is to travel from it's possition to all available stations
