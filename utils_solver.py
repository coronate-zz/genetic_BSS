
"""
Para la primera parte de la solución se tomara un sistema y se intentara resolver siuiendo
"""
import itertools
import pandas as pd  
import numpy as np 
import random
import itertools
import copy
from tqdm import tqdm
import pickle

tqdm.monitor_interval = 0


def save_obj(obj, name ):
    with open('diccionarios/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('diccionarios/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def subsytem_distribution_iterativeOptimization(N_ITERATION,FLEET, MAP, SOLUCIONES):
    """
    This fucntions works as follows:
        First it takes the fleet passed by the user. 
    """
    FLEET.solve_subsystems(MAP, SOLUCIONES)
    FLEET_SECONDARY = copy.deepcopy(FLEET)
    MIN_COST = 10000000
    firstIteration =  True
    for i in  tqdm(range(N_ITERATION), ascii= True, desc = "Iterative optimization"): 
        if firstIteration:
            start_cost = FLEET.accumalated_cost
            firstIteration = False
        #This part change for optimization
        FLEET_SECONDARY.assignArea(MAP)                #Assign subsystem to each car
        FLEET_SECONDARY.solve_subsystems(MAP, SOLUCIONES)
        if FLEET_SECONDARY.accumalated_cost < MIN_COST:
            FLEET_MIN = copy.deepcopy(FLEET_SECONDARY)
            MIN_COST = FLEET_MIN.accumalated_cost
            print("New min cost {}".format(MIN_COST))

    print("The subsystem distribution was optimized:\n\t Initial Cost: {} \n\t Final Cost: {}".format(start_cost, FLEET_MIN.accumalated_cost))

    return MIN_COST, FLEET_MIN 


def subsystem_distribution_iterativeGradientOptimization(N_ITERATION,FLEET, MAP, SOLUCIONES):
    FLEET.solve_subsystems(MAP, SOLUCIONES)
    FLEET_MIN = copy.deepcopy(FLEET)
    MIN_COST_ITERATION = 10000000
    firstIteration =  True
    for i in  tqdm(range(N_ITERATION), ascii= True, desc = "Iterative optimization"): 
        stillChange = True
        stillChange = True
        firstIteration = True
        FLEET_SECONDARY = copy.deepcopy(FLEET)
        FLEET_SECONDARY.assignArea(MAP)               
        FLEET_SECONDARY.solve_subsystems(MAP, SOLUCIONES)
        cont = 0

        while stillChange:
            print("Distribution optimization: {} of max 10".format(cont))
            if firstIteration:
                start_cost = FLEET_SECONDARY.accumalated_cost
                #print("START COST: " + str(start_cost))
                firstIteration =  False

            FLEET_SECONDARY.set_cost_distribution()
            cost_fleet = FLEET_SECONDARY.cost_distribution
            most_expensive_car = cost_fleet.head(1).index[0] #Id of car with the route with more cost.
            most_expensive_car = FLEET_SECONDARY.fleet[most_expensive_car]

            most_expensive_station = most_expensive_car.get_mostExpensive_station()
            most_expensive_car.subsystem_list.remove(most_expensive_station)
            most_expensive_car.set_subsystem(MAP)

            MAP.update_available_stations(FLEET_SECONDARY)
            MAP.change_station(most_expensive_car, MAP) #this function must change the worst station for most_expensive_car to another station
            
            FLEET_SECONDARY.solve_subsystems(MAP, SOLUCIONES)
            if FLEET_SECONDARY.accumalated_cost < FLEET.accumalated_cost:
                FLEET = copy.deepcopy(FLEET_SECONDARY)
                stillChange =  True
                cont = 0
                #print("New FLEET_MIN cost: {}".format(FLEET.accumalated_cost))
            else:
                if cont >= 10:
                    stillChange = False 
                else:
                    cont += 1
        FLEET_MIN = copy.deepcopy(FLEET_SECONDARY)
        MIN_COST =  FLEET_MIN.accumalated_cost
        print("The subsystem distribution was optimized:\n\t Initial Cost: {} \n\t Final Cost: {}".format(start_cost, MIN_COST))

        if FLEET_MIN.accumalated_cost < MIN_COST_ITERATION:
            FLEET_MIN_ITERATION = copy.deepcopy(FLEET)
            MIN_COST_ITERATION = FLEET_MIN_ITERATION.accumalated_cost
            print("New min cost {}".format(MIN_COST_ITERATION))
    return MIN_COST_ITERATION, FLEET_MIN_ITERATION


def subsystem_distribution_gradientOptimization(FLEET, MAP, SOLUCIONES):
    """We know the first element in cost_fleet dictionary is the car incurring in more cost. 
    We will change this car distribution chnaging the station with the higher avg cost
    for an other station in the available_stations list.
    if the available_stations list is empty then we will change with the
    the highest avg cost station of the  second element
    in dictionary (second most expensive car). 
    """
    stillChange = True
    firstIteration = True
    FLEET.solve_subsystems(MAP, SOLUCIONES)
    FLEET_SECONDARY = copy.deepcopy(FLEET)
    FLEET_SECONDARY.assignArea(MAP)               
    FLEET_SECONDARY.solve_subsystems(MAP, SOLUCIONES)
    cont = 0

    while stillChange:
        print("Distribution optimization: {} of max 10".format(cont))
        if firstIteration:
            start_cost = FLEET_SECONDARY.accumalated_cost
            #print("START COST: " + str(start_cost))
            firstIteration =  False

        FLEET_SECONDARY.set_cost_distribution()
        cost_fleet = FLEET_SECONDARY.cost_distribution
        most_expensive_car = cost_fleet.head(1).index[0] #Id of car with the route with more cost.
        most_expensive_car = FLEET_SECONDARY.fleet[most_expensive_car]

        most_expensive_station = most_expensive_car.get_mostExpensive_station()
        most_expensive_car.subsystem_list.remove(most_expensive_station)
        most_expensive_car.set_subsystem(MAP)

        MAP.update_available_stations(FLEET_SECONDARY)
        MAP.change_station(most_expensive_car, MAP) #this function must change the worst station for car i to another station
        
        FLEET_SECONDARY.solve_subsystems(MAP, SOLUCIONES)
        if FLEET_SECONDARY.accumalated_cost < FLEET.accumalated_cost:
            FLEET = copy.deepcopy(FLEET_SECONDARY)
            stillChange =  True
            cont = 0
            #print("New FLEET_MIN cost: {}".format(FLEET.accumalated_cost))
        else:
            if cont >= 10:
                stillChange = False 
            else:
                cont += 1
    FLEET_MIN = FLEET
    MIN_COST =  FLEET_MIN.accumalated_cost
    print("The subsystem distribution was optimized:\n\t Initial Cost: {} \n\t Final Cost: {}".format(start_cost, MIN_COST))

    return MIN_COST, FLEET_MIN 

def mapaAleatorio( N_STATIONS, N_STATES ): 
    #print("-------CREACION DE LISTA CON NOMBRES ESTACION---------")

    listaNombres = list()
    for i in range( N_STATIONS ): 
        nombre  =  "estacion_" +  str(i+1)
        listaNombres.append( nombre)
    #Matriz con columnas == numero de estaciones e index  ==  numero de estaciones

    estacionesMap = pd.DataFrame(  index=  listaNombres , columns= listaNombres )
    #print("-------CREACION DE MAPA ALEATORIO----------")

    for i in range( len( estacionesMap )):
        fila  = estacionesMap.index[i]
        for j in range( len( estacionesMap )):
                columna =  estacionesMap.columns[ j]
                if fila == columna:
                    estacionesMap.loc[ fila, columna ] = "X"
                else:
                    estacionesMap.loc[ fila, columna ] = random.randint( 1, N_STATES ) 
    return estacionesMap


def getCode( S ):
    lista =  []
    for i in S.columns:
        for j in S.columns:
            lista.append(str(S.loc[i, j] ))
    return lista

"""
Es necesario crear unos diccionarios globales para poder insertar las soluciones y que puedan ser consultados en
cualquier scope del codigo.
SOLUCIONES

 L     SYSTEM        CODE 

[2]   [X][p1]   == [][][][] => CODE
      [p2][X]

[3]   [][][]    == [][][][][][][][][] => CODE
      [][][]
      [][][]

[4]   [][][][]  == [][][][][][][][][][][][][][][][] => CODE
      [][][][]
      [][][][]
      [][][][]
...

"""


def getSolution( code, SOLUCIONES ):
    """
    Return the solution of the system if the code of the system is saved in the dictionary.
    1. Look for the dictionary -> L
    2. If L is found the this dictionary have all the previous solved systems of length L
    3. Look for the code s of system S in L

    """ 
    l = len(code)
    code = "".join(code)
    try:
        type(SOLUCIONES[l] )
        try:
            solution_sequence = SOLUCIONES[l][code]['sequence'] 
            solution_cost = SOLUCIONES[l][code]['cost'] 
            return solution_sequence, solution_cost
        except Exception as e:
            #print("No existe respuesta para code = {}".format(code))
            return False

    except Exception as e:
        #print( "No existe el nivel L = {} en el diccionario".format(l))
        SOLUCIONES[l] = dict()
        return False

def insertSolution(code, optimal_mov, min_cost, equivalent_systems, SOLUCIONES):
    """
    Once we solve the system S we save the optimal transition route in SOLUCIONES
    in the L index.
    """
    l = len(code)

    code_string = ''.join(code)
    #level = int(np.sqrt(l))
    #tab = "\t"
    #for i in range(level):
    #    tab = tab + "\t"
    #print("\n\n"+tab+"  --------Ingresando solucion sobre {}: => {} ------".format(l, code ))
    SOLUCIONES[l][code_string] = dict()
    SOLUCIONES[l][code_string]["cost"] = min_cost
    SOLUCIONES[l][code_string]["sequence"] = optimal_mov

    if equivalent_systems:
        #Generates all the possible equivalent systems given the code of a particular system.
        equivalentSystems = getEquivalentSystem(code)
        for equiCode in equivalentSystems:
            SOLUCIONES[l][equiCode]= dict()
            SOLUCIONES[l][equiCode]["cost"] = min_cost
            SOLUCIONES[l][equiCode]["sequence"] = optimal_mov    #TO DO: Exixten varios sistemas que son equivalentes. Es posible obtener todas las permutaciones
        #de las columnas para obtener los sistemas equivalentes.


def getEquivalentSystem(code):
    """
    Given a code of the system S this function return all the possible systems that have the same solution as S
    Example:

    [X][1][2]    == [X][1][2][3][X][4][5][6][X] => CODE
    [3][X][4]
    [5][6][X]

    [X][2][1]    == [X][2][1][5][X][6][3][4][X] => CODE
    [5][X][6]
    [3][4][X]
    """
    #print( "\n\ngetEquivalentSystem: {} ".format(code))
    equivalentSystems = list()
    #first row and column are fixed beacause the starting poitn must be always the same
    ncol = int(np.sqrt(len(code)))
    S= np.matrix(code).reshape( ncol,ncol )
    S = pd.DataFrame(S)

    columnas = S.columns.drop(0)
    permutations = list(itertools.permutations(columnas, ncol-1))

    for per in permutations:
        newIndexOrder = list(per)
        newIndexOrder.insert(0,0)
        S_b = S.loc[newIndexOrder, newIndexOrder]
        equiCode = getCode(S_b)
        equivalentSystems.append( "".join(equiCode) )

    return equivalentSystems


def getSystem(S, movement_node):
    """
    Delete the first column and the first row in order to remove one node from the system

    [a,a][a,b][a,c][a,d]
    [b,a] [][][]
    [b,c] [][][]
    [b,d] [][][]

    After this transformMatrix reorder ht matrix so the movement_node is at the column 0, 
    row 0. 

    """
    starting_node = S.columns[0]
    columnsMinusA = list(S.columns)
    columnsMinusA.remove(starting_node)
    S_minusA = S[columnsMinusA]
    S_minusA = S_minusA.loc[columnsMinusA]

    s_minusA_start = S_minusA.columns[0]
    return transformMatrix( S_minusA, s_minusA_start, movement_node)


def get_SOLUCIONES_size(SOLUCIONES):
    """
    Return the amount of solved and saved systems in SOLUCIONES
    """
    levels_list = SOLUCIONES.keys()
    cuenta = 0 
    for level in levels_list:
        code_list = list(SOLUCIONES[level].keys())
        cuenta += len(code_list)
    return cuenta

def transformMatrix(S, a, b ):
    """
    [a][B][C][b] => [b][B][C][a]

       [a][B][C][b]               [b][B][C][a]                 [b][B][C][a] 

    [a]  [a,a] [] [a,b]         [a]  [a,b] [a,] [a,a]        [b]  [b,b] [b,] [b,a]
    [B]  [a]   [] [b]   =>   [B]  [b]   []   [a]    =>    [B]  [b]   []   [a]   
    [C]  [a]   [] [b]        [C]  [b]   []   [a]          [C]  [b]   []   [a]   
    [b]  [b,a] [] [b,b]      [b]  [b,b] [b,] [b,a]        [a]  [a,b] [a,] [a,a]


    """
    order =  list(S.columns)
    aPosition = order.index(a)
    bPosition = order.index(b)
    #Changing columns order
    order[bPosition] = a
    order[aPosition] = b
    #Changing rows order

    newMatrix = S[order]
    newMatrix = newMatrix.loc[order]

    return newMatrix


####################################################################################################################

def solveSystem(S, SOLUCIONES, N_STATES, N_STATIONS, equivalent_systems= False, start_solutions = False):
    """
    Given a System S of NxN nodes. This function returns the optimal sequence of nodes that minimize the cost of 
    traveling through al the nodes.

    equivalent_systems: each time a system is solved this feature solve all the possible equivalent systems to be 
    solved using the same solution of the primary system. It just bassicaly calculate the equivalent systems
    ans assign the same solution in the SOLUCIONES dictionary.

    start_solutions: It solves and saves all the possible systems equal o lower to X = start_solutions.
    """

    if start_solutions:
        print("Start First level system generator")
        startSolutions( start_solutions,  SOLUCIONES, N_STATES, N_STATIONS)
        start_solutions = False
        print("Complete generator")

    codeS        = getCode(S)           #GeneticCode that represent's the system [A, B, A D np.nan ... ]
    solutionS    = getSolution( codeS, SOLUCIONES ) #False if system haven't been solved.
    starting_node = S.columns[0]         #
    nodesList    = list(S.columns)      # A, B , C , D
    movementList = dict()     # 0->NA,  1->B, 2->C, ... n->N

    #Print function so we ean diferenciate levels
    #level = len(S.columns)
    #tab = "\t"
    #for i in range(level):
    #    tab = tab + "\t"

    for i in range(len(nodesList)):
        if i != 0:
            movementList[ i ] =  nodesList[i]

    #print("\n\n Evaluando el Sistema:\n\n {} \n\n CODIGO {} ".format(S, codeS))

    if solutionS:
        optimal_mov = solutionS[0]
        min_cost = solutionS[1]
        return optimal_mov, min_cost

    else: #No existe solucion guaradda para code en SOLUCIONES
        if len( codeS ) <= 4: #El codigo corresponde a una matriz de 2X2 y solo hay una opciond e moviento
            #print("\n" + tab + "***Se ha llegado al minimo sistema***")
            optimal_mov =[ [0,1]  ]
            movement_node = movementList[1]
            min_cost = S.loc[starting_node, movement_node]
            #print(optimal_mov)

        else:
            min_cost = 10000000
            #print(movementList.keys())
            for mov in movementList.keys():
                movement_node = movementList[mov]
                #print("\n" +tab+ "Moviendo camion de {}  a {} ".format( starting_node, movement_node))
                #list(itertools.permutations([1, 2, 3]))
                mov_cost  = S.loc[starting_node, movement_node]

                #Delete the starting node and transform the system so movement_node -> starting_node
                SminiusMov = getSystem( S, movement_node)
                SminiusMov_movements, SminiusMov_cost= solveSystem(SminiusMov, SOLUCIONES, N_STATES, N_STATIONS, equivalent_systems)
                #print(SminiusMov_movements)

                total_cost = mov_cost + SminiusMov_cost

                #print("\n" +tab+ "Ruta evaluda: mov_cost {} total_cost {} ".format( mov_cost, total_cost))

                if total_cost < min_cost:
                    min_cost    = total_cost
                    optimal_mov = SminiusMov_movements +  [[0,mov]]
                    #print( "\n"+tab+"NUEVO MINIMO: {}   =>  {} xxxx {} ".format(min_cost, optimal_mov, SminiusMov_movements) )

        insertSolution( codeS, optimal_mov, min_cost, equivalent_systems, SOLUCIONES)
        return optimal_mov, min_cost




def startSolutions( level_top, SOLUCIONES, N_STATES, N_STATIONS ):
    """
    This funtion insert in the SOLUCIONES dictionary the solution for all possible
    Systems of level_top, solving firts all the problems for level_top-1.
    """

    level = int(np.sqrt(level_top))
    #print("LEVEL: {}".format(level))
    if( level_top != 4):   
        S = mapaAleatorio(level,level)
        previousLevel = np.power( level-1, 2)
        startSolutions(  previousLevel, SOLUCIONES, N_STATES, N_STATIONS) #es mas facil si resolvemos primero los niveles mas bajos.
    combinationsCode = list(itertools.combinations_with_replacement( range(1, N_STATES ), level_top-level))
    #print("SYSTEM: {}  => {}".format(level, combinationsCode))

    for elements in combinationsCode:
        cont  = 0 
        S = mapaAleatorio(level,level)

        for i in range(level):
            i_name = "estacion_" + str(i+1)
            for j in range(level):
                j_name = "estacion_" + str(j+1)
                if i ==j :
                    S.loc[i_name,j_name] = "X"
                else:
                    S.loc[i_name,j_name] = elements[cont]
                    cont += 1
        #print("Elements: {} \n\tS: {}".format(elements, S))
        solveSystem(S,  SOLUCIONES, N_STATES, N_STATIONS, equivalent_systems = True)
    tamano = sys.getsizeof(SOLUCIONES)
    print("Dictionary Size: {} \n\t {}".format( tamano , list(SOLUCIONES.keys()) ))


def solveSystem_bruteForce(S, SOLUCIONES, N_STATES, N_STATIONS):
    """
    Given a System S of NxN nodes. This function returns the optimal sequence of nodes that minimize the cost of 
    traveling through al the nodes using a brute force algorithm.
    """

    codeS        = getCode(S)           #GeneticCode that represent's the system [A, B, A D np.nan ... ]
    starting_node = S.columns[0]         #
    nodesList    = list(S.columns)      # A, B , C , D
    movementList = dict()     # 0->NA,  1->B, 2->C, ... n->N

    #Print function so we ean diferenciate levels
    #level = len(S.columns)
    #tab = "\t"
    #for i in range(level):
    #    tab = tab + "\t"

    for i in range(len(nodesList)):
        if i != 0:
            movementList[ i ] =  nodesList[i]
    #print("\n\n Evaluando el Sistema:\n\n {} \n\n CODIGO {} ".format(S, codeS))


    if len( codeS ) <= 4: #El codigo corresponde a una matriz de 2X2 y solo hay una opciond e moviento
        #print("\n" + tab + "***Se ha llegado al minimo sistema***")
        optimal_mov =[ [0,1]  ]
        movement_node = movementList[1]
        min_cost = S.loc[starting_node, movement_node]
        #print(optimal_mov)

    else:
        min_cost = 10000000
        for mov in movementList.keys():
            movement_node = movementList[mov]
            #print("\n" +tab+ "Moviendo camion de {}  a {} ".format( starting_node, movement_node))
            #list(itertools.permutations([1, 2, 3]))
            mov_cost  = S.loc[starting_node, movement_node]

            #Delete the starting node and transform the system so movement_node -> starting_node
            SminiusMov = getSystem( S, movement_node)
            SminiusMov_movements, SminiusMov_cost= solveSystem_bruteForce(SminiusMov, SOLUCIONES, N_STATES, N_STATIONS)
            total_cost = mov_cost + SminiusMov_cost

            #print("\n" +tab+ "Ruta evaluda: mov_cost {} total_cost {} ".format( mov_cost, total_cost))

            if total_cost < min_cost:
                min_cost    = total_cost
                optimal_mov = SminiusMov_movements +  [[0,mov]]
                #print( "\n"+tab+"NUEVO MINIMO: {}   =>  {} xxxx {} ".format(min_cost, optimal_mov, SminiusMov_movements) )

        #memory = False
        #insertSolution( codeS, optimal_mov, min_cost, memory, SOLUCIONES)
        return optimal_mov, min_cost


if __name__ == '__main__':

    SOLUCIONES = dict()

    columnas = ['A', 'B', 'C', 'D']
    S = pd.DataFrame( columns = columnas, index=  columnas)
    for i in columnas:
        for j in columnas:
            if i == j:
                S.loc[i, j] = 'X'
            else:    
                S.loc[i, j] = np.random.randint(1, 10)
    N_STATIONS =  5
    N_STATES =  5


    S = mapaAleatorio( N_STATIONS,N_STATES )
    code = getCode(S)

    seq, cost =  solveSystem(S,  SOLUCIONES, N_STATES, N_STATIONS, equivalent_systems =True, start_solutions = 9 )
    #seq, cost =  solveSystem(S, equivalent_systems =False, start_solutions = False, SOLUCIONES, N_STATES, N_STATIONS )
    #seq, cost =  solveSystem(S, equivalent_systems =True, start_solutions = False, SOLUCIONES, N_STATES, N_STATIONS )

    print(get_SOLUCIONES_size(SOLUCIONES))



