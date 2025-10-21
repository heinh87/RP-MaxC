#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


import networkx as nx
import time
from pymprog import *
import glob

from networkx.algorithms.connectivity import minimum_st_node_cut



class Log:
    def __init__(self):
        self.networkLog = None
        self.log = 'log.txt'
        

    def newNetworkLog(self, name):
        self.networkLog = name
        with open(self.networkLog, "w") as _:
            pass  # just create/clear the file

    def print(self, s):
        print(s)
        for file in [self.networkLog, self.log]:
            with open(file, "a") as f:
                f.write(str(s) + '\n')

    def printResults(self, network):
        """
        Prints the results stored in the network object in a structured way.
        
        Args:
            network: Network object with results attribute (dictionary) containing
                     tuples (value, label, description) for each metric
        """
        if not hasattr(network, 'results') or network.results is None:
            self.print("Warning: Network has no results to print.")
            return
        
        self.print("\n" + "="*60)
        self.print("OPTIMIZATION RESULTS")
        self.print("="*60)
        
        # Iterate over results and print each metric
        for key, data in network.results.items():
            if len(data) == 3:
                value, label, description = data
                self.print(f"\n{label}: {value}")
                self.print(f"  Description: {description}")
            else:
                # Fallback if format is different
                self.print(f"\n{key}: {data}")
        
        self.print("\n" + "="*60 + "\n")



log = Log()


class Network:
    """
    Represents a network/graph for server location problem analysis.
    
    This class encapsulates a graph and its structural properties, including
    connectivity metrics (kappa), distances between nodes, and information about
    clients and potential servers.
    
    Attributes:
        fileRead (str): Path of the currently loaded graph source file
        G (nx.Graph): NetworkX graph object
        dist (dict[int, dict[int, int]]): Minimum distances between all pairs of nodes
            in format {i: {j: distance(i,j)}}
        C (list[int]): List of nodes that are clients
        S (list[int]): List of candidate server nodes
        kappa (dict[int, dict[int, int]]): Connectivity (node-connectivity) between pairs
            of nodes {i: {j: kappa(i,j)}}
        kappa2 (dict[int, int]): Maximum connectivity of each node {i: max_j kappa(i,j)}
        maxKappa2 (int): Largest kappa2 value in the graph
        avgKappa2 (float): Average of kappa2 values in the graph

    
    Methods:
        readDimacsGraph(file): Load graph from DIMACS format file
        updateLog(): Update log file with network information
        loadAndValidateGraph(name): Load and validate graph from GML or DIMACS file
        analyzeGraph(): Calculate kappas, distances and graph statistics
        calcGraphKappa(): Calculate connectivities (kappa and kappa2) for all nodes
    """
    

    def __init__(self):
        self.fileRead = None

        self.G = None

        self.dist = {}  # shortest distances between all pairs (iterator -> dict)
        self.C = []  # clients
        self.S = []  # servers

        self.kappa = {}              # kappa[i][j]: connectivity (node-connectivity) between i and j
        self.kappa2 = {}             # kappa2[i]: maximum connectivity of node i
        self.maxKappa2 = None        # largest observed kappa2 value in the graph
        self.avgKappa2 = None        # average of kappa2 values in the graph

        self.results = None

    def __str__(self):
        pass


    def readDimacsGraph(self, file):
        f = open(file, "r")
        G = nx.Graph()
        for line in f:
            x = line.split()
            if x[0] == 'a':
                G.add_edge(x[1], x[2])
        f.close()    
        
        self.G = G

    def updateLog(self):
        name = self.fileRead.split('.')    
        prefix = name[0]
        suffix = name[1]

        log.newNetworkLog(prefix + "_" + str(len(self.G.nodes())) + "_resul.txt")
        log.print(' ')
        if suffix == 'gml':
            log.print(str(self.G.graph['Network']))



    def loadAndValidateGraph(self, name):
        """
        Loads a graph from GML or DIMACS file and validates if it is connected and undirected.
        
        The method identifies the file format by extension (.gml or .dimacs),
        loads the graph, validates its properties (connectivity and undirected),
        and converts node labels to sequential integers.
        
        Args:
            name: string, file path (can be .gml or .dimacs)
            
        Side effects:
            Updates self.fileRead with the file name
            Updates self.G with the loaded and processed graph
            Converts node labels to integers (0, 1, 2, ...)
            Preserves original labels in 'nome' attribute of each node
            
        Raises:
            SystemExit: if the graph is not connected or is directed
        """
        self.fileRead = name

        fileName = name.split('.')    
        suffix = fileName[1]
        
        if suffix == 'gml':
            self.G = nx.read_gml(name, label='id')

        elif suffix == 'dimacs':
            self.G = self.readDimacsGraph(name)

        if not nx.is_connected(self.G):
            print("Graph not connected")
            exit()

        if nx.is_directed(self.G):
            print("Graph is directed")
            exit()
        
        # to simplify the code, transform labels to integers and store them in name
        if len(nx.get_node_attributes(self.G, 'nome')) == 0:
            # In recent versions, convert_node_labels_to_integers signature may not accept 'ordering'
            try:
                self.G = nx.convert_node_labels_to_integers(self.G, ordering="increasing degree", label_attribute='nome')
            except TypeError:
                self.G = nx.convert_node_labels_to_integers(self.G, label_attribute='nome')
        
        # Update log after graph is loaded
        self.updateLog()  






    def analyzeGraph(self):
        """
        Calculates kappas and minimum distances between all pairs of nodes in the graph.
        
        This function performs complete connectivity and distance analysis on the graph,
        calculating and storing:
        - Degree of each node as attribute
        - Connectivity (kappa) between all pairs of nodes
        - Maximum connectivity (kappa2) of each node
        - Minimum distances between all pairs of nodes
        - Statistics (largest kappa2 and average kappa2)
        
        Updates attributes:
            self.kappa: dict, connectivity between pairs of nodes
            self.kappa2: dict, maximum connectivity of each node
            self.dist: dict, minimum distances between pairs of nodes
            self.maxKappa2: int, largest kappa2 value in the graph
            self.avgKappa2: float, average of kappa2 values
            
        Side effects:
            - Prints progress messages to console
            - Logs statistics via log.print()
        """

        # 'degree' attribute for each node (NetworkX API >= 2.x)
        nx.set_node_attributes(self.G, dict(self.G.degree()), 'degree')

        self.calcGraphKappa()        
        print("Calculated kappas")
        
        # shortest distances between all pairs (iterator -> dict)
        self.dist = dict(nx.all_pairs_dijkstra_path_length(self.G))
        print("Calculated shortest distances")

        log.print('nVert ' + str(len(self.G.nodes())) + ' nEdges ' + str(len(self.G.edges())))
 
        maxKappa = 0
        avgKappa = 0
        for i in nx.nodes(self.G):
            if self.kappa2[i] > maxKappa:
                maxKappa = self.kappa2[i]
            avgKappa = avgKappa + self.kappa2[i]
        avgKappa = round(avgKappa / len(self.G.nodes()), 2)
        
        log.print('Max kappa2 = ' + str(maxKappa))
        log.print('Avg kappa2 = ' + str(avgKappa))

        self.maxKappa2 = maxKappa
        self.avgKappa2 = avgKappa



    def calcGraphKappa(self):
        kappa = self.kappa
        kappa2 = self.kappa2

        # initialization of dictionary vectors
        for s in nx.nodes(self.G):
            kappa[s] = {}
            kappa2[s] = 0

        
        # calculate kappas    
        for s in nx.nodes(self.G):
            for t in nx.nodes(self.G):
                if s != t:
                    if t not in kappa[s]:
                        kappa[s][t] = nx.node_connectivity(self.G, s, t)
                        kappa[t][s] = kappa[s][t]  # working only with undirected graphs
                    if kappa[s][t] > kappa2[s]:
                        kappa2[s] = kappa[s][t]
            kappa[s][s] = kappa2[s]

        # assign 'kappa' attribute to nodes (NetworkX API >= 2.x)
        nx.set_node_attributes(self.G, kappa2, 'kappa')

        self.kappa = kappa
        self.kappa2 = kappa2

    
        

def linearProgram(problemType, S, C, dist, kappa, kappa2, p=None, sumDist=None):
    """
    Solves server allocation problem using integer linear programming.
    
    Implements different variants of facility location problem:
    - maxConnect: maximize connectivity (kappa) between clients and servers
    - p-median: minimize sum of distances with p fixed servers
    
    Args:
        problemType: string indicating the problem type to solve:
            - "maxConnect, minimum number of servers": minimize number of servers
            - "maxConnect, minimum sum of distances": minimize distances with maximum connectivity
            - "maxConnect, maximum sum of distances": maximize distances with maximum connectivity
            - "p-median, minimum sum of distances": classic p-median problem
            - "p-median, maximum connectivity": maximize connectivity for p-median
            - "p-median, minimum connectivity": minimize connectivity for p-median
        S: list of candidate server nodes
        C: list of client nodes
        dist: dictionary dist[i][j] with distances between nodes i and j
        kappa: dictionary kappa[i][j] with connectivity (node-connectivity) between i and j
        kappa2: dictionary kappa2[j] with maximum connectivity of node j
        p: number of servers (used in some problem types, None for others)
        sumDist: constraint on maximum sum of distances (used in p-median with connectivity)
    
    Returns:
        dict R with:
            R[0]: objective function value
            R[1]: list of nodes chosen as servers
            R[2]: dictionary {server: [clients]} mapping each server to its clients
            R[3]: execution time in seconds
    """

    begin('model')  # begin modelling
    # verbose(True)  # be verbose

    A = iprod(S, C)  # Cartesian product
    x = var('x', A, bool)
    y = var('y', S, bool)
    
    
    # x[i,j] client at j connected to server at i, if equal to 1
    # y[i] server at i, if equal to 1

    match problemType:
        case "maxConnect, minimum number of servers":
            # minimize number of servers
            minimize(sum(y[i] for i in S))

            # constraint: client j is connected to server i
            # kappa(j,i)=kappa2(j)
            
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]


        case "maxConnect, minimum sum of distances":
            # p-median
            minimize(sum(dist[i][j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p

            # constraint: client j is connected to server i
            # kappa(j,i)=kappa2(j)
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]
    
        case "maxConnect, maximum sum of distances":
            # p-median
            maximize(sum(dist[i][j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p

            # constraint: client j is connected to server i
            # kappa(j,i)=kappa2(j)
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]
    
        case "p-median, minimum sum of distances":
            # p-median
            minimize(sum(dist[i][j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p
    
        case "p-median, maximum connectivity":
            minimize(sum(kappa2[j]*x[i,j]-kappa[j][i]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p
            sum(dist[i][j]*x[i,j] for i,j in A) <= sumDist
        
        case "p-median, minimum connectivity":
            maximize(sum(kappa2[j]*x[i,j]-kappa[j][i]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p
            sum(dist[i][j]*x[i,j] for i,j in A) <= sumDist

        
    # additional constraints    
    # there is a server at i, if client at j is connected to i
    for i,j in A:
        y[i] >= x[i,j] 

    # client at j is connected to only one server
    # if there is a server at i then it is connected to itself x[i,i] = 1
   
    for j in C:
        sum(x[i,j] for i in S) == 1

    t0 = time.time()
    solve()  # solve the model
    execTime = time.time() - t0


    minSum = vobj()
    servers = []
    connections = {}
    
    for i in S:
        if y[i].primal == 1:
            servers.append(i)
            connections[i] = []
            for j in C:
                if x[i,j].primal == 1:
                    connections[i].append(j)            
    R = {}
    R[0] = minSum  # objective function value
    R[1] = servers  # chosen servers
    R[2] = connections  # clients connected to servers
    R[3] = execTime  # execution time

    end() 
   
    return R        



def min_servers_max_connect(network, calcTime = False):
    """
    Executes case 1 (maxConnect: minimum number of servers) repeated N times,
    prints the results and returns p (minimum number of servers) and average time.

    Args:
        network: Network object containing graph, clients (C), servers (S), 
                 distances (dist) and connectivity structures (kappa, kappa2)
        calcTime: if True, runs 4 repetitions to calculate average time;
                  if False, runs only once (default: True)

    Returns:
        (p, pTime):
            p -> integer, minimum number of servers
            pTime -> float, average time in seconds (or time of 1 execution if calcTime=False)
    """

    if calcTime:
        repetitions = 4
    else:
        repetitions = 1  

    problemType = "maxConnect, minimum number of servers"

    time = 0.0
    for _ in range(repetitions):
        R = linearProgram(problemType, network.S, network.C, network.dist, network.kappa, network.kappa2)
        print(R[3])
        time += R[3]

    pTime = round(time / repetitions, 1)

    log.print(problemType)
    log.print(R[0])  # minimum number of servers
    log.print(R[1])  # the servers
    log.print(R[2])  # the clients assigned to servers

    p = round(R[0])

    if not calcTime:
        return p
    else:
        return p, pTime


def max_connect_min_sum_distances(network, p, calcTime = False):
    """
    Executes case 2 (maxConnect: minimum sum of distances) repeated N times,
    prints the results and returns the sum of distances.

    Args:
        network: Network object containing graph, clients (C), servers (S), 
                 distances (dist) and connectivity structures (kappa, kappa2)
        p: number of servers (fixed by case 1)
        calcTime: if True, runs 4 repetitions to calculate average time;
                  if False, runs only once (default: False)

    Returns:
        If calcTime = True:
            (sumDistMaxConnect, distMaxConnectTime): tuple with sum of distances and average time
        If calcTime = False:
            sumDistMaxConnect: int, rounded objective value (sum of distances)
    """

    
    if calcTime:
        repetitions = 4
    else:
        repetitions = 1

    problemType = "maxConnect, minimum sum of distances"

    time = 0.0
    for _ in range(repetitions):
        R = linearProgram(problemType, network.S, network.C, network.dist, network.kappa, network.kappa2, p)
        time += R[3]
    distMaxConnectTime = round(time / repetitions, 2)

    log.print(problemType)
    log.print(R[0])  # objective (sum of distances)
    log.print(R[1])  # servers
    log.print(R[2])  # clients assigned to servers

    sumDistMaxConnect = round(R[0])

    if not calcTime:
        return sumDistMaxConnect
    else:
        return sumDistMaxConnect, distMaxConnectTime


def max_connect_max_sum_distances(network, p, calcTime = False):
    """
    Executes case 2.5 (maxConnect: MAXIMUM sum of distances) repeated N times,
    prints the results and returns the worst sum of distances.

    Args:
        network: Network object containing graph, clients (C), servers (S), 
                 distances (dist) and connectivity structures (kappa, kappa2)
        p: number of servers (fixed by case 1)
        calcTime: if True, runs 4 repetitions to calculate average time;
                  if False, runs only once (default: False)

    Returns:
        If calcTime = True:
            (worstSumDistMaxConnect, worstDistMaxConnectTime): tuple with worst sum of distances and average time
        If calcTime = False:
            worstSumDistMaxConnect: int, rounded objective value (worst sum of distances)
    """


    if calcTime:
        repetitions = 4
    else:
        repetitions = 1

    problemType = "maxConnect, maximum sum of distances"

    time = 0.0
    for _ in range(repetitions):
        R = linearProgram(problemType, network.S, network.C, network.dist, network.kappa, network.kappa2, p)
        time += R[3]
    worstDistMaxConnectTime = round(time / repetitions, 2)

    log.print(problemType)
    log.print(R[0])  # objective (sum of distances)
    log.print(R[1])  # servers
    log.print(R[2])  # clients assigned to servers

    worstSumDistMaxConnect = round(R[0])

    if not calcTime:
        return worstSumDistMaxConnect
    else:
        return worstSumDistMaxConnect, worstDistMaxConnectTime

def p_median_min_sum_distances(network, p, calcTime = False):
    """
    Executes case 3 (p-median: minimum sum of distances) repeated N times,
    prints the results and returns the sum of distances.

    Args:
        network: Network object containing graph, clients (C), servers (S), 
                 distances (dist) and connectivity structures (kappa, kappa2)
        p: number of servers
        calcTime: if True, runs 4 repetitions to calculate average time;
                  if False, runs only once (default: False)

    Returns:
        If calcTime = True:
            (sumDistPmedian, sumDistPmedianTime): tuple with sum of distances and average time
        If calcTime = False:
            sumDistPmedian: int, rounded objective value (sum of distances)
    """

    if calcTime:
        repetitions = 4
    else:
        repetitions = 1

    problemType = "p-median, minimum sum of distances"
    time = 0.0
    R = None
    for _ in range(repetitions):
        R = linearProgram(problemType, network.S, network.C, network.dist, network.kappa, network.kappa2, p)
        time += R[3]
    sumDistPmedianTime = round(time / repetitions, 2)

    log.print(problemType)
    log.print(R[0])  # objective (sum of distances)
    log.print(R[1])  # servers
    log.print(R[2])  # clients assigned to servers

    sumDistPmedian = round(R[0])

    if not calcTime:
        return sumDistPmedian
    else:
        return sumDistPmedian, sumDistPmedianTime


def p_median_max_connectivity(network, p, sumDistPmedian, calcTime = False):
    """
    Executes case 4 (p-median: maximum connectivity), prints the results 
    and returns the sum of gaps and client-server connections.

    Args:
        network: Network object containing graph, clients (C), servers (S), 
                 distances (dist) and connectivity structures (kappa, kappa2)
        p: number of servers
        sumDistPmedian: constraint value for sum of distances
        calcTime: if True, runs 4 repetitions to calculate average time;
                  if False, runs only once (default: False)

    Returns:
        If calcTime = True:
            (sumDiff, clientServerConn, sumDiffTime): tuple with sum of gaps, 
            connections dictionary and average time
        If calcTime = False:
            (sumDiff, clientServerConn): tuple with sum of gaps and connections dictionary
                sumDiff -> int, rounded objective value (sum of connectivity gaps)
                clientServerConn -> dict {server: [clients]}, mapping of clients per server
    """
    if calcTime:
        repetitions = 4
    else:
        repetitions = 1

    problemType = "p-median, maximum connectivity"

    time = 0.0
    for _ in range(repetitions):
        R = linearProgram(problemType, network.S, network.C, network.dist, network.kappa, network.kappa2, p, sumDistPmedian)
        time += R[3]
    sumDiffTime = round(time / repetitions, 2)

    log.print(problemType)
    log.print(R[0])  # objective (sum of distances)
    log.print(R[1])  # servers
    log.print(R[2])  # clients assigned to servers
    
    sumDiff = round(R[0])
    
    if not calcTime:
        return sumDiff, R[2]
    else:
        return sumDiff, R[2], sumDiffTime

def analyzeGaps(network, clientServerConn):
    """
    Analyzes connectivity gaps between clients and their assigned servers.
    
    A gap occurs when the connectivity between a client and its assigned server
    is less than the maximum connectivity of that client (kappa2). The function calculates
    statistics about these gaps.

    Args:
        network: Network object containing connectivity structures (kappa, kappa2)
        clientServerConn: dictionary {server: [clients]} with client-to-server assignment

    Returns:
        (numClientGap, maxGap, kappaMaxGap, avgGap):
            numClientGap -> int, number of clients with connectivity gap
            maxGap -> int, value of the largest gap found
            kappaMaxGap -> int, kappa2 value of the client with the largest gap
            avgGap -> float, average of relative gaps (gap/kappa2) rounded to 2 decimals
    """
    

    kappa = network.kappa
    kappa2 = network.kappa2

    numClientGap = 0
    maxGap = 0
    kappaMaxGap = 0
    avgGap = 0
    for i in clientServerConn:
        for j in clientServerConn[i]:
            if kappa[i][j] != kappa2[j]:
                numClientGap = numClientGap + 1
                gap = kappa2[j] - kappa[i][j]
                log.print('kappa2 = ' + str(kappa2[j]) + ' gap = ' + str(gap))
                avgGap = avgGap + gap/kappa2[j]
                if gap >= maxGap:
                    maxGap = gap
                    if kappa2[j] > kappaMaxGap:
                        kappaMaxGap = kappa2[j]
    if numClientGap != 0:
        avgGap = round(avgGap/numClientGap, 2)


    return numClientGap, maxGap, kappaMaxGap, avgGap



            
class Output():
    """
    Manages generation of outputs and formatted result tables.
    
    Attributes:
        tableMaxC (dict): Nested structure {numVert: {numEdges: {graphName: tableLine}}}
        tableFileMaxC (str): Output file path for MaxC table
        tablePmedian (dict): Nested structure {numVert: {numEdges: {graphName: tableLine}}}
        tableFilePmedian (str): Output file path for p-median table
    """
    
    def __init__(self):
        self.tableMaxC = {}
        self.tableFileMaxC = 'tableMaxC.tex'
        self.tablePmedian = {}
        self.tableFilePmedian = 'tablePmedian.tex'
        self.tableRatioMaxCpmedian = {}
        self.tableFileRatioMaxCpmedian = 'tableRatioMaxCpmedian.tex'

    def addRatioMaxCpmedianTableLine(self, network):
        """
        Adds a formatted line to the experimental results table Ratio between the minimum
        sum of distances with maximum connectivity and the sum of distances of the $p$-median problem
        as the number of resources increases until 6 more.
        
        Table columns (in order):
        1. Network name and year
        2. p — minimum number of servers for RP-MaxC problem
        3. p+1
        4. p+2
        5. p+3
        6. p+4
        7. p+5
        8. p+6
        """
        numVert = len(network.G.nodes())
        numEdges = len(network.G.edges())
        
        # Extract file information
        fileName = network.fileRead.split('.')
        suffix = fileName[1]
        
        if suffix == 'gml':
            networkTableName = str(network.G.graph.get('Network', 'Unknown'))
            networkYear = str(network.G.graph.get('DateYear', ''))
            graphName = networkTableName
        else:
            networkTableName = fileName[0].split('/')[-1]  # Remove path
            networkYear = ''
            graphName = networkTableName
        
        # Extract results from dictionary
        if hasattr(network, 'results') and network.results:
            ratios = network.results['ratioMaxCpmedian']  # list of ratios for p to p+5
        else:
            raise ValueError("Network has no results to generate table row")
        
        # Build LaTeX formatted line following documentation order:
        # Network | p | p+1 | p+2 | p+3 | p+4 | p+5
        line = (
            f"{networkTableName} {networkYear}"
            f" & {ratios[0]}"
            f" & {ratios[1]}"
            f" & {ratios[2]}"
            f" & {ratios[3]}"
            f" & {ratios[4]}"
            f" & {ratios[5]}"
            f" & {ratios[6]}"
            " \\\\"
        )
        
        # Store in hierarchical structure
        if numVert not in self.tableRatioMaxCpmedian:
            self.tableRatioMaxCpmedian[numVert] = {}
        if numEdges not in self.tableRatioMaxCpmedian[numVert]:
            self.tableRatioMaxCpmedian[numVert][numEdges] = {}
        
        self.tableRatioMaxCpmedian[numVert][numEdges][graphName] = line
        
        return line


    def addMaxCTableLine(self, network):
        """
        Adds a formatted line to the experimental results table of RP-MaxC problem.
        
        Table columns (in order):
        1. Network name and year
        2. |V| — number of vertices in the graph
        3. |A| — number of edges in the graph
        4. Max κ₂(v) — maximum κ₂(v) of the graph
        5. p — minimum number of servers for RP-MaxC problem
        6. p/|V| — ratio between minimum number of servers and vertices
        7. t (s) — execution time in seconds to calculate p
        8. Max Σ dist — maximum sum of distances for RP-MaxC
        9. Min Σ dist — minimum sum of distances for RP-MaxC
        10. p-median Σ dist — sum of distances for p-median problem
        11. Max ratio — ratio between maximum sum RP-MaxC and p-median
        12. Min ratio — ratio between minimum sum RP-MaxC and p-median
        
        Args:
            network: Network object with loaded graph and calculated results
            
        Returns:
            str: formatted line in LaTeX for insertion in the table
        """
        numVert = len(network.G.nodes())
        numEdges = len(network.G.edges())
        
        # Extract file information
        fileName = network.fileRead.split('.')
        suffix = fileName[1]
        
        if suffix == 'gml':
            networkTableName = str(network.G.graph.get('Network', 'Unknown'))
            networkYear = str(network.G.graph.get('DateYear', ''))
            graphName = networkTableName
        else:
            networkTableName = fileName[0].split('/')[-1]  # Remove path
            networkYear = ''
            graphName = networkTableName
        
        # Extract results from dictionary
        if hasattr(network, 'results') and network.results:
            p = network.results['p'][0]
            pTime = network.results['pTime'][0]
            sumDistMaxConnect = network.results['sumDistMaxConnect'][0]
            worstSumDistMaxConnect = network.results['worstSumDistMaxConnect'][0]
            sumDistPmedian = network.results['sumDistPmedian'][0]
            distRatio = network.results['distRatio'][0]
            worstDistRatio = network.results['worstDistRatio'][0]
        else:
            raise ValueError("Network has no results to generate table row")
        
        # Build LaTeX formatted line following documentation order:
        # Network | |V| | |A| | Max κ₂ | p | p/|V| | t(s) | Max Σ | Min Σ | p-med Σ | Max ratio | Min ratio
        line = (
            f"{networkTableName} {networkYear}"
            f" & {numVert}"
            f" & {numEdges}"
            f" & {network.maxKappa2}"
            f" & {p}"
            f" & {round(p/numVert, 2)}"
            f" & {pTime}"
            f" & {worstSumDistMaxConnect}"
            f" & {sumDistMaxConnect}"
            f" & {sumDistPmedian}"
            f" & {worstDistRatio}"
            f" & {distRatio}"
            " \\\\"
        )
        
        # Store in hierarchical structure
        if numVert not in self.tableMaxC:
            self.tableMaxC[numVert] = {}
        if numEdges not in self.tableMaxC[numVert]:
            self.tableMaxC[numVert][numEdges] = {}
        
        self.tableMaxC[numVert][numEdges][graphName] = line
        
        return line
    
    def addPmedianTableLine(self, network):
        """
        Adds a formatted line to the p-median experimental results table.
        
        Experimental results to the $p$-median problem and comparisons with $\kappa_2$. The acronyms in the table are expanded as follows:  $|V|$ — number of vertices in the graph; $|A|$ — number of edges in the graph; Max $\kappa_2(v)$ — maximum $\kappa_2(v)$ of the graph; $p$ — minimum number of servers for the RP-MaxC problem; Num clients — number of clients of $p$-median that do not attend the maximum connectivity restraint; Max diff($c$) — the largest difference between the number of vertex-disjoint paths of a client with the server and the $\kappa_2$ of this client; Max $\kappa_2(c)$ — the maximum $\kappa_2$ of a client with the largest difference; Avg diff — the ratio between the sum of the differences and the number of clients that have a difference.
        
        Args:
            network: Network object with loaded graph and calculated results
            
        Returns:
            str: formatted line in LaTeX for insertion in the table
        """
        numVert = len(network.G.nodes())
        numEdges = len(network.G.edges())
        
        # Extract file information
        fileName = network.fileRead.split('.')
        suffix = fileName[1]
        
        if suffix == 'gml':
            networkTableName = str(network.G.graph.get('Network', 'Unknown'))
            networkYear = str(network.G.graph.get('DateYear', ''))
            graphName = networkTableName
        else:
            networkTableName = fileName[0].split('/')[-1]  # Remove path
            networkYear = ''
            graphName = networkTableName
        
        # Extract results from dictionary
        if hasattr(network, 'results') and network.results:
            p = network.results['p'][0]
            numClientGap = network.results['numClientGap'][0]
            maxGap = network.results['maxGap'][0]
            kappaMaxGap = network.results['kappaMaxGap'][0]
            avgGap = network.results['avgGap'][0]
        else:
            raise ValueError("Network has no results to generate table row")
        
        # Build LaTeX formatted line following documentation order:
        # Network | |V| | |A| | Max κ₂ | p | Num clients | Max diff(c) | Max κ₂(c) | Avg diff
        line = (
            f"{networkTableName} {networkYear}"
            f" & {numVert}"
            f" & {numEdges}"
            f" & {network.maxKappa2}"
            f" & {p}"
            f" & {numClientGap}"
            f" & {maxGap}"
            f" & {kappaMaxGap}"
            f" & {avgGap}"
            " \\\\"
        )
        
        # Store in hierarchical structure
        if numVert not in self.tablePmedian:
            self.tablePmedian[numVert] = {}
        if numEdges not in self.tablePmedian[numVert]:
            self.tablePmedian[numVert][numEdges] = {}
        
        self.tablePmedian[numVert][numEdges][graphName] = line

        return line

    def saveTables(self):
        """
        Saves the complete table to file sorted by vertices and edges.
        
        Args:
            file: output file name (default: 'tabela.txt')
        """

        tables = (self.tableMaxC, self.tableFileMaxC, self.tableRatioMaxCpmedian), (self.tablePmedian, self.tableFilePmedian, self.tableFileRatioMaxCpmedian)

        for table, file in tables:
            with open(file, "w") as tab:
                for numVert in sorted(table.keys()):
                    for numEdges in sorted(table[numVert].keys()):
                        for graphName in sorted(table[numVert][numEdges].keys()):
                            tab.write(table[numVert][numEdges][graphName] + '\n')
        

    
    def __str__(self):
        """Returns textual representation of tables."""
        tables = (self.tableMaxC, self.tableFileMaxC, self.tableRatioMaxCpmedian), (self.tablePmedian, self.tableFilePmedian, self.tableFileRatioMaxCpmedian)

        lines = []
        for table, file in tables:
            for numVert in sorted(table.keys()):
                for numEdges in sorted(table[numVert].keys()):
                    for graphName in sorted(table[numVert][numEdges].keys()):
                        lines.append(table[numVert][numEdges][graphName])
        return '\n'.join(lines)




#MAIN
def main():
    fileList = sorted(glob.glob("archive/*.gml"))
    output = Output()

    for name in fileList:

        print(name)

        network = Network()
        network.loadAndValidateGraph(name)


        network.C = sorted(network.G.nodes())  # clients
        network.S = list(network.C)  # servers

        ## calculate kappas and shortest distances between vertices
        network.analyzeGraph()


        # INTEGER LINEAR PROGRAMMING RESOLUTION
    
        # ---MaxConnect---       
        # find minimum number of servers for maximum connectivity
        p, pTime = min_servers_max_connect(network, calcTime=True)

        
        # minimizing sum of distances
        sumDistMaxConnect = max_connect_min_sum_distances(network, p)

        
        # maximizing sum of distances
        worstSumDistMaxConnect = max_connect_max_sum_distances(network, p) 
        
        
        # ---p-median---
        # get sum of distances for p-median        
        sumDistPmedian = p_median_min_sum_distances(network, p)

        # Ratio between the minimum sum of distances with maximum connectivity and the sum of distances of the $p$-median problem 
        # as the number of resources increases until 6 more
        listRatioDistMaxC_pmedian = [round(sumDistMaxConnect/sumDistPmedian, 2)]
        for i in range(1,7):
            a = max_connect_min_sum_distances(network, p + i)
            b = p_median_min_sum_distances(network, p + i)
            listRatioDistMaxC_pmedian.append(round(a/b, 2))
        

        # maximizing connectivity or minimizing gaps
        sumGapsPmedian, clientServerConn = p_median_max_connectivity(network, p, sumDistPmedian)

        # analysis ---MaxConnect---    

        distRatio = round(sumDistMaxConnect/sumDistPmedian, 2)
        worstDistRatio = round(worstSumDistMaxConnect/sumDistPmedian, 2)

        # analysis ---p-median---       

        numClientGap, maxGap, kappaMaxGap, avgGap = analyzeGaps(network, clientServerConn)


        # STORE RESULTS IN NETWORK
        network.results = {
            'p': (p, '$p$', "Minimum number of servers for the RP-MaxC problem."),
            'pTime': (pTime, "$t(s)$", "Running time in seconds to calculate $p$"),
            'ratioMaxCpmedian': listRatioDistMaxC_pmedian,
            'listRatioDistMaxC_pmedian': (listRatioDistMaxC_pmedian, "Ratio", "Ratio between the distances for the RP-MaxC and the $p$-median as the number of resources increases until 6 more."),
            'sumDistMaxConnect': (sumDistMaxConnect, "RP-MaxC max ($\\sum$ dist)", " Maximum sum of distances for the RP-MaxC problem."),
            'worstSumDistMaxConnect': (worstSumDistMaxConnect, "RP-MaxC min ($\\sum$ dist)", " Minimum sum of distances for the RP-MaxC problem."),
            'sumDistPmedian': (sumDistPmedian, "$p$-median $\\sum$ dist", " Sum of distances for the $p$-median problem."),
            'sumGapsPmedian': (sumGapsPmedian, "$\\sum$diff", "Minimum sum of the differences between the number of vertex-disjoints paths of the clients of $p$-median and the  $\\kappa_2$ of these clients."),
            'distRatio': (distRatio, "Min ratio", " Ratio between the minimum sum of distances for the RP-MaxC and the sum for the $p$-median."),
            'worstDistRatio': (worstDistRatio, "Max ratio", " Ratio between the maximum sum of distances for the RP-MaxC and the sum for the $p$-median."),
            'numClientGap': (numClientGap, "Num clients with diff", "Number of clients of $p$-median that do not attend the maximum connectivity restraint."),
            'maxGap': (maxGap, "Max diff", " Maximum difference between the number of vertex-disjoint paths of a client with the server and the $\\kappa_2$ of this client."),
            'kappaMaxGap': (kappaMaxGap, "$\\kappa_2$ with max diff", "Maximum $\\kappa_2$ of a client with the largest difference."),
            'avgGap': (avgGap, "Avg diff", " Average difference between the number of vertex-disjoint paths of the clients of $p$-median and the $\\kappa_2$ of these clients.")
        }

        log.printResults(network)

        # Add row to table using Output class
        tableLine = output.addMaxCTableLine(network)
        log.print(tableLine)
        ratioTableLine = output.addRatioMaxCpmedianTableLine(network)
        log.print(ratioTableLine)
        pmedianTableLine = output.addPmedianTableLine(network)
        log.print(pmedianTableLine)


    # Save complete table at the end
    output.saveTables()

if __name__ == "__main__":
    main()
