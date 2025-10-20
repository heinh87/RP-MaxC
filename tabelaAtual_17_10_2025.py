#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:44:53 2017
updated on OCt 16 2025 

@author: Henrique
"""

import math
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import time
import random
import collections
from pymprog import *
import glob
import shutil
from operator import itemgetter, attrgetter

from networkx.algorithms.connectivity import minimum_st_node_cut



class Log:
    def __init__(self):
        self.logRede = None
        self.log = 'log.txt'
        

    def novoLogRede(self, nome):
        self.logRede = nome
        with open(self.logRede, "w") as _:
            pass  # apenas cria/limpa o arquivo

    def imprime(self,s):
        print(s)
        for arquivo in [self.logRede, self.log]:
            with open(arquivo, "a") as f:
                f.write(str(s) + '\n')

    def imprimeResultados(self, rede):
        """
        Imprime os resultados armazenados no objeto rede de forma estruturada.
        
        Args:
            rede: objeto Rede com atributo resultados (dicionário) contendo
                  tuplas (valor, label, descrição) para cada métrica
        """
        if not hasattr(rede, 'resultados') or rede.resultados is None:
            self.imprime("Aviso: Rede não possui resultados para imprimir.")
            return
        
        self.imprime("\n" + "="*60)
        self.imprime("RESULTADOS DA OTIMIZAÇÃO")
        self.imprime("="*60)
        
        # Itera sobre os resultados e imprime cada métrica
        for chave, dados in rede.resultados.items():
            if len(dados) == 3:
                valor, label, descricao = dados
                self.imprime(f"\n{label}: {valor}")
                self.imprime(f"  Descrição: {descricao}")
            else:
                # Fallback se o formato for diferente
                self.imprime(f"\n{chave}: {dados}")
        
        self.imprime("\n" + "="*60 + "\n")



log = Log()


class Rede:
    """
    Representa uma rede/grafo para análise de problemas de localização de servidores.
    
    Esta classe encapsula um grafo e suas propriedades estruturais, incluindo
    métricas de conectividade (kappa), distâncias entre nós, e informações sobre
    clientes e servidores potenciais.
    
    Attributes:
        arquivoLido (str): Caminho do arquivo de origem do grafo atualmente carregado
        G (nx.Graph): Objeto grafo do NetworkX
        dis (dict[int, dict[int, int]]): Distâncias mínimas entre todos os pares de nós
            no formato {i: {j: distancia(i,j)}}
        C (list[int]): Lista de nós que são clientes
        S (list[int]): Lista de nós candidatos a servidor
        kappa (dict[int, dict[int, int]]): Conectividade (nó-conectividade) entre pares
            de nós {i: {j: kappa(i,j)}}
        kappa2 (dict[int, int]): Conectividade máxima de cada nó {i: max_j kappa(i,j)}
        maiorKappa2 (int): Maior valor de kappa2 no grafo
        kappa2Medio (float): Média dos valores de kappa2 no grafo

    
    Methods:
        leGrafoDimacs(arquivo): Carrega grafo de arquivo formato DIMACS
        atualizaLog(): Atualiza arquivo de log com informações da rede
        carregaEValidaGrafo(name): Carrega e valida grafo de arquivo GML ou DIMACS
        analisa_grafo(): Calcula kappas, distâncias e estatísticas do grafo
        calcKappaGrafo(): Calcula conectividades (kappa e kappa2) para todos os nós
    """
    

    def __init__(self):
        self.arquivoLido = None

        self.G = None

        self.dis = {} #menores distancias entre todos os pares (iterator -> dict)
        self.C = []  # clientes
        self.S = []  # servidores

        self.kappa = {}              # kappa[i][j]: conectividade (nó-conectividade) entre i e j
        self.kappa2 = {}             # kappa2[i]: conectividade máxima do nó i
        self.maiorKappa2 = None      # maior valor observado de kappa2 no grafo
        self.kappa2Medio = None      # média dos valores de kappa2 no grafo

        self.resultados = None

    def __str__(self):
        pass


    def leGrafoDimacs(self,arquivo):
        f = open(arquivo,"r")
        G= nx.Graph()
        for line in f:
            x = line.split()
            if x[0] == 'a':
                G.add_edge(x[1],x[2])
        f.close()    
        
        self.G = G

    def atualizaLog(self):
        nome = self.arquivoLido.split('.')    
        pref = nome[0]
        suf = nome[1]

        log.novoLogRede(pref+"_"+str(len(self.G.nodes()))+"_resul.txt")
        log.imprime(' ')
        if suf == 'gml':
            log.imprime(str(self.G.graph['Network']))



    def carregaEValidaGrafo(self,name):
        """
        Carrega um grafo de arquivo GML ou DIMACS e valida se é conectado e não direcionado.
        
        O método identifica o formato do arquivo pela extensão (.gml ou .dimacs),
        carrega o grafo, valida suas propriedades (conectividade e não-direcionado),
        e converte os rótulos dos nós para inteiros sequenciais.
        
        Args:
            name: string, caminho do arquivo (pode ser .gml ou .dimacs)
            
        Side effects:
            Atualiza self.arquivoLido com o nome do arquivo
            Atualiza self.G com o grafo carregado e processado
            Converte rótulos dos nós para inteiros (0, 1, 2, ...)
            Preserva rótulos originais no atributo 'nome' de cada nó
            
        Raises:
            SystemExit: se o grafo não for conectado ou for direcionado
        """
        self.arquivoLido = name

        nome = name.split('.')    
        suf = nome[1]
        
        if suf == 'gml':
            self.G = nx.read_gml(name, label='id')

        elif suf == 'dimacs':
            self.G = leGrafoDimacs(name)

        if not nx.is_connected(self.G):
            print("Grafo não conectado")
            exit()

        if nx.is_directed(self.G):
            print("Grafo direcionado")
            exit()
        
        #para facilitar o codigo transforma os labels em inteiros e guarda eles em name
        if len(nx.get_node_attributes(self.G,'nome'))==0:
            # Em versões recentes, a assinatura de convert_node_labels_to_integers pode não aceitar 'ordering'
            try:
                self.G = nx.convert_node_labels_to_integers(self.G, ordering="increasing degree", label_attribute='nome')
            except TypeError:
                self.G = nx.convert_node_labels_to_integers(self.G, label_attribute='nome')
        
        #else:#converter os labels de string para inteiros, isso ocorre caso esteja usando o arquivo uma segunda vez
        #    mapping={}    
        #    for i in G.nodes():
        #        mapping[i]=int(i)
        #    G = nx.relabel_nodes(G,mapping)
        
        # Atualiza log após o grafo estar carregado
        self.atualizaLog()  






    def analisa_grafo(self):
        """
        Calcula os kappas e as distâncias mínimas entre todos os pares de nós do grafo.
        
        Esta função executa análises completas de conectividade e distância no grafo,
        calculando e armazenando:
        - Grau de cada nó como atributo
        - Conectividade (kappa) entre todos os pares de nós
        - Conectividade máxima (kappa2) de cada nó
        - Distâncias mínimas entre todos os pares de nós
        - Estatísticas (maior kappa2 e kappa2 médio)
        
        Atualiza os atributos:
            self.kappa: dict, conectividade entre pares de nós
            self.kappa2: dict, conectividade máxima de cada nó
            self.dis: dict, distâncias mínimas entre pares de nós
            self.maiorKappa2: int, maior valor de kappa2 no grafo
            self.kappa2Medio: float, média dos valores de kappa2
            
        Side effects:
            - Imprime mensagens de progresso no console
            - Registra estatísticas no log via log.imprime()
        """

        # atributo 'grau' para cada nó (API NetworkX >= 2.x)
        nx.set_node_attributes(self.G, dict(self.G.degree()), 'grau')

        self.calcKappaGrafo()        
        print("Calculou os kappas")
        
        #menores distancias entre todos os pares (iterator -> dict)
        self.dis = dict(nx.all_pairs_dijkstra_path_length(self.G))
        print("Calculou as menores distancias")

        log.imprime('nVert ' + str(len(self.G.nodes()))+ ' nAres '+str(len(self.G.edges())))
 
        maiorKappa = 0
        kappaMedio = 0
        for i in nx.nodes(self.G):
            if self.kappa2[i] > maiorKappa:
                maiorKappa = self.kappa2[i]
            kappaMedio = kappaMedio + self.kappa2[i]
        kappaMedio = round(kappaMedio/len(self.G.nodes()),2)
        
        log.imprime('Maior kappa2 = '+str(maiorKappa))
        log.imprime('kappa2 medio = '+str(kappaMedio))

        self.maiorKappa2 = maiorKappa
        self.kappa2Medio = kappaMedio



    def calcKappaGrafo(self):
        kappa = self.kappa
        kappa2 = self.kappa2

        #inicializacao dos vetores de didionarios
        for s in nx.nodes(self.G):
            kappa[s]= {}
            kappa2[s]=0

        
        #calcula os kappas    
        for s in nx.nodes(self.G):
            for t in nx.nodes(self.G):
                if s != t:
                    if t not in kappa[s]:
                        kappa[s][t]=nx.node_connectivity(self.G,s,t)
                        kappa[t][s]=kappa[s][t]#estou trabalhando apenas com grafos não direcionados
                    if kappa[s][t] > kappa2[s]:
                        kappa2[s] = kappa[s][t]
            kappa[s][s]=kappa2[s]

        # atribui atributo 'kappa' aos nós (API NetworkX >= 2.x)
        nx.set_node_attributes(self.G, kappa2, 'kappa')

        self.kappa = kappa
        self.kappa2 = kappa2

    
        

def linearProgram(tipo,S,C,dis,kappa,kappa2,p=None,somaDis=None):
    """
    Resolve problema de alocação de servidores usando programação linear inteira.
    
    Implementa diferentes variantes do problema de localização de facilidades:
    - maxConect: maximiza conectividade (kappa) entre clientes e servidores
    - p-mediana: minimiza soma de distâncias com p servidores fixos
    
    Args:
        tipo: string indicando o tipo de problema a resolver:
            - "maxConect, menor número de servidores": minimiza número de servidores
            - "maxConect, maior número de de caminhos disjuntos": maximiza caminhos disjuntos
            - "maxConect, menor número de de caminhos disjuntos": minimiza caminhos disjuntos
            - "maxConect, menor soma de distancias": minimiza distâncias com conectividade máxima
            - "maxConect, maior soma de distancias": maximiza distâncias com conectividade máxima
            - "p-mediana, menor soma de distancias": problema clássico de p-mediana
            - "p-mediana, maior conectividade": maximiza conectividade para p-mediana
            - "p-mediana, menor conectividade": minimiza conectividade para p-mediana
        S: lista de nós candidatos a servidor
        C: lista de nós clientes
        dis: dicionário dis[i][j] com distâncias entre nós i e j
        kappa: dicionário kappa[i][j] com conectividade (nó-conectividade) entre i e j
        kappa2: dicionário kappa2[j] com máxima conectividade do nó j
        p: número de servidores (usado em alguns tipos de problema, None para outros)
        somaDis: restrição de soma máxima de distâncias (usado em p-mediana com conectividade)
    
    Returns:
        dict R com:
            R[0]: valor da função objetivo
            R[1]: lista de nós escolhidos como servidores
            R[2]: dicionário {servidor: [clientes]} mapeando cada servidor aos seus clientes
            R[3]: tempo de execução em segundos
    """

    begin('model') # begin modelling
    #verbose(True)  # be verbose

    A = iprod(S, C)#produto cartesiano
    x = var('x', A, bool) #
    y = var('y', S, bool)
    
    
    #x[i,j] cliente em j conectado ao servidor em i, se igual a 1
    #y[i] servidor em i, se igual a 1

    match tipo:
        case "maxConect, menor número de servidores":
            #minimizar numero de servidores
            minimize(sum(y[i] for i in S))

            #restricao se o cliente j está conectado a servidor i 
            #kappa(j,i)=kappa2(j)
            
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]

        #========================
        #acredito que seja o mesmo valor para ambos os casos, não lembro mais porque coloquei esses casos
        case "maxConect, maior número de de caminhos disjuntos":
            #maximizar o numero de caminhos disjuntos
            maximize(sum(kappa2[j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p

            #restricao se o cliente j está conectado a servidor i 
            #kappa(j,i)=kappa2(j)
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]

        case "maxConect, menor número de de caminhos disjuntos":
            #minimizar o numero de caminhos disjuntos
            minimize(sum(kappa2[j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p

            #restricao se o cliente j está conectado a servidor i 
            #kappa(j,i)=kappa2(j)
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]
        #========================

        case "maxConect, menor soma de distancias":
            #p-mediana
            minimize(sum(dis[i][j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p

            #restricao se o cliente j está conectado a servidor i 
            #kappa(j,i)=kappa2(j)
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]
    
        case "maxConect, maior soma de distancias":
            #p-mediana
            maximize(sum(dis[i][j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p

            #restricao se o cliente j está conectado a servidor i 
            #kappa(j,i)=kappa2(j)
            for i in S:    
                for j in C:
                    kappa2[j]*x[i,j] == kappa[j][i]*x[i,j]
    
        case "p-mediana, menor soma de distancias":
            #p-mediana
            minimize(sum(dis[i][j]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p
    
        case "p-mediana, maior conectividade":
            minimize(sum(kappa2[j]*x[i,j]-kappa[j][i]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p
            sum(dis[i][j]*x[i,j] for i,j in A) <=somaDis
        
        case "p-mediana, menor conectividade":
            maximize(sum(kappa2[j]*x[i,j]-kappa[j][i]*x[i,j] for i,j in A))
            sum(y[i] for i in S) == p
            sum(dis[i][j]*x[i,j] for i,j in A) <=somaDis

        
    #demais restricoes    
    #existe um servidor em i, se o cliente em j está conectado a i
    for i,j in A:
        y[i] >= x[i,j] 

    #cliente em j está conectado a apenas um servidor
    #se ha um servidor em i entao ele esta conectado com ele mesmo x[i,i] = 1
   
    for j in C:
        sum(x[i,j] for i in S) == 1

    t0 = time.time()
    solve() # solve the model
    tempo = time.time() -t0
    #sensitivity() # sensitivity report
    
    #save(mps='_save.mps', sol='_save.sol',
    # clp='_save.clp', glp='_save.glp', 
    # sen='_save.sen', ipt='_save.ipt',
    # mip='_save.mip')

    minSum = vobj()
    serv = []
    conect = {}
    
    for i in S:
        if y[i].primal == 1:
            serv.append(i)
            conect[i]=[]
            for j in C:
                if x[i,j].primal == 1:
                    conect[i].append(j)            
    R = {}
    R[0] = minSum #valor da funcao objetivo
    R[1] = serv #servidores escolhidos
    R[2] = conect #clientes conectados a servidores
    R[3] = tempo #tempo de execução

    end() 
   
    return R        



def menor_servidores_max_conect(rede, calcTempo = False):
    """
    Executa o caso 1 (maxConect: menor número de servidores) repetido N vezes,
    imprime os resultados e retorna p (número mínimo de servidores) e o tempo médio.

    Args:
        rede: objeto Rede contendo o grafo, clientes (C), servidores (S), 
              distâncias (dis) e estruturas de conectividade (kappa, kappa2)
        calcTempo: se True, executa 4 repetições para calcular tempo médio;
                   se False, executa apenas 1 vez (padrão: True)

    Returns:
        (p, pTempo):
            p -> inteiro, número mínimo de servidores
            pTempo -> float, tempo médio em segundos (ou tempo de 1 execução se calcTempo=False)
    """

    if calcTempo:
        repeticoes = 4
    else:
        repeticoes = 1  

    tipo = "maxConect, menor número de servidores"

    tempo = 0.0
    for _ in range(repeticoes):
        R = linearProgram(tipo, rede.S, rede.C, rede.dis, rede.kappa, rede.kappa2)
        print(R[3])
        tempo += R[3]

    pTempo = round(tempo / repeticoes, 2)

    log.imprime(tipo)
    log.imprime(R[0])  # número mínimo de servidores
    log.imprime(R[1])  # os servidores
    log.imprime(R[2])  # os clientes atribuídos aos servidores

    p = round(R[0])

    if not calcTempo:
        return p
    else:
        return p, pTempo


def max_conect_menor_soma_distancias(rede, p, calcTempo = False):
    """
    Executa o caso 2 (maxConect: menor soma de distâncias) repetido N vezes,
    imprime os resultados e retorna a soma das distâncias.

    Args:
        rede: objeto Rede contendo o grafo, clientes (C), servidores (S), 
              distâncias (dis) e estruturas de conectividade (kappa, kappa2)
        p: número de servidores (fixado pelo caso 1)
        calcTempo: se True, executa 4 repetições para calcular tempo médio;
                   se False, executa apenas 1 vez (padrão: False)

    Returns:
        Se calcTempo = True:
            (somaDisMaxConect, disMaxConectTempo): tupla com soma de distâncias e tempo médio
        Se calcTempo = False:
            somaDisMaxConect: int, valor do objetivo arredondado (soma de distâncias)
    """

    
    if calcTempo:
        repeticoes = 4
    else:
        repeticoes = 1

    tipo = "maxConect, menor soma de distancias"

    tempo = 0.0
    for _ in range(repeticoes):
        R = linearProgram(tipo, rede.S, rede.C, rede.dis, rede.kappa, rede.kappa2, p)
        tempo += R[3]
    disMaxConectTempo = round(tempo / repeticoes, 2)

    log.imprime(tipo)
    log.imprime(R[0])  # objetivo (soma de distâncias)
    log.imprime(R[1])  # servidores
    log.imprime(R[2])  # clientes atribuídos aos servidores

    somaDisMaxConect = round(R[0])

    if not calcTempo:
        return somaDisMaxConect
    else:
        return somaDisMaxConect, disMaxConectTempo


def max_conect_maior_soma_distancias(rede, p, calcTempo = False):
    """
    Executa o caso 2.5 (maxConect: MAIOR soma de distâncias) repetido N vezes,
    imprime os resultados e retorna a pior soma de distâncias.

    Args:
        rede: objeto Rede contendo o grafo, clientes (C), servidores (S), 
              distâncias (dis) e estruturas de conectividade (kappa, kappa2)
        p: número de servidores (fixado pelo caso 1)
        calcTempo: se True, executa 4 repetições para calcular tempo médio;
                   se False, executa apenas 1 vez (padrão: False)

    Returns:
        Se calcTempo = True:
            (piorSomaDisMaxConect, piorDisMaxConectTempo): tupla com pior soma de distâncias e tempo médio
        Se calcTempo = False:
            piorSomaDisMaxConect: int, valor do objetivo arredondado (pior soma de distâncias)
    """


    if calcTempo:
        repeticoes = 4
    else:
        repeticoes = 1

    tipo = "maxConect, maior soma de distancias"

    tempo = 0.0
    for _ in range(repeticoes):
        R = linearProgram(tipo, rede.S, rede.C, rede.dis, rede.kappa, rede.kappa2, p)
        tempo += R[3]
    piorDisMaxConectTempo = round(tempo / repeticoes, 2)

    log.imprime(tipo)
    log.imprime(R[0])  # objetivo (soma de distâncias)
    log.imprime(R[1])  # servidores
    log.imprime(R[2])  # clientes atribuídos aos servidores

    piorSomaDisMaxConect = round(R[0])

    if not calcTempo:
        return piorSomaDisMaxConect
    else:
        return piorSomaDisMaxConect, piorDisMaxConectTempo

def p_mediana_menor_soma_distancias(rede, p, calcTempo = False):
    """
    Executa o caso 3 (p-mediana: menor soma de distâncias) repetido N vezes,
    imprime os resultados e retorna a soma de distâncias.

    Args:
        rede: objeto Rede contendo o grafo, clientes (C), servidores (S), 
              distâncias (dis) e estruturas de conectividade (kappa, kappa2)
        p: número de servidores
        calcTempo: se True, executa 4 repetições para calcular tempo médio;
                   se False, executa apenas 1 vez (padrão: False)

    Returns:
        Se calcTempo = True:
            (somaDisPmedian, somaDisPmedianTempo): tupla com soma de distâncias e tempo médio
        Se calcTempo = False:
            somaDisPmedian: int, valor do objetivo arredondado (soma de distâncias)
    """

    if calcTempo:
        repeticoes = 4
    else:
        repeticoes = 1

    tipo = "p-mediana, menor soma de distancias"
    tempo = 0.0
    R = None
    for _ in range(repeticoes):
        R = linearProgram(tipo, rede.S, rede.C, rede.dis, rede.kappa, rede.kappa2, p)
        tempo += R[3]
    somaDisPmedianTempo = round(tempo / repeticoes, 2)

    log.imprime(tipo)
    log.imprime(R[0])  # objetivo (soma de distâncias)
    log.imprime(R[1])  # servidores
    log.imprime(R[2])  # clientes atribuídos aos servidores

    somaDisPmedian = round(R[0])

    if not calcTempo:
        return somaDisPmedian
    else:
        return somaDisPmedian, somaDisPmedianTempo


def p_mediana_maior_conectividade(rede, p, somaDisPmedian, calcTempo = False):
    """
    Executa o caso 4 (p-mediana: maior conectividade), imprime os resultados 
    e retorna a soma de gaps e as conexões cliente-servidor.

    Args:
        rede: objeto Rede contendo o grafo, clientes (C), servidores (S), 
              distâncias (dis) e estruturas de conectividade (kappa, kappa2)
        p: número de servidores
        somaDisPmedian: valor da restrição de soma de distâncias
        calcTempo: se True, executa 4 repetições para calcular tempo médio;
                   se False, executa apenas 1 vez (padrão: False)

    Returns:
        Se calcTempo = True:
            (somaDif, conexClieServ, somaDifTempo): tupla com soma de gaps, 
            dicionário de conexões e tempo médio
        Se calcTempo = False:
            (somaDif, conexClieServ): tupla com soma de gaps e dicionário de conexões
                somaDif -> int, valor do objetivo arredondado (soma dos gaps de conectividade)
                conexClieServ -> dict {servidor: [clientes]}, mapeamento de clientes por servidor
    """
    if calcTempo:
        repeticoes = 4
    else:
        repeticoes = 1

    tipo = "p-mediana, maior conectividade"

    tempo = 0.0
    for _ in range(repeticoes):
        R = linearProgram(tipo, rede.S, rede.C, rede.dis, rede.kappa, rede.kappa2, p, somaDisPmedian)
        tempo += R[3]
    somaDifTempo = round(tempo / repeticoes, 2)

    log.imprime(tipo)
    log.imprime(R[0])  # objetivo (soma de distâncias)
    log.imprime(R[1])  # servidores
    log.imprime(R[2])  # clientes atribuídos aos servidores
    
    somaDif = round(R[0])
    
    if not calcTempo:
        return somaDif, R[2]
    else:
        return somaDif, R[2], somaDifTempo

def analiseGaps(rede, conexClieServ):
    """
    Analisa os gaps de conectividade entre clientes e seus servidores atribuídos.
    
    Um gap ocorre quando a conectividade entre um cliente e seu servidor atribuído
    é menor que a conectividade máxima daquele cliente (kappa2). A função calcula
    estatísticas sobre esses gaps.

    Args:
        rede: objeto Rede contendo as estruturas de conectividade (kappa, kappa2)
        conexClieServ: dicionário {servidor: [clientes]} com a atribuição de clientes a servidores

    Returns:
        (numClieGap, maiorGap, kappaMaiorGap, gapMedio):
            numClieGap -> int, número de clientes com gap de conectividade
            maiorGap -> int, valor do maior gap encontrado
            kappaMaiorGap -> int, valor de kappa2 do cliente que possui o maior gap
            gapMedio -> float, média dos gaps relativos (gap/kappa2) arredondado para 2 casas
    """
    

    kappa = rede.kappa
    kappa2 = rede.kappa2

    numClieGap=0
    maiorGap=0
    kappaMaiorGap=0
    gapMedio = 0
    for i in conexClieServ:
        for j in conexClieServ[i]:
            if kappa[i][j] != kappa2[j]:
                numClieGap = numClieGap + 1
                gap = kappa2[j]- kappa[i][j]
                log.imprime('kappa2 = '+str(kappa2[j])+ ' gap = '+ str(gap))
                gapMedio = gapMedio + gap/kappa2[j]
                if gap >= maiorGap:
                    maiorGap = gap
                    if  kappa2[j] > kappaMaiorGap:
                        kappaMaiorGap= kappa2[j]
    if numClieGap != 0:
        gapMedio = round(gapMedio/numClieGap,2)


    return numClieGap, maiorGap, kappaMaiorGap, gapMedio



            
class Saida():
    """
    Gerencia a geração de saídas e tabelas formatadas dos resultados.
    
    Attributes:
        tabela (dict): Estrutura aninhada {numVert: {numAres: {nomeGrafo: linhaTabela}}}
    """
    
    def __init__(self):
        self.tabela = {}
    
    def adicionaLinha(self, rede):
        """
        Adiciona uma linha formatada à tabela a partir dos dados da rede.
        
        Args:
            rede: objeto Rede com grafo carregado e resultados calculados
        """
        numVert = len(rede.G.nodes())
        numAres = len(rede.G.edges())
        
        # Extrai informações do arquivo
        nome_arquivo = rede.arquivoLido.split('.')
        suf = nome_arquivo[1]
        
        if suf == 'gml':
            nomeRedeTabela = str(rede.G.graph.get('Network', 'Unknown'))
            anoRede = str(rede.G.graph.get('DateYear', ''))
            nomeGrafo = nomeRedeTabela
        else:
            nomeRedeTabela = nome_arquivo[0].split('/')[-1]  # Remove path
            anoRede = ''
            nomeGrafo = nomeRedeTabela
        
        # Extrai resultados do dicionário
        if hasattr(rede, 'resultados') and rede.resultados:
            p = rede.resultados['p'][0]
            pTempo = rede.resultados['pTime'][0]
            somaDisMaxConect = rede.resultados['somaDisMaxC'][0]
            piorSomaDisMaxConect = rede.resultados['piorSomaDisMaxC'][0]
            somaDisPmedian = rede.resultados['somaDisPmedian'][0]
            somaGapsPmedian = rede.resultados['somaGapsPmedian'][0]
            taxaDis = rede.resultados['taxaDis'][0]
            piorTaxaDis = rede.resultados['piorTaxaDis'][0]
            numClieGap = rede.resultados['numClieGap'][0]
            maiorGap = rede.resultados['maiorGap'][0]
            kappaMaiorGap = rede.resultados['kappaMaiorGap'][0]
            gapMedio = rede.resultados['gapMedio'][0]
        else:
            raise ValueError("Rede não possui resultados para gerar linha da tabela")
        
        # Monta linha LaTeX formatada
        linhaTabelaInteira = (
            f"{nomeRedeTabela} {anoRede}"
            f" & {numVert} & {numAres}"
            f" & {rede.maiorKappa2} & {p} & {round(p/numVert, 2)}"
            f" & {pTempo}"
            f" & {piorSomaDisMaxConect}"
            f" & {somaDisMaxConect}"
            f" & {somaDisPmedian}"
            f" & {piorTaxaDis} & {taxaDis}"
            f" & {somaGapsPmedian} & {numClieGap} & {maiorGap} & {kappaMaiorGap}"
            f" & {gapMedio}"
            " \\\\"
        )
        
        # Armazena na estrutura hierárquica
        if numVert not in self.tabela:
            self.tabela[numVert] = {}
        if numAres not in self.tabela[numVert]:
            self.tabela[numVert][numAres] = {}
        
        self.tabela[numVert][numAres][nomeGrafo] = linhaTabelaInteira
        
        return linhaTabelaInteira
    
    def salvaTabela(self, arquivo='tabela.txt'):
        """
        Salva a tabela completa em arquivo ordenada por vértices e arestas.
        
        Args:
            arquivo: nome do arquivo de saída (padrão: 'tabela.txt')
        """
        with open(arquivo, "w") as tab:
            for numVert in sorted(self.tabela.keys()):
                for numAres in sorted(self.tabela[numVert].keys()):
                    for nomeGrafo in sorted(self.tabela[numVert][numAres].keys()):
                        tab.write(self.tabela[numVert][numAres][nomeGrafo] + '\n')
        
        print(f"Tabela salva em: {arquivo}")
    
    def __str__(self):
        """Retorna representação textual da tabela."""
        linhas = []
        for numVert in sorted(self.tabela.keys()):
            for numAres in sorted(self.tabela[numVert].keys()):
                for nomeGrafo in sorted(self.tabela[numVert][numAres].keys()):
                    linhas.append(self.tabela[numVert][numAres][nomeGrafo])
        return '\n'.join(linhas)




#MAIN
def main():
    listNomes = sorted(glob.glob("archive/*.gml"))
    saida = Saida()

    for name in listNomes:

        print(name)

        rede = Rede()
        rede.carregaEValidaGrafo(name)


        rede.C = sorted(rede.G.nodes())  # clientes
        rede.S = list(rede.C)  # servidores

        ## calcula os kappas e as menores distancias entre os vértices
        rede.analisa_grafo()


        #RESOLUCAO POR PROGRAMACAO INTEIRA LINEAR
    
        #---MaxConect---       
        #encontrar menor numero de servidores para maxima conectividade
        p, pTempo  = menor_servidores_max_conect( rede, calcTempo = True)

        
        # minimizando a soma das distancias
        somaDisMaxConect = max_conect_menor_soma_distancias( rede, p)

        
        # maximizando a soma das distancias
        piorSomaDisMaxConect = max_conect_maior_soma_distancias( rede, p ) 
        
        
        #---p-median---
        #obtem soma de distancias para p-mediana        
        somaDisPmedian = p_mediana_menor_soma_distancias( rede, p )
        
        
        # maximizando a conectividade ou minimizando os gaps
        somaGapsPmedian, conexClieServ = p_mediana_maior_conectividade( rede, p, somaDisPmedian )

        #analise ---MaxConect---    

        taxaDis = round(somaDisMaxConect/somaDisPmedian, 2)
        piorTaxaDis = round(piorSomaDisMaxConect/somaDisPmedian, 2)

        #analise ---pmedian---       

        numClieGap, maiorGap, kappaMaiorGap, gapMedio = analiseGaps( rede, conexClieServ)


        # ARMAZENAMENTO DOS RESULTADOS NA REDE
        rede.resultados = {
            'p': (p, '$p$', "Minimum number of servers for the RP-MaxC problem."),
            'pTime': (pTempo, "$t(s)$", "Running time in seconds to calculate $p$"),
            'somaDisMaxC': (somaDisMaxConect, "RP-MaxC max ($\\sum$ dist)", " Maximum sum of distances for the RP-MaxC problem."),
            'piorSomaDisMaxC': (piorSomaDisMaxConect, "RP-MaxC min ($\\sum$ dist)", " Minimum sum of distances for the RP-MaxC problem."),
            'somaDisPmedian': (somaDisPmedian, "$p$-median $\\sum$ dist", " Sum of distances for the $p$-median problem."),
            'somaGapsPmedian': (somaGapsPmedian, "$\\sum$diff", "Minimum sum of the differences between the number of vertex-disjoints paths of the clients of $p$-median and the  $\\kappa_2$ of these clients."),
            'taxaDis': (taxaDis, "Min ratio", " Ratio between the minimum sum of distances for the RP-MaxC and the sum for the $p$-median."),
            'piorTaxaDis': (piorTaxaDis, "Max ratio", " Ratio between the maximum sum of distances for the RP-MaxC and the sum for the $p$-median."),
            'numClieGap': (numClieGap, "Num clients with diff", "Number of clients of $p$-median that do not attend the maximum connectivity restraint."),
            'maiorGap': (maiorGap, "Max diff", " Maximum difference between the number of vertex-disjoint paths of a client with the server and the $\\kappa_2$ of this client."),
            'kappaMaiorGap': (kappaMaiorGap, "$\\kappa_2$ with max diff", "Maximum $\\kappa_2$ of a client with the largest difference."),
            'gapMedio': (gapMedio, "Avg diff", " Average difference between the number of vertex-disjoint paths of the clients of $p$-median and the $\\kappa_2$ of these clients.")
        }

        log.imprimeResultados(rede)

        # Adiciona linha à tabela usando a classe Saida
        linhaTabela = saida.adicionaLinha(rede)
        log.imprime(linhaTabela)


    # Salva tabela completa ao final
    saida.salvaTabela('tabela.txt')

if __name__ == "__main__":
    main()