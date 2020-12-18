#coding=UTF-8
"""
This code provides a series of basic functions to create pajek-style files, such as. net,. clu.
"""

# Author: Liang Chen <squirrel_d@126.com>
# License: BSD 3 clause

import networkx as nx
import copy
import queue_set
import Queue
from xgmml import *

class PajekUtil(object):
    def __init__(self):
        self.g = None
        self.similarity_matrix = None

    def createNetFromList(self,pair_array):
        """
        create networkx DiGraph g from a list with [sourcenode,endnode] in each element

        Args:
            param pair_array(list):

        Returns:
            DiGraph:
        """
        g=nx.DiGraph()
        nodes_collection = sum(pair_array,[])
        nodes = [i for i in set(nodes_collection)]
        for i in range(len(nodes)):
            g.add_node(i+1,ga = nodes[i])
        for i in pair_array:
            g.add_edge(nodes.index(i[0])+1,nodes.index(i[1])+1,{'weight':1.0})
        return g

    def createNetFromFile(self,filePath):
        """
        create networkx DiGraph g from a txt file with sourcenode,endnode in each line

        Args:
            filePath(str): absolute path of txt file

        Returns:
            DiGraph:
        """
        g=nx.DiGraph()
        arcs = []
        with open(filePath, 'r') as fp:
            line = fp.readline()
            while line:
                #if line.startswith('\xef\xbb\xbf'):
            	#	line = line[3:]
                x = line.strip().split(',')
                x_normalized = [i.strip() for i in x if i is not '']
                arcs.append([x_normalized[0],x_normalized[1]])
                line = fp.readline()
        g = self.createNetFromList(arcs)
        return g

    def createNetFromMatrixWithWeight(self,cooccur_array):
        """
        create networkx DiGraph g from a 2-D numpy array which indicates its adjacency matrix

        Args:
            cooccur_array(np.array): 2-D numpy array

        Returns:
             DiGraph: each node has two attributions, namely,{'label':str,'weight':float}
        """
        g=nx.DiGraph()
        arcs = []
        for i in range(cooccur_array.shape[0]):
            for j in range(i):
                # arc: [node1,node2,weight]
                arcs.append([i+1,j+1,cooccur_array[i,j]])
        nodes = [i+1 for i in range(cooccur_array.shape[0])]
        for i in range(len(nodes)):
            g.add_node(i+1,label = nodes[i])
        for i in arcs:
            g.add_edge(nodes.index(i[0])+1,nodes.index(i[1])+1,{'weight':i[2]})
        return g

    def createNetFromPairWithWeight(self,filePath):
        """
        create networkx DiGraph g from a txt file with sourcenode,endnode,weight in each line

        Args:
            filePath(str): absolute path of txt file

        Returns:
            DiGraph:
        """
        g=nx.DiGraph()
        arcs = []
        with open(filePath, 'r') as fp:
            line = fp.readline()
            while line:
                  x = line.strip().split(',')
                  x_normalized = [i for i in x if i is not '']
                  arcs.append([x_normalized[0],x_normalized[1],x_normalized[2]])
                  line = fp.readline()
        nodes_collection = sum([arc[:2] for arc in arcs],[])
        nodes = [i for i in set(nodes_collection)]
        for i in range(len(nodes)):
            g.add_node(i+1,label = nodes[i])
        for i in arcs:
            g.add_edge(nodes.index(i[0])+1,nodes.index(i[1])+1,weight=i[2])
        return g

    def createNetWithRelTypeFromList(self,triple_array):
        """
        create networkx DiGraph g from a list, notice that reltype is int type

        Args:
            triple_array(list): with [sourcenode,endnode,arctype] in each element

        Returns:
             DiGraph:
        """
        g=nx.DiGraph()
        nodes_collection = sum([arc[:2] for arc in triple_array],[])
        nodes = [i for i in set(nodes_collection)]
        for i in range(len(nodes)):
            g.add_node(i+1,ga = nodes[i])
        for i in triple_array:
            g.add_edge(nodes.index(i[0])+1,nodes.index(i[1])+1,type=i[2])
        return g

    def createPartition4EntityTypeFromList(self, g, entityType_list, filePath):
        '''
        create partition file from a list containing entity type of each node.
        notice:len(entitytype_list) maybe larger than len(g.nodes()), since DiGraph g is only the independent component
        of origial network while entitytype_list contains the entity types of all nodes from original network

        Args:
            g(DiGraph):
            entityType_list(list): entityType_list format [entity,type] and type is Integer
            filePath(str): absolute path of output file

        Returns:
            None
        '''
        partition = [0 for i in range(len(g.nodes()))]
        for i in g.node.keys():
            for j in range(len(entityType_list)):
                if entityType_list[j][0] == g.node[i]['label']:
                    partition[i-1] = entityType_list[j][1]
                    break;
        with open(filePath, 'w') as fp:
            fp.write("*Vertices "+str(len(g))+"\n")
            for i in range(len(partition)):
                fp.write(str(partition[i])+"\n")

    def writeGraph2pajekNetFile(self, g, filePath, networkType='Arcs'):
        '''
        This function is added on 20191007 by Liang Chen

        Args:
            g(DiGraph):
            filePath(str): absolute path
            networkType(str): {'Arcs','Edges'}, indicate whether the output network is directed or not,networkType

        Returns:
            None:
        '''
        nodes = g.nodes()
        edges_list = []
        type_dict = {}
        #In type_dict, key = type,value = [[head node，tail node，weight][...]]
        for i in g.edges():
            # if edge has 'weight' attribution, output it as edge weight
            if g[i[0]][i[1]].get('weight', -1.0) != -1.0:
                edge_tmp = str(nodes.index(i[0]) + 1) + " " + str(nodes.index(i[1]) + 1) + " "\
                            + str(g[i[0]][i[1]]['weight'])
            # if edge dont have 'weight' attribution, output default value -1 as edge weight
            else:
                edge_tmp = str(nodes.index(i[0]) + 1) + " " + str(nodes.index(i[1]) + 1) + " " + str(
                    1.0)
            if g[i[0]][i[1]].get('type',-1.0)!=-1.0:
                type_tmp = g[i[0]][i[1]]['type']
                if type_dict.get(type_tmp,[])==[]:
                    type_dict[type_tmp]=[]
                type_dict[type_tmp].append(edge_tmp)
            else:
                edges_list.append(edge_tmp)
        with open(filePath, 'w') as fp:
            fp.write("*Vertices "+str(len(g))+"\n")
            for i in range(len(nodes)):
                fp.write(str(i+1)+" \""+str(g.node[nodes[i]]['label'])+"\" 0.0 0.0 0.0\n")
            # check edge of network
            if edges_list!=[]:
                if networkType == 'Arcs':
                    fp.write("*Arcs \n")
                elif networkType == 'Edges':
                    fp.write("*Edges \n")
                else:
                    raise Exception( networkType + " is an Invalid type of Network!")
                for str_tmp in edges_list:
                    fp.write(str_tmp+"\n")
            else:
                assert type_dict!={}
                for type_tmp in type_dict.keys():
                    if networkType == 'Arcs':
                        fp.write("*Arcs :"+ str(type_tmp) +"\n")
                    elif networkType == 'Edges':
                        fp.write("*Edges :"+ str(type_tmp) +"\n")
                    else:
                        raise Exception( networkType + " is an Invalid type of Network!")
                    for str_tmp in type_dict[type_tmp]:
                        fp.write(str_tmp + "\n")

    def writeGraph2CytoscapeXGMML(self, g, filePath):
        '''
        output DiGraph in cytoscape format

        Args:
            g(DiGraph):
            filePath(str): absolute path

        Returns:
            None
        '''
        with open(filePath, 'w') as fid:
            addHead(fid, 'network')
            for nodeID in g.vs:
                addNode(fid, nodeID['name'], nodeID.index, fill='#007FFF', shape='RECTANGLE')
            for edge in g.get_edgelist():
                addEdge(fid, edge[0], edge[1], str(edge[0]) + ' to ' + str(edge[1]), SourceArrowShape='NONE',
                        TargetArrowShape='ARROW', )
            fid.write('</graph>\n')
    def writeNodeAttOfCytoscape2file(self, g, filePath):
        '''
        write attribution of nodes in DiGraph in cytoscape file format

        Args:
            g(DiGraph):
            filePath(str): absolute path

        Returns:
            None
        '''
        with open(filePath, 'w') as f:
            f.write('node_type' + '\n')
            for nodeID in g.nodes():
                f.write(g.node[nodeID]['label'] + '=' + str(g.node[nodeID]['type']) + '\n')

    def writeEdgeAttOfCytoscape2file(self, g, filePath):
        '''
        write attribution of edges in DiGraph in cytoscape file format

        Args:
            g(DiGraph):
            filePath(str): absolute path

        Returns:
            None
        '''
        with open(filePath, 'w') as f:
            f.write('edge_type' + '\n')
            for edge in g.edges():
                f.write(
                    g.node[edge[0]]['label'] + '(pp)' + g.node[edge[1]]['label'] + '=' + str(g[edge[0]][edge[1]]['type']) + '\n')
    def writeGraph2CytoscapeCSV(self, g, filePath):
        '''
        write DiGraph in cytoscape supported CSV format

        Args:
            g(DiGraph):
            filePath(str): absolute path

        Returns:
            None
        '''
        with open(filePath, 'w') as f:
            for edge in g.edges():
                f.write(g.node[edge[0]]['label'] + '\t' + g.node[edge[1]]['label'] + '\t' + str(
                    g[edge[0]][edge[1]]['type'])  + '\n')

    def getMaxWeightPathBySingleNode(self,nodeID):
        '''
        debugged by Liang Chen on 20201129,
        search global main path in self.g given a source node

        Args:
            nodeID(int): ID of source node

        Returns:
            list: a list with two element [weight of resulted path, an instance of DiGraph representing the resulted path]
        '''
        q = Queue.Queue()
        q.put(nodeID)
        dic = dict()
        dic[nodeID]=[0.0,[nodeID]]
        while not q.empty():
            nodeID_tmp = q.get()
            successors = self.g.successors(nodeID_tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    nodeID_tmp_2 = copy.copy(dic.get(nodeID_tmp)[1])
                    nodeID_tmp_2.append(i)
                    dic[i]=[dic.get(nodeID_tmp)[0] + self.g[nodeID_tmp][i]['weight'], nodeID_tmp_2]
                else:
                    nodeID_tmp_3 = dic.get(i)
                    if nodeID_tmp_3[0] < dic.get(nodeID_tmp)[0]+self.g[nodeID_tmp][i]['weight']:
                        nodeID_tmp_2 = copy.copy(dic.get(nodeID_tmp)[1])
                        nodeID_tmp_2.append(i)
                        dic[i]=[dic.get(nodeID_tmp)[0] + self.g[nodeID_tmp][i]['weight'], nodeID_tmp_2]
        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def createPartitionByLabelsListFile(self, g, clu_input, clu_output):
        '''
        Given a txt file containing a list of patentNo, create corresponding partition file

        Args:
            g(DiGraph): an instance of DiGraph object
            clu_input(str): path of a txt file with each line in 'patentNo,...' format
            clu_output(str): path of a clu file for pajek to read the partition information

        Returns:
            None
        '''
        partition = []
        nodes =[]
        with open(clu_input, 'r') as fp:
            line = fp.readline()
            while line:
                  pos = line.strip().find(',')
                  patentNo = line[:pos]
                  nodes.append(patentNo.strip())
                  line = fp.readline()
        for i in range(len(g.nodes())):
            partition.append(0)
            for j in range(len(nodes)):
                if nodes[j] == g.node[i]['label']:
                    partition[i] = 1
                    break;
        with open(clu_output, 'w') as fp:
            fp.write("*Vertices "+str(len(g))+"\n")
            for i in range(len(partition)):
                fp.write(partition[i]+"\n")

    def createPartitionByNetFile(self, graphFile, subgraphFile, partitionFile):
        '''
        Given a graph and a subgraph,create a partition file to label the nodes of subgraph from graph parent

        Args:
            graphFile(str): path of a pajek net file
            subgraphFile(str): path of a pajek net file
            partitionFile(str): path of output file

        Returns:
            None:
        '''
        g1 = self.getGraphFromPajeknet(graphFile)
        g2 = self.getGraphFromPajeknet(subgraphFile)
        partition = []
        for i in range(len(g1.nodes())):
            partition.append(0)
            for j in range(len(g2.nodes())):
                if g2.node[j+1]['label'] == g1.node[i+1]['label']:
                    partition[i] = 1
                    break;
        with open(partitionFile, 'w') as fp:
            fp.write("*Vertices "+str(len(g1))+"\n")
            for i in range(len(partition)):
                fp.write(str(partition[i])+"\n")

    def getMaxWeightPathBySingleNode_Graph(self,nodeID,g):
        '''
        search global main path in a graph given a source node

        Args:
            nodeID(int): ID of source node
            g(DiGraph): Graph

        Returns:
            list: a list with two element

             [weight of resulted path, an instance of DiGraph representing the resulted path]
        '''
        q = Queue.Queue()
        graph = nx.DiGraph()
        q.put(nodeID)
        dic = dict()
        graph.add_node(nodeID,ga = g.node[nodeID]['label'])
        dic[nodeID]=[0.0,graph]
        while not q.empty():
            tmp = q.get()
            successors = g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp3 = copy.deepcopy(dic.get(tmp)[1])
                    tmp3.add_node(i,ga = g.node[i]['label'])
                    tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                    dic[i]=[dic.get(tmp)[0]+g[tmp][i]['weight'],tmp3]
                else:
                    tmp2 = dic.get(i)
                    if tmp2[0] < dic.get(tmp)[0]+g[tmp][i]['weight']:
                        tmp3 = copy.deepcopy(dic.get(tmp)[1])
                        tmp3.add_node(i,ga = g.node[i]['label'])
                        tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                        dic[i]=[dic.get(tmp)[0]+g[tmp][i]['weight'],tmp3]
        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def getmulti_MaxWeightPathBySingleNode_Graph(self,nodeID,g):
        '''
        Given a source node, if there exists multiple maxWeight paths in Graph g,find them all

        Args:
            nodeID(int):
            g(DiGraph): DiGraph

        Returns:
             list: a list in format like [4.200226000000001, [graph1,graph2,......]]
        '''
        q = Queue.Queue()
        graph = nx.DiGraph()
        q.put(nodeID)
        dic = dict()
        graph.add_node(nodeID,ga = g.node[nodeID]['label'])
        dic[nodeID]=[0.0,[graph]]
        while not q.empty():
            tmp = q.get()
            successors = g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp4 = copy.deepcopy(dic.get(tmp)[1])
                    for tmp3 in tmp4:
                        tmp3.add_node(i,ga = g.node[i]['label'])
                        tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                    dic[i]=[dic.get(tmp)[0]+g[tmp][i]['weight'],tmp4]
                else:
                    tmp2 = dic.get(i)
                    if tmp2[0] < dic.get(tmp)[0]+g[tmp][i]['weight']:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,ga = g.node[i]['label'])
                            tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                        dic[i]=[dic.get(tmp)[0]+g[tmp][i]['weight'],tmp4]
                    elif tmp2[0] == dic.get(tmp)[0]+g[tmp][i]['weight']:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,ga = g.node[i]['label'])
                            tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                        tmp2[1].extend(tmp4)
                        dic[i]=tmp2
        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def getmulti_MaxWeightPathBySingleNode_Graph_newSumMethod_textSim_topology(self, nodeID, g, patentNos, sim_matrix, semantic_weight =1.0, topology_weight =1.0):
        '''
        Search the max weight path from a Graph give source node nodeID, notice the edge weight is calculated by semantic_sim*semantic_weight+topology*topology_weight_sim
        ,notice newSumMethod refers to internode_sum_distance()

        Args:
            nodeID(int): source node
            g(DiGraph):
            patentNos(list): list of patent NO.
            sim_matrix(np.array): matrix of text similarity
            semantic_weight(float): weight of textual similarity between nodes
            topology_weight(float): weight of topological value between nodes, such as SPC,SPNC

        Returns:
            list: a list in format like [4.200226000000001, [graph1,graph2,......]]
        '''
        q = Queue.Queue()
        graph = nx.DiGraph()
        q.put(nodeID)
        dic = dict()
        graph.add_node(nodeID, ga=g.node[nodeID]['label'])
        dic[nodeID] = [0.0, [graph]]
        while not q.empty():
            tmp = q.get()
            successors = g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp4 = copy.deepcopy(dic.get(tmp)[1])
                    tmp_value = self.internode_sum_distance(g, patentNos, tmp4[0].nodes(), i, sim_matrix)
                    for tmp3 in tmp4:
                        tmp3.add_node(i, ga=g.node[i]['label'])
                        tmp3.add_edge(tmp, i, weight=g[tmp][i]['weight'])
                    dic[i] = [dic.get(tmp)[0] + semantic_weight*tmp_value+topology_weight*g[tmp][i]['weight'], tmp4]
                else:
                    tmp2 = dic.get(i)
                    tmp_value = self.internode_sum_distance(g, patentNos, dic.get(tmp)[1][0].nodes(), i, sim_matrix)
                    if tmp2[0] < dic.get(tmp)[0] + semantic_weight*tmp_value+topology_weight*g[tmp][i]['weight']:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i, ga=g.node[i]['label'])
                            tmp3.add_edge(tmp, i, weight=g[tmp][i]['weight'])
                        dic[i] = [dic.get(tmp)[0] + semantic_weight*tmp_value+topology_weight*g[tmp][i]['weight'], tmp4]
                    elif tmp2[0] == dic.get(tmp)[0] + semantic_weight*tmp_value+topology_weight*g[tmp][i]['weight']:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i, ga=g.node[i]['label'])
                            tmp3.add_edge(tmp, i, weight=g[tmp][i]['weight'])
                        tmp2[1].extend(tmp4)
                        dic[i] = tmp2
        result = sorted(dic.items(), lambda x, y: cmp(x[1][0], y[1][0]), reverse=True)
        return result[0]

    def getmulti_MaxWeightPathBySingleNode_Graph_newSumMethod(self, nodeID, g, patentNos, sim_matrix):
        '''
        Search the max weight path from a Graph give source node nodeID, notice that edge weight is measured by textual similarity and stored in sim_matrix, in addition newSumMethod refers to internode_sum_distance()

        Args:
            nodeID(int): source node
            g(DiGraph):
            patentNos(list): list of patent NO.
            sim_matrix(np.array): matrix of text similarity
            semantic_weight(float): weight of textual similarity between nodes
            topology_weight(float): weight of topological value between nodes, such as SPC,SPNC

        Returns:
            list: a list in format like [4.200226000000001, [graph1,graph2,......]]
        '''
        q = Queue.Queue()
        graph = nx.DiGraph()
        q.put(nodeID)
        dic = dict()
        graph.add_node(nodeID,ga = g.node[nodeID]['label'])
        dic[nodeID]=[0.0,[graph]]
        while not q.empty():
            tmp = q.get()
            successors = g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp4 = copy.deepcopy(dic.get(tmp)[1])
                    tmp_value = self.internode_sum_distance(g, patentNos, tmp4[0].nodes(), i, sim_matrix)
                    for tmp3 in tmp4:
                        tmp3.add_node(i,ga = g.node[i]['label'])
                        tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                    dic[i]=[dic.get(tmp)[0]+tmp_value,tmp4]
                else:
                    tmp2 = dic.get(i)
                    tmp_value = self.internode_sum_distance(g, patentNos, dic.get(tmp)[1][0].nodes(), i, sim_matrix)
                    if tmp2[0] < dic.get(tmp)[0]+tmp_value:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,ga = g.node[i]['label'])
                            tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                        dic[i]=[dic.get(tmp)[0]+tmp_value,tmp4]
                    elif tmp2[0] == dic.get(tmp)[0]+tmp_value:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,ga = g.node[i]['label'])
                            tmp3.add_edge(tmp,i,weight = g[tmp][i]['weight'])
                        tmp2[1].extend(tmp4)
                        dic[i]=tmp2
        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def internode_sum_distance(self,sources,target,sim_matrix):
        '''
        calculate the weight between source nodes to target node by sum up the edge weight of target node to all source nodes

        Args:
            sources(list): source node IDs
            target(int): target node ID
            sim_matrix(np.array): textual similarities between all nodes

        Returns:
            float:result weight
        '''
        sum_value =0.0
        for source in sources:
            sim = sim_matrix[target-1,source-1]
            sum_value=sum_value+sim
        return sum_value

    def globalMainPath(self):
        '''
        search a list of max weighted paths given all source nodes

        Returns:
             list: a list of max weighted paths given all source nodes
        '''
        l = []
        sourceNodes = self.getSourceNodes(self.g)
        for i in sourceNodes:
            tmp = self.getMaxWeightPathBySingleNode(i)
            l.append(tmp)
        result = sorted(l,lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result

    def multi_sources_globalMainPath_return_list(self, g):
        '''
        search a list of max weighted paths given all source nodes

        Args:
            g(DiGraph): DiGraph

        Returns:
            list: list containing main paths from all source nodes
        '''
        result = []
        sources = self.getSourceNodes(g)
        for i in sources:
            result.append(self.getmulti_MaxWeightPathBySingleNode_Graph(i, g))
        result.sort(cmp=lambda x, y: cmp(x[1][0], y[1][0]), reverse=True)
        return result

    def multi_sources_globalMainPath_return_DiGraph(self,g):
        '''
        search a list of max weighted paths given all source nodes and merge them into a DiGraph

        Args:
            g(DiGraph): DiGraph

        Returns:
            DiGraph:
        '''
        resultGraph = nx.DiGraph()
        sub_graphs = []
        sourceNodes = self.getSourceNodes(g)
        for i in sourceNodes:
            tmp = self.getmulti_MaxWeightPathBySingleNode_Graph(i,g)
            sub_graphs.extend(tmp[1][1])
        nodes = []
        edges=[]
        for i in sub_graphs:
            nodes.extend(i.nodes())
            edges.extend(i.edges())
        nodes = [i for i in set(nodes)]
        edges = [i for i in set(edges)]
        resultGraph.add_nodes_from(nodes)
        resultGraph.add_edges_from(edges)
        for i in resultGraph.nodes():
            resultGraph.node[i]['label'] = g.node[i]['label']
        for i in resultGraph.edges():
            resultGraph.edge[i[0]][i[1]]['weight'] = g.edge[i[0]][i[1]]['weight']
        return resultGraph

    def loadNetworkFromPajeknet(self,netAddress):
        '''
        load a pajek net file into self.g as an DiGraph object

        Args:
            netAddress(str): absolute path of pajek net file

        Returns:
            None:
        '''
        G=nx.DiGraph()
        with open(netAddress, 'r') as fp:
          if fp.readline().lower().find("*vertices")!=-1:
              line = fp.readline()
              while line.lower().find("*edges")==-1 and line.lower().find("*arcs")==-1:
                  tmp = line.strip().split(' ')
                  tmp_2 = [i for i in tmp if i is not '']
                  tmp_2[1] = tmp_2[1].replace('\"','')
                  G.add_node(int(tmp_2[0]),label = tmp_2[1])
                  line = fp.readline()
              line = fp.readline()
              while line:
                  tmp = line.strip().split(' ')
                  tmp_2 = [i for i in tmp if i is not '']
                  G.add_edge(int(tmp_2[0]),int(tmp_2[1]),weight=float(tmp_2[2]))
                  line = fp.readline()
        self.g = G

    def loadNetworkFromPajeknetWithMultipleRelType(self, netAddress):
        '''
        load pajek net file with multi-type edges into an DiGraph object

        Args:
            netAddress(str): path of pajek net file

        Returns:
            DiGraph:
        '''
        G = nx.DiGraph()
        with open(netAddress, 'r') as fp:
            if fp.readline().lower().find("*vertices") != -1:
                line = fp.readline()
                while line.lower().find("*arcs") == -1:
                    tmp = line.strip().split(' ')
                    tmp_2 = [i for i in tmp if i is not '']
                    tmp_2[1] = tmp_2[1].replace('\"', '')
                    G.add_node(int(tmp_2[0]), ga=tmp_2[1])
                    line = fp.readline()
                line = line.replace('"','')
                current_type = int(line[line.find(':')+1:].strip())
                line = fp.readline()
                while line:
                    if line.lower().find("*arcs") == -1:
                        tmp = line.strip().split(' ')
                        tmp_2 = [i for i in tmp if i is not '']
                        G.add_edge(int(tmp_2[0]), int(tmp_2[1]), weight=float(tmp_2[2]),type = current_type)
                        line = fp.readline()
                    else:
                        line = line.replace('"', '')
                        current_type = int(line[line.find(':') + 1:].strip())
                        line = fp.readline()
        return G


    def getGraphFromPajeknet(self,netAddress):
        '''
        load a pajek net file into  an DiGraph object

        Args:
            netAddress(str): path of pajek net file

        Returns:
            DiGraph:
        '''
        G=nx.DiGraph()
        with open(netAddress, 'r') as fp:
          if fp.readline().lower().find("*vertices")!=-1:
              line = fp.readline()
              while line.lower().find("*arcs")==-1:
                  a = line.strip().split(' ')
                  b = [i for i in a if i is not '']
                  #print b
                  b[1] = b[1].replace('\"','')
                  G.add_node(int(b[0]),ga = b[1])
                  line = fp.readline()
              line = fp.readline()
              while line:
                  a = line.strip().split(' ')
                  b = [i for i in a if i is not '']
                  G.add_edge(int(b[0]),int(b[1]),weight=float(b[2]))
                  line = fp.readline()
        return G

    def addPartition2Graph(self,g,partition,partition_name):
        '''
        add paritition information to DiGraph

        Args:
            g(DiGraph):
            partition(list):  list
            partition_name(str): String

        Return:
            DiGraph:
        '''
        for i in g.nodes():
            g.node[i][partition_name]=partition[i-1]
        return g

    def convertPajek2Cytoscape(self,pajek_net,pajek_par,cyto_csv,cyto_node,cyto_edge):
        '''
        convert pajek file to cytoscape file

        Args:
            pajek_net(str): path of net file
            pajek_par(str): path of partition file
            cyto_csv(str): file path to write graph information
            cyto_node(str): file path to write node information
            cyto_edge(str): file path to write edge information

        Returns:
            None:
        '''
        g = self.loadNetworkFromPajeknetWithMultipleRelType(pajek_net)
        par = self.loadPartitionFromClu(pajek_par)
        g = self.addPartition2Graph(g,par,'type')
        self.writeGraph2CytoscapeCSV(g, cyto_csv)
        self.writeNodeAttOfCytoscape2file(g, cyto_node)
        self.writeEdgeAttOfCytoscape2file(g, cyto_edge)

    def loadNetworkFromPajeknet_edges(self,netAddress):
        '''
        load graph into self.g from net file

        Args:
            netAddress(str): absolute path of net file

        Returns:
            None:
        '''
        G=nx.DiGraph()
        with open(netAddress, 'r') as fp:
          if fp.readline().lower().find("*vertices")!=-1:
              line = fp.readline()
              while line.lower().find("*edges")==-1:
                  tmp = line.strip().split(' ')
                  tmp_2 = [i for i in tmp if i is not '']
                  tmp_2[1] = tmp_2[1].replace('\"','')
                  G.add_node(int(tmp_2[0]),label = tmp_2[1])
                  line = fp.readline()
              line = fp.readline()
              while line:
                  tmp = line.strip().split(' ')
                  tmp_2 = [i for i in tmp if i is not '']
                  G.add_edge(int(tmp_2[0]),int(tmp_2[1]),weight=float(tmp_2[2]))
                  line = fp.readline()
        self.g = G

    def loadPartitionFromClu(self,netAddress):
        '''
        load partition info from clu file

        Args:
            netAddress(str): absolute path of clu file

        Returns:
            list: a list of partition info
        '''
        clu=[]
        with open(netAddress, 'r') as fp:
          if fp.readline().lower().find("*vertices")!=-1:
              line = fp.readline()
              while line:
                  tmp = line.strip().split(' ')
                  tmp_2 = [i for i in tmp if i is not '']
                  clu.append(tmp_2[0])
                  line = fp.readline()
        return clu

    def writeLabel_par_map_2_txt(self,g,clu,fileAddre):
        '''
        create node_label-partition map,and write to a file

        Args:
            g(DiGraph): DiGraph
            clu(list): list
            fileAddre(str):

        Returns:
            None:
        '''
        label_par_map = []
        for i in g.nodes():
            label_par_map.append([g.node[i]['label'],clu[i-1]])
        label_par_map.sort(cmp=lambda x,y: cmp(x[1], y[1]), reverse=False)
        with open(fileAddre,'w') as writeFile:
            for i in range(len(label_par_map)):
                writeFile.write(label_par_map[i][1]+','+label_par_map[i][0]+'\n')

    def createPartitionByList(self, g, clu, fileAddr):
        '''
        create partition file given partition info contained in clu

        Args
            g(DiGraph): DiGraph
            clu(list): list
            fileAddr(str): absolute path

        Returns:
            None:
        '''
        with open(fileAddr, 'w') as fp:
            fp.write("*Vertices "+str(len(g))+"\n")
            for i in range(len(g)):
                if i+1 in clu:
                    fp.write(str(1)+"\n")
                else:
                    fp.write("0\n")
                    
    def getSourceNodes(self,graph):
        '''
        get source node list from a directed graph

        Args:
            graph(DiGraph): DiGraph

        Returns:
            None:
        '''
        sources = []
        d_in = graph.in_degree(graph)
        for node in graph.nodes():
            if d_in[node]==0:
                sources.append(node)
        return sources

    def localMainPath(self,graph):
        '''
        get local main path from graph

        Args:
            graph(DiGraph): DiGraph

        Returns:
            DiGraph: local main path as an instance of DiGraph
        '''
        queue = queue_set.Queue_Set()
        sources = self.getSourceNodes(graph)
        result_g =nx.DiGraph()
        maxSuccessorLists = []
        for i in sources:    
            tmp_max_endNodes = self.getMaxWeightArcFromSrc(graph,i)
            if len(maxSuccessorLists)==0:
                maxSuccessorLists.append([i,tmp_max_endNodes])
                value_tmp = tmp_max_endNodes[0]
            elif value_tmp<tmp_max_endNodes[0]:
                maxSuccessorLists=[]
                maxSuccessorLists.append([i,tmp_max_endNodes])
                value_tmp = tmp_max_endNodes[0]
            elif value_tmp==tmp_max_endNodes[0]:
                maxSuccessorLists.append([i,tmp_max_endNodes])
        
        # maxSuccessorLists STRUCTURE:[[5, [0.083333333, [1, 2]]], [8, [0.083333333, [1]]], [9, [0.083333333, [1, 2, 4]]]]
        for i in maxSuccessorLists:
            result_g.add_node(i[0],ga=graph.node[i[0]]['label'])
            for j in i[1][1]:
                result_g.add_node(j,ga=graph.node[j]['label'])
                queue.put(j)
                result_g.add_edge(i[0],j,{'weight': graph[i[0]][j]['weight']})
        
        while queue.isEmpty()==False:
            src = queue.get()
            tmp_max_endNodes = self.getMaxWeightArcFromSrc(graph,src)
            for i in tmp_max_endNodes[1]:
                result_g.add_node(i,ga=graph.node[i]['label'])
                result_g.add_edge(src,i,{'weight': graph[src][i]['weight']})
                queue.put(i)
        
        return result_g 

    def getMaxWeightArcFromSrc(self,graph,src):
        '''
        get max weighted arc given a source node

        Args:
            graph(DiGraph): DiGraph
            src(int): source node id

        Returns:
            list: [weight, end_node_list],all nodes in end_node_list with equal weight on the arc to source node
        '''
        successors = graph.successors(src)
        result_tmp = []
        value_tmp = 0.0
        for i in successors:
            if graph[src][i]['weight']>value_tmp:
                if len(result_tmp)>0:
                    result_tmp = []
                value_tmp = graph[src][i]['weight']
                result_tmp.append(i)
            elif graph[src][i]['weight']==value_tmp:
                result_tmp.append(i)
        return [value_tmp,result_tmp]

    def localMainPathFromOneSource(self,graph,i):
        '''
        return local main path give a source node

        Args:
            graph(DiGraph): DiGraph
            i(int): source node ID

        Returns:
            DiGraph:
        '''
        queue = queue_set.Queue_Set()
        result_g =nx.DiGraph()
        maxSuccessorLists = [] 
        tmp_max_endNodes = self.getMaxWeightArcFromSrc(graph,i)
        maxSuccessorLists.append([i,tmp_max_endNodes])
        
        # maxSuccessorLists STRUCTURE:[[5, [0.083333333, [1, 2]]], [8, [0.083333333, [1]]], [9, [0.083333333, [1, 2, 4]]]]
        for i in maxSuccessorLists:
            result_g.add_node(i[0],ga=graph.node[i[0]]['label'])
            for j in i[1][1]:
                result_g.add_node(j,ga=graph.node[j]['label'])
                queue.put(j)
                result_g.add_edge(i[0],j,{'weight': graph[i[0]][j]['weight']})
        
        while queue.isEmpty()==False:
            src = queue.get()
            tmp_max_endNodes = self.getMaxWeightArcFromSrc(graph,src)
            for i in tmp_max_endNodes[1]:
                result_g.add_node(i,ga=graph.node[i]['label'])
                print result_g.node[i]['label']
                result_g.add_edge(src,i,{'weight': graph[src][i]['weight']})
                queue.put(i)
        
        return result_g 
    def getSubGraphByOneSource(self,graph,sourceNode):
        '''
        return a subgraph constituted by successors of a given source node

        Args:
            graph(DiGraph): DiGraph
            sourceNode(int):source node ID

        Returns:
            DiGraph:
        '''
        q = Queue.Queue()
        q.put(sourceNode)
        sub_g = nx.DiGraph()
        sub_g.add_node(sourceNode)
        while not q.empty():
            src = q.get()
            #judge if subgraph has this node
            successors= graph.successors(src)
            for i in successors:
                if sub_g.has_node(i)==False:
                    sub_g.add_node(i)    
                    q.put(i)
                if sub_g.has_edge(src,i)==False:
                    sub_g.add_edge(src,i,weight = graph[src][i]['weight'])
        return sub_g

    def multiSourcePath(self,g):
        '''
        return a DiGraph constituted by local main paths induced by all source nodes

        Args:
            g(DiGraph): DiGraph

        Returns:
            DiGraph:
        '''
        sources = self.getSourceNodes(g)
        sub_graphs = []
        resultGraph =nx.DiGraph()
        for i in sources:
            g_tmp =self.localMainPathFromOneSource(g,i)
            sub_graphs.append(copy.deepcopy(g_tmp))
        
        nodes = []
        edges=[]
        for i in sub_graphs:
            nodes.extend(i.nodes())
            edges.extend(i.edges())
        nodes = [i for i in set(nodes)]
        edges = [i for i in set(edges)]
        resultGraph.add_nodes_from(nodes)
        resultGraph.add_edges_from(edges)
        for i in resultGraph.nodes():
            resultGraph.node[i]['label'] = g.node[i]['label']
        for i in resultGraph.edges():
            resultGraph.edge[i[0]][i[1]]['weight'] = g.edge[i[0]][i[1]]['weight']
        return resultGraph

    def sort_subGraph(self,sub_graphs):
        '''
        sort a list of subGraphs by graph weight

        Args:
            sub_graphs(list): [sub_graph1,sub_graph2,sub_graph3,.......]

        Returns:
            list: [[weight_sum_sub1,sub_graph_1],[weight_sum_sub2,sub_graph_2],...]
        '''
        result = []
        for sub_graph in sub_graphs:
            sum = 0.0
            for arc in sub_graph.edges():
                sum+= sub_graph[arc[0]][arc[1]]['weight']
            result.append([sum,sub_graph])
        result.sort(cmp=lambda x,y: cmp(x[0], y[0]), reverse=True)
        return result
    def combine_subGraph(self,sub_graphs,g):
        '''
        combine a list of sub_graphs into one DiGraph object

        Args:
            sub_graphs(list): list
            g(DiGraph): DiGraph

        Returns:
            DiGraph:
        '''
        resultGraph =nx.DiGraph()
        nodes = []
        edges=[]
        for i in sub_graphs:
            nodes.extend(i[1].nodes())
            edges.extend(i[1].edges())
        nodes = [i for i in set(nodes)]
        edges = [i for i in set(edges)]
        resultGraph.add_nodes_from(nodes)
        resultGraph.add_edges_from(edges)
        for i in resultGraph.nodes():
            resultGraph.node[i]['label'] = g.node[i]['label']
        for i in resultGraph.edges():
            resultGraph.edge[i[0]][i[1]]['weight'] = g.edge[i[0]][i[1]]['weight']
        return resultGraph

    def combine_subGraphArray(self,sub_graphs,g):
        '''
        combine a list of sub_graphs into one DiGraph object

        Args:
            sub_graphs(list): [sub_graph1,sub_graph2,......]
            g(DiGraph):

        Returns:
            None:
        '''
        resultGraph =nx.DiGraph()
        nodes = []
        edges=[]
        for i in sub_graphs:
            nodes.extend(i.nodes())
            edges.extend(i.edges())
        nodes = [i for i in set(nodes)]
        edges = [i for i in set(edges)]
        resultGraph.add_nodes_from(nodes)
        resultGraph.add_edges_from(edges)
        for i in resultGraph.nodes():
            resultGraph.node[i]['label'] = g.node[i]['label']
        g_edges_list = g.edges();
        if len(g_edges_list)>0 and 'weight' in g.edge[g_edges_list[0][0]][g_edges_list[0][1]].keys():
            for i in resultGraph.edges():
                resultGraph.edge[i[0]][i[1]]['weight'] = g.edge[i[0]][i[1]]['weight']
        return resultGraph

    def combine_multisubGraph(self,sub_graphs,g):
        '''
        combine a list of [[0.56,[subgraph1,subgraph2],[0.55,[subgraph3,subgraph4]]] into a DiGraph object

        Args:
            sub_graphs(list):  [[0.56,[subgraph1,subgraph2],[0.55,[subgraph3,subgraph4]]]
            g(DiGraph):  DiGraph

        Returns:
            DiGraph:
        '''
        resultGraph =nx.DiGraph()
        nodes = []
        edges=[]
        for i in sub_graphs:
            for j in i[1]:
                nodes.extend(j.nodes())
                edges.extend(j.edges())
        nodes = [i for i in set(nodes)]
        edges = [i for i in set(edges)]
        resultGraph.add_nodes_from(nodes)
        resultGraph.add_edges_from(edges)
        for i in resultGraph.nodes():
            resultGraph.node[i]['label'] = g.node[i]['label']
        for i in resultGraph.edges():
            resultGraph.edge[i[0]][i[1]]['weight'] = g.edge[i[0]][i[1]]['weight']
        return resultGraph
