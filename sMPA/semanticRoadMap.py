#coding=UTF-8
"""
This code provides basic functions for semantic main path analysis
"""

# Author: Liang Chen <squirrel_d@126.com>
# License: BSD 3 clause

import numpy as np
import pickle
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import logging
from sMPA.pajekUtil import *
from scipy.sparse import csr_matrix
class semanticRoadMap:

    def __init__(self, lsi_serializationAddress, dictionary_serialAddress, pns_serialAddress, index_serialAddress,corpus_serialAddress):
        '''
        init semanticRoadMap class

        Args:
            lsi_serializationAddress(str): absolute path of LSI model file
            dictionary_serialAddress(str): absolute path of dictionary file
            pns_serialAddress(str): absolute path of patent NO file
            index_serialAddress(str): absolute path of index serialization file
            corpus_serialAddress(str): 'corpus.data',it contains all nodes' texts,even more than that,but never less
                              the order of texts should be consistant with that of pns
        '''
        self.g=[]
        self.pu = PajekUtil()
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
        f = file(lsi_serializationAddress)
        self.lsi = pickle.load(f)
        f.close();
        f = file(dictionary_serialAddress)
        self.dictionary = pickle.load(f)
        f.close();
        f = file(pns_serialAddress)
        self.pns = pickle.load(f)
        f.close();
        f = file(index_serialAddress)
        self.index = pickle.load(f)
        f.close();
        f = file(corpus_serialAddress)
        corpus = pickle.load(f)
        self.matrix = self.create_sim_matrix(self.lsi, corpus)
        self.english_punctuations = [',','.',':','?','(',')','[',']','&','!','*','@','#','$','%']

    def getSimByNodeID(self,nodeID,sims):
        '''
        This function is used in algorithms for searching query-based main path, i.e.,

        getMaxWeightPathBySingleNode_graph_nodeWeight(),
        getMaxWeightPathBySingleNode_nodeWeight(),
        getMaxWeightPathBySingleNode_graph_nodeWeight_arcWeight(),
        getMaxWeightPathBySingleNode_nodeWeight_raw(),

        and it returns similarity between nodeID text and the query text

        Args:
            nodeID(int): node index in self.g.nodes()
            sims(list): a list containing similarities between docs in corpus and the query text

        Returns:
            float: similarity between nodeID text and the query text
        '''
        tmp = self.g.node[nodeID]['label']
        tmp_id = self.pns.index(tmp)
        sim = sims[tmp_id]
        return sim

    def getSimByNode_pnIndex(self, node_pnIndex, sims):
        '''
        get similarity by node index in pn list

        Args:
            node_pnIndex(int): node index in pn list
            sims(list): a list containing similarities between corpus and the query text

        Returns:
            float:
        '''
        return sims[node_pnIndex - 1]

    def getNodeIDByNodePN(self, NodePN):
        '''
        return nodeID in self.g.nodes() by node's patent NO

        Args:
            NodePN(str):node's patent NO

        Returns:
            int:
        '''
        result = None
        for n in self.g.nodes():
            if self.g.node[n]['label'] == NodePN:
                result = n
                break
        return result

    def getMaxWeightPathInGraph_returnGraphType_nodeWeight(self, sims):
        '''
        this function uses node weight, which usually is the similarity between query text and docs in corpus, to generate query based global main path

        Args:
            sims(list): containing the similarities between query text and docs in corpus

        Returns:
            list: a list of sorted max weight paths induced by all source nodes
        '''
        l = []
        sourceNodes = self.pu.getSourceNodes(self.g)
        for i in sourceNodes:
            tmp = self.getMaxWeightPathBySingleNode_graph_nodeWeight(i,sims)
            l.append(tmp)
        l.sort(cmp = lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return l

    def getMaxWeightPathInGraph_nodeWeight_arcWeight(self, sims,node_weight=1.0,arc_weight=1.0):
        '''
        This function uses node weight and arc weight to generate query based global main path

        Args:
            node_weight(float): relative weight of node
            arc_weight(float): relative weight of arc
            sims(list): a list containing the similarites between the query text and every doc in corpus.

        Returns:
            list: A structure like [[path_weight,[path_1_graph_type,path_2_graph_type]]...]
        '''
        l = []
        sourceNodes = self.pu.getSourceNodes(self.g)
        for i in sourceNodes:
            tmp = self.getMaxWeightPathBySingleNode_graph_nodeWeight_arcWeight(i, sims,node_weight,arc_weight)
            l.append(tmp)
        l.sort(cmp=lambda x, y: cmp(x[1][0], y[1][0]), reverse=True)
        return l
    
    def getMaxWeightPathBySingleNode_nodeWeight(self, nodeID, sims):
        '''this function used for generating query based main path,
        Attention! Here only use node weight to search max weight main path
        it searches the path with the max Weight given a source nodeID

        Args:
            nodeID(int): a given source nodeID.,"circulation cooling air temperature control"
            sims(list): a list containing the similarites between the query text and every doc in corpus.

        Returns:
            list: A structure like [path_weight,[12,3,5,namely, node list on the result path]]
        '''
        q = Queue.Queue()
        q.put(nodeID)
        dic = dict()
        dic[nodeID]=[sims[nodeID],[nodeID]]
        while not q.empty():
            tmp = q.get()
            successors = self.g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp3 = copy.copy(dic.get(tmp)[1])
                    tmp3.append(i)
                    dic[i]=[dic.get(tmp)[0]+self.getSimByNodeID(i,sims),tmp3]
                else:
                    tmp2 = dic.get(i)
                    if tmp2[0] < dic.get(tmp)[0]+self.getSimByNodeID(i,sims):
                        tmp3 = copy.copy(dic.get(tmp)[1])
                        tmp3.append(i)
                        dic[i]=[dic.get(tmp)[0]+self.getSimByNodeID(i,sims),tmp3]
        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def getMaxWeightPathBySingleNode_graph_nodeWeight(self, nodeID, sims):
        '''this function used for generating query based main path,notice here only use node weight to search max weight main path,it searches the path with the max Weight given a source nodeID

        Args:
            nodeID(int): a given source nodeID.,"circulation cooling air temperature control"
            sims(list): a list containing the similarites between the query text and every doc in corpus.

        Returns:
            list: A structure like

            [path_weight,[path_1_graph_type,path_2_graph_type]],

            commonly speaking, there is only one path output as result, but if there are two paths with identical weight, output them both
        '''
        q = Queue.Queue()
        graph = nx.DiGraph()
        q.put(nodeID)
        dic = dict()
        graph.add_node(nodeID, label=self.g.node[nodeID]['label'])
        dic[nodeID]=[self.getSimByNodeID(nodeID,sims),[graph]]
        while not q.empty():
            tmp = q.get()
            successors = self.g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp4 = copy.deepcopy(dic.get(tmp)[1])
                    for tmp3 in tmp4:
                        tmp3.add_node(i, label=self.g.node[i]['label'])
                        tmp3.add_edge(tmp, i)
                    dic[i] = [dic.get(tmp)[0] + self.getSimByNodeID(i,sims), tmp4]
                else:
                    tmp2 = dic.get(i)
                    if tmp2[0] < dic.get(tmp)[0]+self.getSimByNodeID(i,sims):
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,ga = self.g.node[i]['label'])
                            tmp3.add_edge(tmp,i)
                        dic[i]=[dic.get(tmp)[0]+self.getSimByNodeID(i,sims),tmp4]
                    elif tmp2[0] == dic.get(tmp)[0]+self.getSimByNodeID(i,sims):
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,ga = self.g.node[i]['label'])
                            tmp3.add_edge(tmp,i)
                        tmp2[1].extend(tmp4)
                        dic[i]=tmp2

        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def getMaxWeightPathBySingleNode_graph_nodeWeight_arcWeight(self, nodeID, sims,node_weight,arc_Weight):
        '''this function used for generating query based main path,
        notice here use both node weight and arc weight to search max weight main path
        it searches the path with the max Weight given a source nodeID

        Args:
            nodeID(int): a given source nodeID.,"circulation cooling air temperature control"
            sims(list): a list containing the similarites between the query text and every doc in corpus.

        Returns:
            list: A structure like [path_weight,[path_1_graph_type,path_2_graph_type]],

            commonly speaking, there is only one path output as result, but if there are two paths
            with identical weight, output them both
        '''
        q = Queue.Queue()
        graph = nx.DiGraph()
        q.put(nodeID)
        dic = dict()
        graph.add_node(nodeID, label=self.g.node[nodeID]['label'])
        dic[nodeID]=[self.getSimByNodeID(nodeID,sims)*node_weight,[graph]]
        while not q.empty():
            tmp = q.get()
            successors = self.g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp4 = copy.deepcopy(dic.get(tmp)[1])
                    for tmp3 in tmp4:
                        tmp3.add_node(i, label=self.g.node[i]['label'])
                        tmp3.add_edge(tmp, i,weight = self.g[tmp][i]['weight'])
                    dic[i] = [dic.get(tmp)[0] + arc_Weight*self.g[tmp][i]['weight']+ node_weight*self.getSimByNodeID(i,sims), tmp4]
                else:
                    tmp2 = dic.get(i)
                    if tmp2[0] < dic.get(tmp)[0]+node_weight*self.getSimByNodeID(i,sims)+arc_Weight*self.g[tmp][i]['weight']:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,label = self.g.node[i]['label'])
                            tmp3.add_edge(tmp,i,weight = self.g[tmp][i]['weight'])
                        dic[i]=[dic.get(tmp)[0]+node_weight*self.getSimByNodeID(i,sims)+arc_Weight*self.g[tmp][i]['weight'],tmp4]
                    elif tmp2[0] == dic.get(tmp)[0]+node_weight*self.getSimByNodeID(i,sims)+arc_Weight*self.g[tmp][i]['weight']:
                        tmp4 = copy.deepcopy(dic.get(tmp)[1])
                        for tmp3 in tmp4:
                            tmp3.add_node(i,label = self.g.node[i]['label'])
                            tmp3.add_edge(tmp,i,weight = self.g[tmp][i]['weight'])
                        tmp2[1].extend(tmp4)
                        dic[i]=tmp2

        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def getMaxWeightPathBySingleNode_nodeWeight_raw(self,nodeID,sims):
        '''this function is used for LLDA

        Args:
            nodeID(int): given nodeID
            sims(list): #sims is llda.n_m_z[:,mc_id]

        Returns:
            list: [sum_weight,[1,2,5,namely,nodeIDs of the result path]]
        '''
        q = Queue.Queue()
        q.put(nodeID)
        dic = dict()
        dic[nodeID]=[sims[nodeID-1],[nodeID]]
        while not q.empty():
            tmp = q.get()
            successors = self.g.successors(tmp)
            for i in successors:
                if dic.get(i) is None:
                    q.put(i)
                    tmp3 = copy.copy(dic.get(tmp)[1])
                    tmp3.append(i)
                    dic[i]=[dic.get(tmp)[0] + self.getSimByNode_pnIndex(i, sims), tmp3]
                else:
                    tmp2 = dic.get(i)
                    if tmp2[0] < dic.get(tmp)[0]+self.getSimByNode_pnIndex(i, sims):
                        tmp3 = copy.copy(dic.get(tmp)[1])
                        tmp3.append(i)
                        dic[i]=[dic.get(tmp)[0] + self.getSimByNode_pnIndex(i, sims), tmp3]
        result = sorted(dic.items(),lambda x,y:cmp(x[1][0],y[1][0]),reverse=True)
        return result[0]

    def getSimByNodePair(self, g, pns, node1, node2):
        '''
        return similarity between node1 and node2

        Args:
            g(DiGraph):
            pns(list):
            node1(int): nodeID
            node2(int): nodeID

        Returns:
            float:
        '''
        node1_label = g.node[node1]['label']
        node2_label = g.node[node2]['label']
        node1_pn_index = pns.index(node1_label)
        node2_pn_index = pns.index(node2_label)
        sim = self.matrix[node1_pn_index][node2_pn_index]
        return sim
    def calculateSimilarityByTerms(self, terms):
        '''
        calculate similarities between terms and docs in self.corpus

        Args:
            terms(str): query text

        Returns:
            list: similarities list
        '''
        terms = self.preprocess(terms)
        terms_stemmed_bow = self.dictionary.doc2bow(terms)
        terms_stemmed_lsi = self.lsi[terms_stemmed_bow]
        sims = self.index[terms_stemmed_lsi]
        return sims
    def preprocess(self, query_text):
        '''
        a function to pre-process query_text

        Args:
            query_text(str): query text

        Returns:
            str: pre-processed string
        '''
        terms_lower = [word for word in query_text.lower().split()]
        #remove stopword
        english_stopwords = stopwords.words('english')
        terms_filtered_stopwords = [word for word in terms_lower if not word in english_stopwords]
        #some punctuations havent been removed completely, continues
        terms_filtered = [word for word in terms_filtered_stopwords if not word in self.english_punctuations]
        #extract stem
        st = LancasterStemmer()
        terms_stemmed = [st.stem(word) for word in terms_filtered] 
        return terms_stemmed

    def computeArcWeight4GraphByTextSim(self, g, pns):
        '''
        set textual similarity as arc weight in DiGraph g

        Args:
            g(DiGraph):
            pns(list): a list of patent NO

        Returns:
            None:
        '''
        #tranverse all the arc
        arcs = g.edges()
        for arc in arcs:
            sim = self.getSimByNodePair(g, pns, arc[0], arc[1])
            g[arc[0]][arc[1]]['weight'] = sim
    def setArcWeight4GrahphByLSI(self,texts):
        '''
        set textual similarity to corresponding arc in self.g

        Args:
            texts(list): a list of splitted string

        Return:
            None
        '''
        arcs = self.g.edges()
        for arc in arcs:
            tmp_id1 = self.pns.index(self.g.node[arc[0]]['label'])
            tmp_id2 = self.pns.index(self.g.node[arc[1]]['label'])
            terms = texts[tmp_id1]
            terms_stemmed_bow = self.dictionary.doc2bow(terms)
            terms_stemmed_lsi = self.lsi[terms_stemmed_bow]
            sims = self.index[terms_stemmed_lsi]
            if sims[tmp_id2]==0.0:
                self.g[arc[0]][arc[1]]['weight'] = 0.001
            self.g[arc[0]][arc[1]]['weight'] = sims[tmp_id2]

    def addLSIWeight2TopologyWeight4ArcInGraph(self,texts,topology_weight=1.0,text_weight=1.0):
        '''
        add texutual similarity to original arc weight in self.g, textual similarity is calcualted by LSI

        Args:
            texts(list): a list of splitted text
            topology_weight(float): weight of topological arc weight in main path searching, such as SPC(Search Path Count), SPNC(Search Path Node Pair)
            text_weight(float): weight of textual similarity in main path searching

        Returns:
            None
        '''
        arcs = self.g.edges()
        for arc in arcs:
            tmp_id1 = self.pns.index(self.g.node[arc[0]]['label'])
            tmp_id2 = self.pns.index(self.g.node[arc[1]]['label'])
            terms = texts[tmp_id1]
            terms_stemmed_bow = self.dictionary.doc2bow(terms)
            terms_stemmed_lsi = self.lsi[terms_stemmed_bow]
            sims = self.index[terms_stemmed_lsi]
            # as pajek doesn't show the line with weight value of 0, here we set line weight to 0.001 if the weight value is 0
            if topology_weight*self.g[arc[0]][arc[1]]['weight']+text_weight*sims[tmp_id2] ==0.0:
                self.g[arc[0]][arc[1]]['weight'] = 0.001
            else:
                self.g[arc[0]][arc[1]]['weight'] = topology_weight*self.g[arc[0]][arc[1]]['weight']+text_weight*sims[tmp_id2]

    def setArcWeight4GrahphByLSIViaSimMatrix(self):
        '''
        set arc weight in self.g with similarities of nodes connected by the arc, the similarities are calculated by LSI model

        Args:

        Returns:
            None
        '''
        arcs = self.g.edges()
        for arc in arcs:
            tmp_id1 = self.pns.index(self.g.node[arc[0]]['label'])
            tmp_id2 = self.pns.index(self.g.node[arc[1]]['label'])
            if type(self.matrix) is csr_matrix:
                sim = self.matrix[tmp_id1,tmp_id2]
            else:
                sim = self.matrix[tmp_id1][tmp_id2]
            self.g[arc[0]][arc[1]]['weight'] = sim

    def addLSIWeight2TopologyWeight4ArcInGrahphViaSimMatrix(self,topology_weight,text_weight):
        '''
        add texutual similarity to original arc weight in self.g, textual similarity is calcualted by LSI

        Args:
            topology_weight(float):
            text_weight(float):

        Returns:
            None
        '''
        arcs = self.g.edges()
        for arc in arcs:
            tmp_id1 = self.pns.index(self.g.node[arc[0]]['label'])
            tmp_id2 = self.pns.index(self.g.node[arc[1]]['label'])
            sim = self.matrix[tmp_id1][tmp_id2]
            self.g[arc[0]][arc[1]]['weight'] = text_weight*sim+topology_weight*self.g[arc[0]][arc[1]]['weight']
    def create_sim_matrix(self,model,corpus):
        '''
        store similarities between nodes in self.g into a matrix

        Args:
            model(gensim.model): trained gensim.model, i.e., loaded lsi_serialization
            corpus(list): i.e., [dictionary.doc2bow(text) for text in texts]

        Returns:
            np.array: np 2D array
        '''
        topics =[model[c] for c in corpus]
        dense = np.zeros((len(topics),100),float)
        for ti,t in enumerate(topics):
            for tj,v in t:
                dense[ti,tj] = v
        pairwise = np.nan_to_num(1.0-distance.squareform(distance.pdist(dense,'cosine')))
        return pairwise

    def create_sim_sparse_matrix(self):
        '''
        store similarities between nodes in self.g into an csr_matrix object

        Returns:
            csr_matrix:
        '''
        row = []
        col = []
        data = []
        arcs = self.g.edges()
        #add row col and data twice because matrix is symmetric;
        for arc in arcs:
            tmp_id1 = arc[0]-1
            tmp_id2 = arc[1]-1
            row.append(tmp_id1)
            col.append(tmp_id2)
            data.append(self.g[arc[0]][arc[1]]['weight'])
            row.append(tmp_id2)
            col.append(tmp_id1)
            data.append(self.g[arc[0]][arc[1]]['weight'])
            pairwise = csr_matrix((data,(row,col)),shape=(self.g.number_of_nodes(),self.g.number_of_nodes()),dtype = np.float32)
        return pairwise

    def queryBasedMainPath_nodeWeight(self,query_text,topN_graph=1):
        """Create query based main path according to node weight and merge top N query paths into one DiGraph object.

        Args:
            query_text(str): query text,e.g.,"circulation cooling air temperature control"
            topN_graph(int): combine top N query paths as the result path.

        Returns:
            DiGraph: networkx.DiGraph object
            """
        retriveResult = self.calculateSimilarityByTerms(query_text)
        maxWeightPath_list = self.getMaxWeightPathInGraph_returnGraphType_nodeWeight(retriveResult)
        subgraph_list = [graph_tmp[1][1] for graph_tmp in maxWeightPath_list[:topN_graph]]
        subgraph_list = sum(subgraph_list,[])
        g_result = self.pu.combine_subGraphArray(subgraph_list, self.g)
        return g_result

    def queryBasedMainPath_nodeWeight_arcWeight(self,query_text,node_Weight=1.0,arc_Weight=1.0):
        """Create query based main path according to node weight and arc weight

        Args:
            query_text(str): query text,e.g.,"circulation cooling air temperature control"
            topN_graph(int): combine top N query paths as the result path.

        Returns:
            DiGraph: A networkx.DiGraph object
            """
        retriveResult = self.calculateSimilarityByTerms(query_text)
        maxWeightPath_list = self.getMaxWeightPathInGraph_returnGraphType_nodeWeight_arcWeight(retriveResult,node_Weight,arc_Weight)
        return maxWeightPath_list

    def multi_sources_globalMainPath_textSim_topology_new_sum_method(self,spc_network_file,topology_weight=1.0,text_weight=1.0):
        '''
        this function return multiple global main path according to textual similarity and topological attributes, notice newSumMethod refers to internode_sum_distance()

        Args:
            spc_network_file(str): absolute path of pajek net file, i.e., 'deleteLoops_spc_3603.net'
            topology_weight(int): 1.0
            text_weight(int): 1.0

        Returns:
            list: containing main paths from all source nodes
        '''
        result = []
        self.pu.loadNetworkFromPajeknet(spc_network_file);
        sources = self.pu.getSourceNodes(self.g)
        for i in sources:
            result.append(self.pu.getmulti_MaxWeightPathBySingleNode_Graph_newSumMethod_textSim_topology\
                              (i, self.g, self.pns, self.matrix, semantic_weight= text_weight, topology_weight = topology_weight))
        result.sort(cmp=lambda x, y: cmp(x[1][0], y[1][0]), reverse=True)
        return result