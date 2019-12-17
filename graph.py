import dgl

def build_graph_fc():
    g = dgl.DGLGraph()
    g.add_nodes(15)
    edge_list = []
    for i in range(15):
        for j in range(i+1):
            edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g

def build_graph_skeleton():
    g = dgl.DGLGraph()
    g.add_nodes(15)
    edge_list = [(0,0), (0,2), (0,8), (0,9), (0,12), (0,13),
            (1,1), (1,5), (1,6), (1,10), (1,14),
            (2,2), (2,3), (2,7), (2,11), (2,13),
            (3,3), (3,8), (3,12), (3,14),
            (4,4), (4,5), (4,6), (4,9), (4,10),
            (5,5), (5,6), (5,10),
            (9,9), (9,8), (9,12), (9,14),
            (13,13), (13,7), (13,11),
            (6,6), (7,7), (8,8), (10,10), (11,11),
            (12,12), (14,14)]
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g

