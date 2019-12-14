import dgl

def build_graph():
    g = dgl.DGLGraph()
    g.add_nodes(15)
    edge_list = []
    for i in range(15):
        for j in range(i):
            edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g

