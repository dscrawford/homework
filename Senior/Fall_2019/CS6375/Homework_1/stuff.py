def ID3_Grow_Tree(S, tA, cols):
    node = ()
    s_all = len(S)
    s_1   = len(S[S[tA] == 1])
    s_0   = len(S[S[tA] == 0])
    
    s_uniq = S[tA].unique()
    if (len(s_uniq) == 1 and s_uniq[0] == 0):
        print("checkers")
        return 0
    elif (len(s_uniq) == 1 and s_uniq[0] == 1):
        print("checkers")
        return 1
    if (cols[cols != tA] == 0):
        print("checkers")
        return 1 if s_1 > s_0 else 0
    
    if (len(S) == 0):    
        #O(cols * max unique attributes)
        x_j = Get_Best_Attribute(S,
                                 Var_Impurity(s_all, s_0, s_1),
                                 tA,
                                 cols[cols != tA])

        #O(n * )
        node = (x_j,)
        for val in np.unique(S[x_j]):
            S_v = S[S[x_j] == val]
            if len(S_v) == 0:
                node = 1 if s_1 > s_0 else 0
            else:
                node = node + (ID3_Grow_Tree
                              (S_v,
                              tA,
                              cols[cols != x_j]),)

        return node
