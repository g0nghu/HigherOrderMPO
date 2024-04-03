import numpy as np
import numpy.linalg as LA
import Sub180221 as Sub
from itertools import permutations
from math import factorial

# get N powers of MPO
# MPO = I C D
#         A B
#           I
def Powers_MPO(MPO,N):
    # Here we always choose the first two indices as the physical indices
    shape_MPO = np.shape(MPO)
    MPO_bond = shape_MPO[0]
    O = np.copy(MPO)
    shape_levels = [3]
    shape_LEVELS = [MPO_bond]
    for i in range(N-1):
        O = Sub.NCon([O,MPO],[[-1,1,-5,-6],[-2,-3,-4,1]])
        O = Sub.Group(O,[[0,1],[2],[3,4],[5]])
        shape_levels.append(3)
        shape_LEVELS.append(MPO_bond)
    levels = np.arange(3**N)
    levels = np.reshape(levels,shape_levels)
    LEVELS = np.arange(MPO_bond**N)
    LEVELS = np.reshape(LEVELS,shape_LEVELS)
    O = O.astype(complex)
    return O,levels,LEVELS

# Change 1 in index into list[1,...,N-2]
def change_idx1(idx,MPO_bond):
    IDX = []
    for i in idx:
        if i == 1:
            IDX.append(list(range(1,MPO_bond-1)))
        elif i == 2:
            IDX.append([MPO_bond-1])
        else:
            IDX.append([0])
    return IDX

# Get the possible combination of index
def change_idx2(idx):
    row = len(idx)
    Col = [len(a) for a in idx]
    N = [0]*row
    N[row - 1] = 1
    for i in range (row -2, -1,-1):
        N[i] = N[i+1]*Col[i+1] # After the j th row has been determined, the combination number of the following rows is N[j]

    A = N[0]*Col[0] # The number of possible combination, the product of elements in Col
    IDX = [list() for i in range(A)]
    index = [0]*row
    for i in range(A):
        i_copy = i
        for j in range(row):
            index[j] = i_copy//N[j] #When i becomes larger than N[j] the corresponding index[j] + 1 
            i_copy = i_copy%N[j]
        for k in range(row):
            IDX[i].append(idx[k][index[k]])
    return IDX

# Change the tensor index into the matrix index
def change_idx3(idx,LEVElS):
    matrix_idx = []
    for i in idx:
        #print('idx',i)
        #print('corresponding matrix indices',LEVElS[tuple(i)])
        matrix_idx.append(LEVElS[tuple(i)])
    return matrix_idx
    
# Total change from 0,1,2 index into the matrix index
def change_idx(idx,MPO_bond,LEVELS):
    return change_idx3(change_idx2(change_idx1(idx,MPO_bond)),LEVELS)

# Incorporating higher-order terms
def Add_HigherOrder(MPO,dt,N):
    shape_MPO = np.shape(MPO)
    MPO_bond = shape_MPO[0]

    O,levels,LEVELS = Powers_MPO(MPO,N)
    O1,levels1,LEVELS1 = Powers_MPO(MPO,N+1)
    for b, b_idx in np.ndenumerate(levels):
        b = list(b)
        if 0 in b:
            continue
        b_matrix_idx = change_idx(b,MPO_bond,LEVELS)
        # we need to change the index into the form can be accepted for the 2-d Operator
        for a, a_idx in np.ndenumerate(levels):
            
            a = list(a)
            a_matrix_idx = change_idx(a,MPO_bond,LEVELS)
            #print('element',a,b,'\n',O[a_idx,:,b_idx,:])
            if 1 not in a and 2 in a:
                continue
            for c in range(0,N+1):
                ae = np.insert(a,c,0).tolist()
                n0 = ae.count(0)
                ae_matrix_idx = change_idx(ae,MPO_bond,LEVELS1)
                #print('a and b',a_matrix_idx,b_matrix_idx)
                for d in range(0,N+1):
                    be = np.insert(b,d,2).tolist()
                    n2 = be.count(2)
                    be_matrix_idx = change_idx(be,MPO_bond,LEVELS1)
                    #print(a_matrix_idx,b_matrix_idx)
                    #print('ae and be',ae_matrix_idx,be_matrix_idx)
                    #print('adding\n',ae,be,'\n',O1[levels1[tuple(ae)],:,levels1[tuple(be)],:])
                    for pos0 in range(len(a_matrix_idx)):
                        for pos1 in range(len(b_matrix_idx)):
                            #print('postion',a_matrix_idx[pos0],b_matrix_idx[pos1])
                            #print('element\n',O[a_matrix_idx[pos0],:,b_matrix_idx[pos1],:])
                            O[a_matrix_idx[pos0],:,b_matrix_idx[pos1],:] += dt*O1[ae_matrix_idx[pos0],:,be_matrix_idx[pos1],:]/((N+1)*n0*n2)
    return O

# change H^N into time evolution MPO
def time_evo_MPO(O,MPO_bond,LEVELS,N,dt):
    for i in range(N):
        # generate a = [0,0,...,2,2] with i+1's 2
        a = []
        for j in range(N):
            if j < N-i-1:
                a.append(0)
            else:
                a.append(2)
        #print('a',a)
        #print('permutations(a)',list(permutations(a)))

        for b in permutations(a):
            b_idx = change_idx(b,MPO_bond,LEVELS)
            #print('b',b)
            #print('b_idx',b_idx)
            O[:,:,0,:] = O[:,:,0,:]+((dt**(i+1))*factorial(N-i-1)/factorial(N))*O[:,:,b_idx[0],:]
            O[:,:,b_idx[0],:] = np.zeros_like(O[:,:,b_idx[0],:])
            O[b_idx[0],:,:,:] = np.zeros_like(O[b_idx[0],:,:,:])
    return O

if __name__ == "__main__":
    Test = {}
    Test['change_idx1'] = 0
    Test['change_idx2'] = 0
    Test['change_idx3'] = 0
    Test['change_idx'] = 0
    Test['Add_HigherOrder'] = 0
    Test['time_evo_MPO'] = 0

    if Test['change_idx1'] == 1:
        a = (1,)
        idx = list(a)
        IDX = change_idx1(idx,3)
        print(IDX)

    if Test['change_idx2'] == 1:
        idx = [[1]]
        #idx = [[1,1,2],[3]]
        IDX = change_idx2(idx)
        print(IDX)

    if Test['change_idx3'] == 1:
        LEVELS = np.array([0,1,2])
        #LEVELS = range(4**2)
        #LEVELS = np.reshape(LEVELS,[4,4])
        p = change_idx3(IDX,LEVELS)
        print(p)

    if Test['change_idx'] == 1:
        MPO_bond = 5
        LEVELS = range(MPO_bond**2)
        LEVELS = np.reshape(LEVELS,[MPO_bond,MPO_bond])
        idx = [1,1]
        IDX = change_idx(idx,MPO_bond,LEVELS)
        print(IDX)

    if Test['Add_HigherOrder'] == 1:
        # H = \sum Z_i Z_i+1
        S0,Sp,Sm,Sz,Sx,Sy = Sub.SpinOper(2)
        MPO = np.zeros([3,2,3,2])
        MPO[0,:,0,:] = MPO[2,:,2,:] = S0
        MPO[0,:,1,:] = MPO[1,:,2,:] = Sz
        
        dt = 0.2
        # The 1-order MPO including 2-order terms is
        # I C+{CD}*t/2      D+DD*t/2
        #   A+{AD}+{BC}*t/2 B+{BD}*t/2
        #                   I
        O = Add_HigherOrder(MPO,dt,2)
        print(O[1,:,4,:])
        def GetMpo(Dp):
            S0, Sp, Sm, Sz, Sx, Sy = Sub.SpinOper(Dp)

            Dmpo = 4
            MPO = np.zeros([Dmpo,Dp,Dmpo,Dp],dtype=complex)
            MPO[0,:,0,:] = S0
            MPO[0,:,1,:] = -2*Sz
            MPO[0,:,3,:] = -2*Sx
            MPO[1,:,2,:] = 2*Sy
            MPO[1,:,3,:] = 2*Sz
            MPO[2,:,3,:] = 2*Sz
            MPO[3,:,3,:] = S0

            return MPO 
        MPO = GetMpo(2)
        O = Add_HigherOrder(MPO,dt,1)
        print(O)

    if Test['time_evo_MPO'] == 1:
        # H = \sum Z_i Z_i+1
        S0,Sp,Sm,Sz,Sx,Sy = Sub.SpinOper(2)
        MPO = np.zeros([3,2,3,2])
        MPO[0,:,0,:] = MPO[2,:,2,:] = S0
        MPO[0,:,1,:] = MPO[1,:,2,:] = Sz
        
        O,levels,LEVELS= Powers_MPO(MPO,2)
        dt = 0.1
        O = time_evo_MPO(O,3,LEVELS,2,dt)
        # from 0 to 8 the corresponding values are 1,0.025,0,0.025,0.00125,0.0025,0,0.0025,0
        print(O[8,:,0,:])