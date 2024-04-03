import numpy as np
import numpy.linalg as LA
import Sub180221 as Sub

# get N powers of MPO
# MPO = I C D
#         A B
#           I
def Powers_MPO(MPO,N,if_kaiisone=False):
    shape_MPO = np.shape(MPO)
    ds = shape_MPO[1]
    d_bond = shape_MPO[0]

    I = MPO[0,:,0,:]
    C = MPO[0,:,1:-1,:]
    D = MPO[0,:,-1,:]
    A = MPO[1:-1,:,1:-1,:]
    B = MPO[1:-1,:,-1,:]

    O = np.copy(MPO)
    shape_levels = [3]
    for i in range(N-1):
        O = Sub.NCon([O,MPO],[[-1,1,-5,-6],[-2,-3,-4,1]])
        O = Sub.Group(O,[[0,1],[2],[3,4],[5]])
        shape_levels.append(3)
    levels = np.arange(3**N)
    levels = np.reshape(levels,shape_levels)
    return O,levels

# Incorporating higher-order terms
def Add_HigherOrder(MPO,dt,N):
    O,levels = Powers_MPO(MPO,N)
    O1,levels1 = Powers_MPO(MPO,N+1)

    for b, b_idx in np.ndenumerate(levels):
        b = list({b})
        if 0 in b:
            continue
        for a, a_idx in np.ndenumerate(levels):
            a = list({a})
            if 1 not in a and 2 in a:
                continue
            for c in range(0,N+1):
                ae = np.insert(a,c,0).tolist()
                n0 = ae.count(0)
                for d in range(0,N+1):
                    be = np.insert(b,d,2).tolist()
                    n2 = be.count(2)
                    O[a_idx,:,b_idx,:] = np.add(O[a_idx,:,b_idx,:], dt*O1[levels1[tuple(ae)],:,levels1[tuple(be)],:]/((N+1)*n0*n2))
    return O

if __name__ == "__main__":
    Test = {}
    Test['Powers_MPO'] = 0
    Test['Add_HigherOrder'] = 1

    if Test['Powers_MPO'] == 1:
        # H = \sum Z_i Z_i+1
        S0,Sp,Sm,Sz,Sx,Sy = Sub.SpinOper(2)
        MPO = np.zeros([3,2,3,2])
        MPO[0,:,0,:] = MPO[2,:,2,:] = S0
        MPO[0,:,1,:] = MPO[1,:,2,:] = Sz

        O1,levels1 = Powers_MPO(MPO,1)
        print(O1[0,:,1,:])
        O2,levels2 = Powers_MPO(MPO,2)
        print(levels2)
        for level in np.ndenumerate(levels2):
            print(level)
        # (2,2,2) of O3 is ZZZ
        O3,levels3 = Powers_MPO(MPO,3) 
        print(O3[0,:,13,:])

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
        O = Add_HigherOrder(MPO,dt,1)
        print(O[1,:,1,:]) 


