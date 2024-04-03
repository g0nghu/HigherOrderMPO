import numpy as np
import Sub180221 as Sub
import HigherOrderMPO
import copy

def InitMps(Ns,Dp,Ds):
    T = [None]*Ns

    for i in range(Ns):
        Dl = min(Dp**i, Dp**(Ns-i), Ds)
        Dr = min(Dp**(Ns-i-1), Dp**(i+1), Ds)
        T[i] = np.random.rand(Dl,Dp,Dr)
    
    U = np.eye(np.shape(T[0])[0])
    for i in range(Ns):
        T[i],U = Sub.Mps_QRP(U,T[i])
    
    return T

# Time evolution for dt
def SingleStep(T0,O):
    T = T0.copy()
    Ns = len(T)
    
    MPO_bond = np.shape(O)[0]
    Dp = np.shape(O)[1]
    L = np.zeros([MPO_bond])
    R = np.zeros([MPO_bond])
    L[0] = R[0] = 1

    # The first
    shape_T = np.shape(T[0])
    A = Sub.NCon([L,O,T[0]],[[1],[1,-2,-3,2],[-1,2,-4]])
    A = Sub.Group(A,[[0,1],[2,3]])
    T[0],S,V,Dc= Sub.SplitSvd_Lapack(A,shape_T[-1],iweight=1)
    T[0] = np.reshape(T[0],[shape_T[0],shape_T[1],Dc])
    V = np.reshape(V,[Dc,MPO_bond,shape_T[-1]])

    # The middle
    for i in range(1,Ns-1):
        shape_T = np.shape(T[i])
        #V = np.tensordot(S,V,(1,0))
        #A =Sub.Ncon([V,T[i],O],[[-1,2,1],[2,-2,-3,3],[1,3,4]])
        A = Sub.NCon([np.diag(S),V,O,T[i]],[[-1,1],[1,2,3],[2,-2,-3,4],[3,4,-4]])
        A = Sub.Group(A,[[0,1],[2,3]])
        T[i],S,V,Dc= Sub.SplitSvd_Lapack(A,shape_T[-1],iweight=1)
        T[i] = np.reshape(T[i],[shape_T[0],shape_T[1],Dc])
        V = np.reshape(V,[Dc,MPO_bond,shape_T[-1]])

    # The final
    shape_T = np.shape(T[-1])
    T[-1] = Sub.NCon([np.diag(S),V,O,T[-1],R],[[-1,1],[1,2,3],[2,-2,5,4],[3,4,-3],[5]])

    # Normalization
    A = copy.copy(T[0])
    for j in range(1,Ns):
        A = np.tensordot(A,T[j],(-1,0))
    T_state = np.reshape(A,[Dp**Ns])
    Nom = np.linalg.norm(T_state)
    for i in range(len(T)):
        T[i] = T[i] / Nom
    print('Nom',Nom)
    #print(T)
    return T

def get_product(I,S,Ns,Dp):
    H = np.zeros([Dp**Ns,Dp**Ns])
    for i in range(Ns-1):
        A = 1
        for j in range(Ns):
            if j == i or j == i+1:
                A = np.kron(A,S)
            else:
                A = np.kron(A,I)
        H = H+A
    return H

# XXZ model

Ns = 6
Dp = 2
Ds = 6

MPO_bond = 5

MPO = np.zeros([MPO_bond,Dp,MPO_bond,Dp])
S0,Sp,Sm,Sz,Sx,Sy = Sub.SpinOper(2)
MPO[0,:,0,:] = S0
MPO[0,:,1,:] = 0.5*Sp
MPO[0,:,2,:] = 0.5*Sm
MPO[0,:,3,:] = 0.5*Sz
MPO[1,:,4,:] = Sz
MPO[2,:,4,:] = Sp
MPO[3,:,4,:] = Sm
MPO[4,:,4,:] = S0




N = 3
dt = -0.01j
T = InitMps(Ns,Dp,Ds)

O,levels,LEVELS = HigherOrderMPO.Powers_MPO(MPO,N)
O = HigherOrderMPO.time_evo_MPO(O,MPO_bond,LEVELS,N,-dt)

H = 0.5*get_product(S0,Sp,Ns,Dp) + 0.5*get_product(S0,Sm,Ns,Dp) + 0.5*get_product(S0,Sz,Ns,Dp)
Energies, Eigenstates = np.linalg.eigh(H)


t = np.linspace(0,0.1,11)
T1 = [None]*len(t)
T_exa = [None]*len(t)
T1[0] = copy.copy(T)

A = copy.copy(T[0])
for i in range(1,Ns):
    A = np.tensordot(A,T[i],(-1,0))
T_exa[0] = np.reshape(A,[Dp**Ns])
print(np.dot(T_exa[0],T_exa[0]))


for i in range(1,len(t)):
    print('time',t[i])
    T1[i] = SingleStep(T1[i-1],O)

    T_exa[i] = np.zeros([Dp**Ns],dtype=complex)
    for k in range(len(Energies)):
        T_exa[i] += np.inner(np.conj(Eigenstates[:,k]),T_exa[0])*np.exp(-1j*t[i]*Energies[k])*Eigenstates[:,k]

#print(T1[0])

for i in range(len(t)):
    A = T1[i][0]
    for j in range(1,Ns):
        A = np.tensordot(A,T1[i][j],(-1,0))
    T1[i] = np.reshape(A,[Dp**Ns])
print(np.dot(T1[0],T1[0]))


err = np.zeros_like(t)
for i in range(len(t)):
    err[i] = np.linalg.norm(T_exa[i]-T1[i])
    #print(np.linalg.norm(T_exa[i]))
    print(np.inner(np.conj(T1[i]),T1[i]))
#print(err)
#print(T1[2])


