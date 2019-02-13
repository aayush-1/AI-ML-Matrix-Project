import numpy as np
import matplotlib.pyplot as plt
import math

A=np.array([1,-1])
n1=np.array([2,1])
n2=np.array([1,-1])
p1=3
p2=1

p=np.array([[p1],[p2]])
N=np.vstack((n1,n2))

center=np.matmul(np.linalg.inv(N),p)
dist=math.sqrt((center[0]-A[0])**2 + (center[1]-A[1])**2)
nmat=np.array([[0,1],[-1,0]])
dvec=A-center.T.ravel() #ravel converts N-D array to 1-D array

plt.grid()
plt.xlim(-2,5)
plt.ylim(-4,3)

def line_through_center():
    m=50
    check=np.zeros((2,m))
    space1=np.linspace(-3,3,m)
    for i in range(m):
        temp1 = A + space1[i]*(dvec)
        check[:,i]=(temp1.T).ravel()
    plt.plot(check[0,:],check[1,:],label='$Normal$')
   
line_through_center()

def normal_vector():
    return np.matmul(dvec,nmat)

nv=normal_vector()



def circle():
    len=100
    c=np.zeros(len+1)
    x=np.zeros(len+1)
    y=np.zeros(len+1)
    for i in range(len+1):
        c[i]=0 + i*(2*math.pi)/len
        x[i]=center[0] + dist*math.cos(c[i])
        y[i]=center[1] + dist*math.sin(c[i])
    plt.plot(1,-1,'o')
    plt.plot(x,y,label='$Circle$')
    plt.plot(center[0],center[1],'o')
    plt.text(center[0]*(1+0.05),center[1]*(1+0.05),'C')

circle()
l=50
ans=np.zeros((2,l))
space=np.linspace(-3,3,l)
for i in range(l):
    temp1 = A + space[i]*(nv)
    ans[:,i]=(temp1.T).ravel()
plt.plot(ans[0,:],ans[1,:],label='$Tangent$')
plt.legend(loc='best')
plt.show()
