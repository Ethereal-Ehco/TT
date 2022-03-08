import numpy as np

def TT_SVD(A,accuracy):
    '''

    :param A: tensor
    :param accuracy:
    :return: core
    '''
    #初始化
    cores=[]
    C = A  # 临时张量C
    ranks = [1, ]  # TT-ranks
    N = len(A.shape) # 张量阶数

    delta = (accuracy / np.sqrt(N - 1)) * np.linalg.norm(A)  # 截断值
    for k in range(1, N):
        temp_prod = int(ranks[k - 1] * A.shape[k - 1])
        C = np.reshape(C, (temp_prod, C.size // temp_prod))

        # computing q-truncated SVD and sigma-rank:
        U, sigma, V = np.linalg.svd(C)
        q_rank = 1
        b = (U[:, :q_rank] * sigma[:q_rank]) @ V[:q_rank, :]
        while np.linalg.norm(C - b) > delta and q_rank != len(sigma):
            q_rank += 1
            b = (U[:, :q_rank] * sigma[:q_rank]) @ V[:q_rank, :]

        ranks.append(q_rank)

        # creating new core
        cores.append(np.reshape(U[:, :q_rank], (ranks[k - 1], A.shape[k - 1], ranks[k])))
        C = np.diag(sigma[:q_rank]) @ V[:q_rank, :]

        temp_core = np.zeros((C.shape[0], C.shape[1], 1))
    for i in np.arange(C.shape[0]):
        for j in np.arange(C.shape[1]):
            temp_core[i, j, 0] = C[i, j]
    cores.append(temp_core)  # the last core, doing this for common style
    return cores

def jacobi_SVD(matrix):
    #matrix=m*n
    U = matrix.copy()
    n = matrix.shape[1]#矩阵列数
    V = np.identity(n)#n阶单位矩阵
    iter=0
    tol=0.0001
    for i in range(0, n-1):
        for j in range(i+1,n):
            #选择第i与j列
            a_i=U[:,i]
            a_j=U[:,j]
            if np.dot(a_i,a_j)<tol:continue
            # Computing alpha,beta,gamma
            tao=(np.dot(a_j,a_j)-np.dot(a_i,a_i))/2*np.dot(a_i,a_j)
            t=np.sign(tao)/(np.abs(tao)+np.sqrt(1+tao*tao))
            c=1/np.sqrt(1+t*t)
            s=t*c

            #jacobi旋转矩阵
            J=np.identity(n)
            J[i,i]=c
            J[i,j]=-1*s
            J[j,i]=s
            J[j,j]=c
            #更新
            V=np.dot(V,J)
            U=np.dot(U,J)
            iter+=1

    sigma = np.zeros(n)
    print(iter)
    for i in range(n):#对U归一化
        norm = np.linalg.norm(U[:, i])
        sigma[i] = norm
        U[:, i] = U[:, i] / norm
    return [U,sigma,V]

def ADTT():

    return
if __name__=="__main__":
    #A=tl.tensor(np.arange(1,9).reshape(2,2,2))
    #cores=TT_SVD(A,0.)
    A=np.arange(1,9).reshape((2,4))
    [u,s,v]=np.linalg.svd(A)
    print('--------------------------')
    print(A)
    [U,ans,V]=jacobi_SVD(A)
    print(u,s,v)
    print('--------------------')
    print(U,ans,V)
    print('-------------------')
    sigma=np.zeros((4,4))
    for i in range(4):
        sigma[i,i]=ans[i]
    res=np.dot(np.dot(U,sigma),V)
    print(res)

    #print(np.dot(np.dot(U,ans)),
