import numpy as np
import math

def jacobi_SVD(matrix):
    '''
    :param matrix:
    :param V:
    :return: SVD分解后的U，sigma，V。T
    '''
    #matrix=m*n
    U = matrix.copy()
    n = matrix.shape[1]#矩阵列数
    V = np.identity(n)#n阶单位矩阵
    iter=0
    tol=0.0000001
    for i in range(0, n-1):
        for j in range(i+1,n):
            tmp_i=i
            tmp_j=j
            #print(tmp_i,tmp_j)
            #选择第i与j列
            a_i=U[:,i].copy()#原来这是U
            a_j=U[:,j].copy()

            ele = a_i @ a_j
            # if(ele<=0.):
            #     continue
            rows = matrix.shape[0]
            cols = matrix.shape[1]

            ele1 = a_i @ a_i
            ele2 = a_j @ a_j

            if (ele1 < ele2):
                #print("swap")
                tmp = a_i
                U[:, i] = a_j
                U[:, j] = tmp

                tmp=V[:,i].copy()
                V[:,i]=V[:,j]
                V[:,j]=tmp

                tmp_i=j
                tmp_j=i
            #print(tmp_i,tmp_j)
            if (ele <= 0.):
                continue

            tao = (ele2 - ele1) / (2 * ele)
            t = np.sign(tao) / (np.abs(tao) + np.sqrt(1 + tao * tao))
            c = 1 / np.sqrt(1 + t * t)
            s = t * c

            #jacobi旋转矩阵
            J=np.identity(n)
            J[tmp_i,tmp_i]=c
            J[tmp_i,tmp_j]=s
            J[tmp_j,tmp_i]=-1*s
            J[tmp_j,tmp_j]=c
            #更新
            V=np.dot(V,J)
            U=np.dot(U,J)
            iter+=1
            #print("U------")
            #print(U[:,i]@U[:,j])

    # print("-----hisj-----")
    # print(U)
    # print(V)
    # print(U@V.T)

    E = []
    S=np.zeros(matrix.shape)
    print(" cols is ")
    print(cols)
    for i in range(cols):
        norm = np.linalg.norm(U[:, i])
        E.append(norm)
    print("E is")
    print(E)
    none_zeros=min(rows,cols)
    tmp_U=np.zeros((rows,rows))
    for i in range(none_zeros):
        S[i][i]=E[i]
        tmp_U[:,i]=U[:,i]/E[i]
    # for i in range(rows):
    #     S[i][i] = E[i]
    #     for j in range(none_zeros):
    #         tmp_U[i][j] = U[i][j]/E[j]
    U=tmp_U
    print("S is")
    print(S)
    return [U,S,V.T]

def mergeBycols(X1,X2):
    '''
    每两个子矩阵合并
    :return: 合并后的矩阵
    '''
    print("two two merge")
    B = np.concatenate((X1, X2), axis=1)
    print(B)
    return jacobi_SVD(B)
def nergeByrows(Y1,Y2):
    B = np.concatenate((Y1, Y2), axis=1)
    [U,S,V]=jacobi_SVD(B)
    return [V,S.T,U]
def ADTT_al(A,threshold):
    '''

    :param A:  输入的张量
    :param threshold: 分块列数阈值，若按列分块后，列数大于此参数，则继续分
    :return: 返回张量列分解结果
    '''
    #init
    cores=[]
    ranks=[1,2,2,2]
    C=A
    N = len(A.shape)
    #
    # temp_prod = int(1 * A.shape[0])
    # print(temp_prod, C.size)
    # C = np.reshape(C, (temp_prod, C.size // temp_prod))
    for k in range(1, N):# 张量阶数，跟最后分解为几个张量相乘有关
        print(k)
        #对张量或者矩阵展开
        print(C.shape)
        temp_prod = int(ranks[k - 1] * A.shape[k - 1])
        print(temp_prod, C.size)
        C = np.reshape(C, (temp_prod, C.size // temp_prod))
        # print("--------reshape----------")
        # print(C)
        #分块，按照k+1的维数分为Ik+1个子矩阵，与阈值比较,分块这里要保证行小于列数
        C_block=[]#用来存每个子矩阵
        #print(A.shape)
        M=math.ceil(C.shape[1]/C.shape[0])
        #M=A.shape[k]
        #M=A.shape[k+1] #分块数
        print(M)
        col=C.shape[1]//M#每个子矩阵的列数
        print(col)
        if col>threshold:
            col=threshold
            M=math.ceil(C.shape[1]/col)
        print(M)
        #分块
        C_block=np.array_split(C, M, axis=1)
        print("--------分块矩阵--------------")
        print(C_block)
        #这儿应该使用并行计算，当前先用串行代替
        #-------------------------------------------
        #n=C.shape[1]
        #J=np.identity(n)
        #J=[]#每个子矩阵的jacobi行列式
        U=[]#每个子矩阵的左奇异值
        #sigma=[]#每个子矩阵的右奇异值
        B=[]#左奇异值与sigma相乘
        for c_i in C_block:
            #_,tmp_J=jacobi_SVD(c_i)
            #J.append(tmp_J.T)
            #这里应该可以考虑不算U和sigma
            #tmp_B=c_i@tmp_J
            [tmp_U,tmp_S,tmp_V]=jacobi_SVD(c_i)
            print('----U and S-----')
            print(tmp_U)
            print("------")
            print(tmp_S)
            print("----block is-----")
            print(c_i)
            print(tmp_U@tmp_S@tmp_V)
            B.append(tmp_U@tmp_S)
            U.append(tmp_U)
            #B.append(tmp_B)
            #tmp_sigma=[]
            '''
            for i in range(tmp_B.shape[1]):  # 对B归一化，得到U和sigma
                norm = np.linalg.norm(tmp_B[:, i])
                tmp_sigma.append(norm)
                tmp_B[:, i] = tmp_B[:, i] / norm
            '''
            #U.append(tmp_B)
            #igma.append(tmp_sigma)
        #执行合并算法,按照两两合并
        print("------B is-----------")
        print(B)
        for i in range(len(B)-1):
            [U,tmp,V]=mergeBycols(B[i],B[i+1])
            print("res is:")
            print(U @ tmp @ V)
            B[i+1]=U@tmp
            #print(V.shape)
            #print("-------block J is-----------")
            #print(block_diag(J[i], J[i+1]).shape)
            #J[i+1]=V.T@block_diag(J[i], J[i+1])
        #res=B[-1]
        cores.append(U)
        C=U.T@C
        print("C.shape")
        print(C.shape)
    cores.append(C)
    return cores

if __name__=="__main__":
    #A=(np.random.rand(12)*10).reshape((3,4))
    #A=np.arange(1,19).reshape(3,6)
    A = np.arange(1,49).reshape(2,2,3,4)
    print("input tensor is:")
    print(A)
    cores = ADTT_al(A, threshold=10)
    print("core is")
    print(cores)
    '''
    [U,S,V]=jacobi_SVD(A)
    print(S)
    print(U)
    print(A)
    res=np.dot(np.dot(U,S),V.T)
    print(res)
    '''
