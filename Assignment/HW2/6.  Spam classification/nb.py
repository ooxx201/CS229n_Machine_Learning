import numpy as np
k = 2

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    M, N = matrix.shape
    ###################
    freq_0 = np.sum(matrix[category == 0], axis = 0) + np.ones((N,))
    freq_1 = np.sum(matrix[category == 1], axis = 0) + np.ones((N,)) 
    f_0 = np.sum(category == 0) 
    f_1 = np.sum(category == 1) 

    log_phi_0 = np.log(freq_0) - np.log(f_0 + k) 
    log_phi_1 = np.log(freq_1) - np.log(f_1 + k)
    
    log_p_0 = np.log(f_0) - np.log(M)
    log_p_1 = np.log(f_1) - np.log(M)
    state = [log_phi_0, log_phi_1, log_p_0, log_p_1]
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    log_phi_0, log_phi_1, log_p_0, log_p_1 = state
    
    log_opt_0 = np.matmul(matrix, log_phi_0) + log_p_0
    log_opt_1 = np.matmul(matrix, log_phi_1) + log_p_1
    
    output = (log_opt_1 > log_opt_0).astype(int)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('spam_data/MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('spam_data/MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)
    
    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()