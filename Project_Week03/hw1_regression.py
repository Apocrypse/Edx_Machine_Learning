import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():

    n, d = X_train.shape
    X = np.matrix(X_train)
    y = np.matrix(y_train).T
    I = np.identity(d)

    wRR = (lambda_input * I + X.T * X) ** - 1 * X.T * y

    return wRR


wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file

    n, d = X_train.shape
    X = np.matrix(X_train)
    x = np.matrix(X_test)
    I = np.identity(d)

    Sigma = (lambda_input * I + sigma2_input ** -1 * X.T * X) ** - 1

    obj = np.array(X * Sigma * X.T)

    return np.argsort(obj)[::-1][:10]


active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
