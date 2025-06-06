import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


with open("mammogram cancer image dataset/csv/calc_case_description_train_set.csv") as f:
    index = 0
    allLines = [] 
    for line in f:
        allLines.append(line)
    for i in range(1, len(allLines),2):
        #get all lines, since lines are two at one time
        print(allLines[i].split(","), allLines[i-1].split(",")[-1])
        print(allLines[i].split(",")[-2])
        image = mpimg.imread(f'mammogram cancer image dataset/jpeg/{allLines[i].split(",")[-2]}.jpg')

        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Optional: Turn off the axes
        plt.show()
        input("")
        



train_set = []
test_set = []


def sigmoid(num):
    return(1/(1+np.exp(-num)))
sigmoid_vectorized = np.vectorize(sigmoid)


def activationFuncDerivative(activationFunc,x):
    return activationFunc(x)*(1-activationFunc(x))


#needs editing
def test(inputs, w, b, activationFunc):
    correct = 0
    pCorrect= 0
    for inp in inputs: 
        As ={}
        As[0] = inp[0]
        dots = {}        
        for layer in range(1, len(w)): 
            dots[layer] = (w[layer]@As[layer-1])+b[layer]
            As[layer] = activationFunc(dots[layer])
        max_i = 0
        for i in range(len(As[len(w)-1])):
            if(As[len(w)-1][i, 0] > As[len(w)-1][max_i, 0]):
                max_i = i
        if(inp[1][max_i, 0] == 1):
            correct +=1
            if(max_i == 0):
                pCorrect+=1
    print(correct) 
    print(pCorrect)
    return correct/len(inputs)
       

def back_propagation(inputs, w, b, activationFunc, learningRate, epochs):
    for epoch in range(epochs):     
        for ind, inp in enumerate(inputs): 
            if(ind%1000 == 0):
                print(ind)
            As ={}
            As[0] = inp[0]
            dots = {}        
            for layer in range(1, len(w)):
                dots[layer] = (w[layer]@As[layer-1])+b[layer]
                As[layer] = activationFunc(dots[layer])
            deltas= {}
            deltas[len(w)-1] = activationFuncDerivative(activationFunc, dots[len(w)-1]) * (inp[1]-As[len(w)-1])
            
            for layer in range(len(w)-2, 0,-1):
                deltas[layer] = activationFuncDerivative(activationFunc, dots[layer]) *(np.transpose(w[layer+1])@deltas[layer+1])
            for layer in range(1, len(w)):
                b[layer] = b[layer]+learningRate*deltas[layer]
                w[layer] = w[layer]+learningRate*deltas[layer] *np.transpose(As[layer-1])
        with open("w_b_mammogram_tumor.pkl", "wb") as f:
            pickle.dump((w1,b1), f)
        print("w/b saved")
        print(str(test(test_set, w1, b1, sigmoid)) + "% \n")
    return (w,b)

def create_rand_values(dimensions):
    weights= [None]
    biases = [None]
    for i in range(1,len(dimensions)):
        weights.append(2*np.random.rand(dimensions[i],dimensions[i-1]) - 1)
        biases.append(2*np.random.rand(dimensions[i],1)-1)
    return weights, biases


w1, b1 = create_rand_values([4096,1000, 300,100, 2])
 
# with open("w_b_mammogram_tumor.pkl", "rb") as f:
#     w1,b1 = pickle.load(f)
print(str(test(test_set, w1, b1, sigmoid)) + "% \n")
w1, b1 = back_propagation(train_set, w1, b1, sigmoid, 0.01, 5)
print(str(test(test_set, w1, b1, sigmoid)) + "% \n")
