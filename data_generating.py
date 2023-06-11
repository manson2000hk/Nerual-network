#Generate data for training

import math
import numpy as np
import os

def forward(input , weight , bias):
    return np.dot(input , weight) + bias
def create_w(inp, outp):
        # w = np.random.random(size=(inp , outp) )
        w = np.random.choice(10, size = (inp,outp))
        return w
def create_b(outp):
        # b = np.random.random(size=(1 , outp) )
        b = np.random.choice(10, size = (1 , outp))
        return b


# Input the size of input, the number of layers, the size of each layer, the sie of output, it will generate Input and output for training

input_size = int(input("How large is the input ? "))
input_size = int(input_size)

output_size = int(input("How large is the output ? "))
output_size = int(output_size)

n_layer = int(input("How many layer in the nerual network ? "))
n_layer = int(n_layer)

n_w_list = []
w_list = []
b_list = []

# Generate weights and bias

if n_layer != 0:
      
    n_w_list.append(input_size)
    for i in range(n_layer):
        
        n_w = input(f"How many neural in {i+1} layer ? ")
        n_w_list.append(int(n_w))
    n_w_list.append(output_size)
    # print("n_layer !=0 " , "\n", n_w_list)


    

    for i in range(len(n_w_list)-1):
        w_list.append(create_w(n_w_list[i] , n_w_list[i+1]))
        b_list.append(create_b(n_w_list[i+1]))
else:
     
     n_w_list.append(input_size)
     n_w_list.append(output_size)
    #  print("n_layer ==0 " , "\n", n_w_list)
     w_list.append(create_w(input_size , output_size))
     b_list.append(create_b(output_size))

print("w = " , "\n" , w_list , "\n")
print("b = " , "\n" , b_list , "\n")
# Creating random Input


training_size = int(input("How large is the training size ? "))
training_size = int(training_size)
Input = np.random.choice(10, size = (training_size,input_size))

# print("\n" , "Input = " , "\n" , Input.shape , "\n")
# print("\n" , "Input[0] = " , "\n" , Input[0] , "\n")

#Forward calculation

output_list = []
inp = Input
for i in range(n_layer+1):
     outp = forward(inp , w_list[i] , b_list[i])
     output_list.append(outp)
     inp = outp 

# print(output_list , "last = " , "\n" ,outp)

Output = outp
print("Input = " , "\n" , Input , "\n")
print("Output = " , "\n" , Output) 
