import numpy as np
import math
import time


"""################### Training set 1 #############################

Input = np.array([[2,3,3,7,1,8,7,8,1,9,1,5,9,4,4,4,8,9],	[7,4,2,5,6,3,1,4,6,9,3,1,3,3,5,9,2,3],	[4,5,5,2,6,5,4,1,2,4,4,9,1,1,9,7,3,1],	[6,1,5,6,9,6,7,8,1,3,2,9,2,5,4,2,6,8],	[2,1,4,8,8,2,5,8,5,7,5,4,5,7,1,2,7,7],	[8,9,6,8,2,7,8,4,2,6,3,6,7,7,4,5,6,8],	[6,8,5,1,5,5,9,2,5,8,5,9,7,7,8,6,7,9],	[3,1,8,4,4,3,8,1,2,3,2,2,7,4,2,8,5,6],	[9,7,1,9,3,4,3,9,6,6,8,8,5,4,9,2,1,2],	[4,7,3,6,4,7,1,5,2,6,5,6,4,2,5,9,3,1],	[4,4,8,1,1,6,9,1,9,4,2,7,5,8,2,6,5,1],	[4,7,8,1,5,7,4,7,5,6,7,3,4,6,3,4,2,6],	[5,3,2,6,4,8,6,4,2,2,7,3,1,5,8,8,8,1],	[8,8,7,7,5,8,7,4,6,1,5,7,2,1,9,5,5,3],	[5,8,9,6,4,4,5,6,1,9,4,4,5,3,1,5,9,5],	[6,6,3,4,3,1,1,5,1,5,8,9,1,2,4,9,7,9],	[4,2,3,9,8,4,6,7,2,8,4,4,5,5,8,8,9,4],	[5,5,9,7,6,8,1,2,1,6,6,4,5,2,8,6,3,3],	[5,4,3,6,3,8,9,8,4,5,7,1,8,4,3,3,5,4],	[4,8,1,7,3,1,1,3,2,3,5,8,7,4,1,2,8,1]])
Output = np.array([[997],	[726],	[707],	[865],	[893],	[995],	[1170],	[760],	[854],	[754],	[806],	[823],	[840],	
                   [845],	[856],	[913],	[1034],	[789],	[842],	[671]])

"""###############################################################


"""################### Training set 2 #############################

Input = np.array([[1,4,3,1,3],	[2,3,4,4,6],	[4,6,5,6,3],	[1,1,1,2,2],	[1,5,1,5,5],	[6,4,1,1,3],	[6,3,6,6,6],	[4,5,1,6,1],	[1,1,2,6,6],	[4,3,2,5,1],	[2,3,4,2,1],	[3,5,5,6,2],	[5,4,4,5,3],	[1,4,4,4,3],	[5,3,4,5,5],	[2,6,6,6,5],	[4,6,4,2,6],	[1,4,3,3,1],	[6,2,4,1,5],	[1,3,6,2,5],	[5,5,5,1,6],	[3,3,2,5,3],	[2,4,4,1,6],	[5,2,6,3,6],	[4,5,3,5,5],	[4,2,1,4,2],	[4,4,2,1,2],	[3,1,5,4,4],	[4,4,3,4,6],	[4,4,5,6,5]])

Output = np.array([[56],	[92],	[101],	[38],	[83],	[58],	[118],	[70],	[86],	[63],	[52],	[90],	[88],	[75],	[97],	[113],	[95],	[54],	[76],	[82],	[93],	[73],	[80],	[98],	[97],	[57],	[52],	[80],	[95],	[107]])


"""################################################################

##################### Training Set 3   ################################

Input = np.array([[4,8,3,5,1,8,4,8,6,5,4,2,6,2,2,6,8,8,5,6],	[3,9,1,1,1,6,4,3,5,3,6,9,2,4,3,6,1,1,4,1],	[4,4,2,4,5,3,1,5,3,1,3,9,2,4,2,9,7,3,7,5],	[1,2,8,4,5,1,2,1,4,9,4,7,3,7,2,4,8,4,1,6],	[2,9,5,7,1,1,8,8,5,6,9,5,2,8,6,5,5,8,9,5],	[1,2,9,5,7,9,3,9,6,9,6,8,8,3,8,1,4,2,5,4],	[8,1,1,3,3,6,7,3,5,7,4,2,2,1,9,4,4,5,5,7],	[6,1,4,5,1,4,3,5,2,2,1,5,3,7,8,1,1,1,1,3],	[1,5,5,6,5,4,8,1,6,9,3,3,3,1,9,3,8,3,4,6],	[2,2,6,8,3,2,6,7,7,4,2,7,9,7,2,7,6,2,4,7],	[4,1,3,6,7,5,1,6,2,5,7,1,7,4,6,7,4,3,8,1],	[6,9,5,3,5,4,4,2,3,1,1,7,8,4,1,9,9,3,2,2],	[2,6,7,3,1,4,2,6,3,7,3,8,2,5,3,3,7,6,1,8],	[1,8,2,2,2,6,2,2,2,5,3,6,4,5,2,3,1,3,6,9],	[3,4,5,6,6,2,6,7,5,2,3,9,5,9,7,7,2,3,8,7],	[7,5,7,6,9,1,7,6,2,9,4,3,8,8,6,4,5,3,9,8],	[3,6,7,5,8,2,4,2,9,7,3,3,5,9,3,7,7,4,9,7],	[9,5,3,5,6,8,3,2,8,2,1,5,3,1,7,7,3,6,8,8],	[9,6,6,3,7,1,5,7,7,3,3,2,8,3,2,5,4,1,3,1],	[1,9,5,3,7,4,5,3,4,3,5,8,4,4,6,8,6,4,7,9],	[4,8,4,9,4,5,7,8,8,2,7,5,1,2,6,5,6,3,7,1],	[9,9,3,1,3,1,8,8,7,9,6,2,9,8,5,9,6,4,8,6],	[6,4,9,3,9,1,9,8,3,8,7,3,8,1,8,5,3,5,8,5],	[1,6,7,3,1,3,1,9,1,6,2,2,2,3,1,1,3,2,7,1],	[9,8,6,6,9,4,9,3,8,1,1,8,8,5,9,9,9,7,4,5],	[2,4,8,5,8,7,4,5,1,1,2,8,9,1,8,1,8,9,2,9],	[9,3,9,4,5,1,8,4,1,6,9,8,8,6,9,5,4,4,9,5],	[3,2,7,9,7,8,5,2,6,9,9,9,1,3,4,9,6,2,3,9],	[4,1,5,4,8,6,9,8,2,8,2,7,6,6,5,2,6,2,5,2],	[6,3,9,8,9,4,8,1,2,8,3,2,2,6,2,3,8,2,9,8],	[8,4,3,6,2,7,9,8,3,4,7,9,3,3,3,2,5,3,7,5],	[4,7,3,6,3,9,5,5,7,6,8,8,2,9,3,6,6,4,8,9],	[9,1,7,8,5,3,3,1,3,1,4,3,8,9,6,8,2,9,7,7],	[9,5,1,2,8,7,9,5,7,2,8,6,6,3,7,5,9,4,4,2],	[6,6,9,3,4,6,7,9,3,3,4,1,8,3,8,1,8,6,8,8],	[3,5,7,5,3,6,7,7,1,6,6,5,2,5,7,6,8,8,1,7],	[8,1,9,8,5,1,8,7,1,6,1,8,9,3,8,2,9,6,8,2],	[4,1,4,9,9,6,3,6,8,2,4,6,7,5,6,1,1,8,4,4],	[9,6,2,1,5,8,9,7,8,3,6,4,1,4,7,3,5,3,2,9],	[4,9,6,5,9,6,7,9,7,4,2,2,7,1,1,1,6,3,6,3],	[1,6,3,6,5,9,8,2,7,2,6,8,3,7,2,3,6,7,8,5],	[3,1,4,9,7,6,1,3,3,2,3,3,5,7,1,3,8,8,4,8],	[6,1,5,2,9,2,4,4,9,6,2,4,7,7,1,8,8,6,5,7],	[6,5,8,1,6,7,3,1,9,3,5,1,5,3,5,9,1,4,5,2],	[6,9,8,4,4,5,8,2,6,2,3,7,3,8,7,1,4,4,7,2],	[3,4,6,7,7,8,1,7,6,6,4,7,8,6,1,6,3,7,5,4],	[2,7,1,9,8,6,2,1,5,6,6,6,7,5,3,1,1,6,1,8],	[2,1,8,4,7,9,2,7,9,9,2,6,1,6,5,3,8,7,9,2],	[9,4,1,9,8,9,2,7,3,7,8,4,2,5,9,2,7,5,4,2],	[5,3,4,4,6,6,3,5,7,7,5,6,1,3,1,2,5,7,1,5],	[2,9,9,6,7,7,3,4,3,2,6,1,9,4,6,8,8,3,1,1],	[7,6,7,6,4,2,5,7,4,1,9,1,1,5,3,7,2,3,8,8],	[2,5,3,5,6,9,6,3,8,7,5,6,8,3,8,9,5,4,3,3],	[4,5,1,1,3,3,3,2,8,9,6,2,2,6,6,3,9,3,9,8],	[2,5,6,4,2,4,9,7,4,5,7,8,3,1,2,1,7,2,8,8],	[9,7,4,6,9,5,5,6,6,2,5,5,8,4,1,3,4,1,6,7],	[6,6,8,2,5,8,4,2,5,7,1,6,1,6,5,6,5,3,9,1],	[8,2,1,9,2,4,3,2,2,5,3,7,9,4,1,2,5,4,3,7],	[5,3,9,6,3,9,7,3,4,2,4,3,2,1,2,4,5,4,5,6],	[3,2,3,7,9,1,7,1,3,5,3,2,1,1,2,3,3,8,2,8],	[4,8,3,2,3,7,1,7,1,5,5,2,8,9,9,9,9,2,5,4],	[9,4,6,3,3,3,7,3,5,6,8,6,1,4,6,8,4,1,6,9],	[5,9,6,2,5,6,7,9,4,6,6,5,9,9,1,8,2,8,3,5],	[2,6,3,5,1,5,4,8,1,5,2,6,8,1,6,9,2,8,8,3],	[9,9,7,2,4,7,6,5,5,7,5,4,5,6,5,4,2,3,2,3],	[7,5,5,9,6,4,1,3,3,7,2,4,4,8,3,5,9,4,4,1],	[1,5,4,1,9,6,1,3,7,6,2,7,8,7,2,6,5,4,8,5],	[5,3,7,6,6,2,1,5,4,4,1,3,5,1,9,2,1,6,6,6],	[8,2,1,1,8,1,7,9,4,1,8,3,2,1,7,7,2,8,8,8],	[6,6,4,5,7,2,5,5,5,4,3,7,2,1,2,3,6,4,4,2],	[9,4,4,7,4,5,2,4,1,8,1,4,5,7,8,9,8,5,2,6],	[3,4,3,8,9,5,8,6,4,6,3,5,4,2,3,4,9,3,6,8],	[9,4,9,6,8,6,2,9,1,4,5,1,6,3,3,6,3,4,5,9],	[5,4,2,6,1,5,4,6,3,4,4,2,3,4,6,2,6,7,2,4],	[4,1,7,7,2,8,2,5,1,2,3,3,4,7,6,6,5,1,2,6],	[8,4,3,9,1,9,1,1,3,2,8,7,8,1,6,3,2,6,5,8],	[8,2,3,6,9,9,2,7,1,2,7,1,7,6,8,5,1,3,7,9],	[4,2,8,1,4,3,5,9,1,9,8,5,1,8,5,3,1,2,4,5],	[6,9,2,5,5,5,9,3,8,4,5,1,3,2,2,8,5,9,6,5],	[2,5,1,8,5,3,9,9,9,1,4,5,9,3,5,8,1,1,6,4],	[5,8,2,8,3,4,9,8,4,9,9,9,9,8,2,4,8,2,5,7],	[4,6,5,6,9,8,7,1,4,2,6,9,9,4,2,6,4,8,8,8],	[5,9,5,1,9,9,9,5,3,1,7,4,9,8,4,7,9,9,7,7],	[2,8,5,9,1,3,2,2,7,7,8,6,1,5,3,4,3,6,1,4],	[5,5,7,2,5,4,3,8,2,6,6,3,4,1,6,4,8,5,6,4],	[7,9,5,2,4,2,3,5,7,7,2,3,3,7,8,7,3,2,7,6],	[8,3,9,4,6,8,2,5,4,2,9,7,1,6,5,3,9,7,1,1],	[9,9,9,7,4,6,6,9,4,6,3,8,7,2,6,9,7,5,8,6],	[2,9,9,7,9,2,3,9,6,5,3,1,5,4,7,1,5,9,6,5],	[4,1,5,6,1,1,3,6,1,8,4,6,9,1,3,8,2,2,2,3],	[2,7,9,5,1,1,9,9,4,1,3,6,2,7,8,4,7,5,6,1],	[9,7,6,5,8,7,6,5,4,3,8,4,8,5,9,8,2,7,9,7],	[4,6,3,3,5,6,8,6,7,3,4,9,6,6,6,6,5,4,9,2],	[1,8,1,4,4,8,4,7,6,2,9,7,4,3,3,2,5,6,4,1],	[4,5,4,3,2,2,2,2,7,4,2,8,8,9,4,5,7,2,4,7],	[5,9,7,3,6,5,1,7,2,7,3,2,8,4,3,6,5,6,1,5],	[6,9,3,7,5,9,8,4,9,9,7,7,7,1,6,5,8,1,5,3],	[8,3,3,8,2,7,1,2,5,1,4,8,6,7,8,1,6,9,4,9],	[7,8,5,4,6,2,6,6,7,2,9,4,2,1,8,8,3,4,5,8],	[5,9,4,9,6,6,1,8,5,4,7,1,1,7,5,6,4,9,3,3]])
Output = np.array([[1114],	[745],	[985],	[951],	[1287],	[1122],	[985],	[647],	[1019],	[1122],	[980],	[901],	[980],	[866],	[1205],	[1259],	[1243],	[1088],	[773],	[1215],	[1001],	[1339],	[1183],	[623],	[1341],	[1145],	[1276],	[1204],	[1007],	[1071],	[1031],	[1322],	[1191],	[1125],	[1206],	[1162],	[1173],	[1017],	[1039],	[898],	[1158],	[1034],	[1191],	[894],	[983],	[1117],	[943],	[1182],	[1062],	[879],	[967],	[1005],	[1165],	[1150],	[1045],	[990],	[984],	[902],	[867],	[805],	[1189],	[1106],	[1195],	[1086],	[913],	[951],	[1126],	[894],	[1124],	[803],	[1133],	[1111],	[1026],	[873],	[883],	[1028],	[1112],	[926],	[1078],	[1032],	[1309],	[1279],	[1416],	[886],	[1019],	[1062],	[987],	[1332],	[1098],	[823],	[1032],	[1356],	[1184],	[926],	[1073],	[958],	[1165],	[1178],	[1103],	[1032]])


##########################################################################

#####################   Function defination  ##################

def sigmoid(x):


    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def forward(input , weight , bias):
    return np.dot(input , weight) + bias

class layer:

    def __init__(self , n_input , n_neu):
        self.n_input = n_input
        self.n_neu = n_neu
    
    def create_w(self):
        self.w = np.random.random(size=(self.n_input , self.n_neu)) 
        return self.w
    
    def create_b(self):
        self.b = np.random.rand(1,self.n_neu)
        return self.b
    
    def forward(self, inp , W, B):
        # self.out = sigmoid(np.dot(input, W) + B)
        self.out = np.dot(inp , W) + B
        return self.out
    
    def output_delta(self , lable , output):
        self.init_delta = 2 * (lable - output)
        return self.init_delta
    
    def output_error(self , output_delta , last_outpt):
        self.init_error = np.dot(last_outpt.T , output_delta)
        return self.init_error
    
    def layer_delta(self , last_delta, last_weight):
        self.l_delta = np.dot(last_delta , last_weight.T)
        return self.l_delta
    
    def layer_error(self , layer_delta , last_layer):
        self.l_error = np.dot(last_layer.T , layer_delta)
        return self.l_error
    
    def update_w(self , layer_er , w , learning_r):
        self.new_w = w + (learning_r * layer_er)
        return self.new_w
    def update_b(self , layer_del , b , learning_r):
        self.new_b = b + (learning_r * layer_del)
        return self.new_b
    
    def cross_entropy_loss(self , l , op):
        self.cel = -l*np.log(op)
        return self.cel
######################################################################


##################    Get nerual structure    #########################

n_layer = int(input("How many layer in the nerual network? "))
# input = np.array([])
n_layer = int(n_layer)
n_w_list = [Input.shape[1]]

for i in range(n_layer):
    
    n_w = input(f"How many neural in {i+1} layer ?")
    n_w_list.append(int(n_w))
n_w_list.append(Output.shape[1])

Start_T = time.time()

if (n_layer == 1) and (n_w_list[-1] == n_w_list[-2]):
    n_w_list.pop(-2)
    n_cal = n_layer
else:
    n_cal = n_layer + 1
# print(n_w_list)
##############################################################


##############    Create weight and bias (Code 2)    ##################

name_w = []
name_b = []

for i in range(len(n_w_list)-1):
    name_w.append(layer(n_w_list[i], n_w_list[i+1]).create_w())
    name_b.append(layer(n_w_list[i], n_w_list[i+1]).create_b())
name_w.append(np.array([[1]]))

########################################################################
# print(name_w)

############  Forward Calculation   ####################################
r = 0.00015
epoch = 1500
layer_out = []
delta_list = []
error_lisy = []
predit = []



for i in range(epoch):
    predit  =[]
    if (i)%(epoch/100) == 0 :                                            
        print("\r" , "|" * (int(100*i/epoch) ), " "*(99-int(100*i/epoch)),f"| progression: {math.ceil(100*i/(epoch-1))} %          ", end="")
    for j in range(Input.shape[0]):
        
        inp = Input[j]  
        layer_out = []
        delta_list = []
        error_list = []
        
        for k in range(n_cal):
            out = forward(inp , name_w[k] , name_b[k])
            layer_out.append(out)
            inp = out
            
            


            if k == n_cal-1:
                error = 2 * (Output[j] - layer_out[-1])
                layer_out.insert(0,np.array([Input[j]]))
                predit.append(layer_out[-1])
                delta = error
                # print("error = " , error , "\n")
                for l in range(n_cal):

                    delt = np.dot( delta , name_w[-(l+1)].T)
                    delta_list.append(delt)
                    delta = delt

                for m in range(len(delta_list)):
                    error_list.append(np.dot( layer_out[-(m+2)].T , delta_list[m] ))

                for n in range(len(error_list)):
                    name_w[-(n+2)] = name_w[-(n+2)] + error_list[n] * r     
                    name_b[-(n+1)] = name_b[-(n+1)] + delta_list[n] * r *10
print("\n" , "Weight = " ,"\n" ,   name_w[0].T, "\n")
print("Bias = ", "\n" , name_b[0], "\n")

############################################################################################

################## Print Predition and Total Error  ########################################

# print(predit)
predition = []
for i in predit:
    predition.append(i[0][0])

# print("Predition = " , "\n" , predition)
p = np.array([predition])
q = 0
for j in range(p.shape[1]):
    q = (Output[j] - p.T[j]) ** 2

print("\n" , "Total Training Error = " , math.sqrt(q))

#########################################################################

###################  Testing Accurancy   ################################

##################### Testing Set 3 #####################################
test_in = np.array([[2,1,2,9,4,6,8,7,7,5,8,5,3,8,4,2,7,8,3,9],	[4,6,4,2,6,3,1,7,4,2,3,3,9,1,1,8,1,2,7,3],	[7,8,4,9,5,3,3,5,1,9,1,5,8,6,6,8,5,7,7,7],	[2,4,2,8,1,3,4,9,6,4,1,8,6,9,7,1,7,1,1,1],	[5,9,7,5,8,7,2,9,6,5,3,8,1,5,1,3,7,5,1,7],	[3,8,6,9,4,6,8,7,6,9,2,3,6,1,7,5,2,1,6,8],	[6,1,2,2,1,3,8,4,5,1,5,2,2,4,6,3,3,4,9,6],	[4,2,2,9,4,7,9,3,8,8,4,8,4,9,5,5,2,9,2,3],	[5,5,8,1,7,1,8,3,7,3,2,2,5,3,1,5,8,8,6,8],	[3,9,4,5,9,2,2,2,9,3,1,8,2,5,9,1,9,9,9,6],	[5,6,1,5,7,1,1,8,6,3,5,5,4,1,9,2,2,6,6,9],	[2,8,7,4,3,5,6,5,9,5,9,2,8,9,2,1,6,1,4,7],	[8,4,1,6,6,4,1,8,7,2,3,8,7,2,5,3,1,2,5,4],	[2,4,3,4,7,8,9,2,5,1,4,1,2,6,7,1,2,9,2,4],	[8,8,3,4,7,8,7,6,6,5,7,2,7,2,7,2,9,9,5,5],	[8,7,4,6,4,9,2,3,5,2,6,6,6,3,9,3,3,7,7,3],	[5,7,2,7,3,1,9,6,8,4,3,8,1,8,8,8,8,9,9,7],	[5,3,3,2,2,3,8,4,4,4,3,2,7,8,7,1,8,4,3,9],	[2,1,6,3,5,5,5,1,9,9,5,6,9,7,6,3,5,1,4,1],	[3,6,2,4,1,6,4,3,4,3,5,6,7,8,2,3,7,3,2,1],	[6,8,2,2,6,1,7,8,5,3,2,4,9,9,3,6,7,2,4,8],	[9,2,7,3,5,9,5,1,5,8,4,6,6,8,4,4,5,3,2,3],	[4,6,8,6,3,8,7,1,8,3,9,8,6,2,4,4,2,9,1,6],	[9,2,9,8,9,3,2,1,7,4,6,6,3,8,9,4,3,7,5,8],	[9,5,2,8,5,3,6,5,7,6,7,1,8,2,4,2,5,5,8,5],	[8,2,1,8,5,5,7,6,8,2,8,5,6,2,3,5,7,9,8,6],	[8,9,6,3,6,7,8,9,5,5,5,5,4,1,7,8,3,5,4,2],	[6,3,3,4,5,3,6,8,5,3,5,9,3,2,3,2,8,8,6,7],	[5,6,5,2,1,8,3,3,9,6,4,7,5,7,5,8,9,5,3,4],	[7,7,5,1,7,9,5,6,5,6,9,3,9,3,5,2,6,5,3,1],	[5,9,5,6,2,3,9,1,4,6,8,2,1,1,5,3,5,3,5,2],	[6,1,1,8,3,4,9,3,7,2,7,8,9,3,5,6,7,5,5,7],	[6,2,6,3,3,7,6,5,7,3,7,3,8,7,3,2,6,8,3,1],	[6,5,6,9,9,2,6,9,1,3,2,9,3,9,4,2,7,5,1,6],	[8,7,9,6,7,5,2,7,8,3,4,8,4,1,3,8,1,3,1,2],	[5,8,2,9,3,9,1,5,7,9,9,4,9,4,7,7,3,1,1,8],	[7,9,9,7,7,1,8,9,5,9,3,7,7,4,5,2,2,5,9,1],	[3,5,8,7,3,5,7,4,5,5,9,4,4,6,8,6,7,4,2,6],	[1,2,5,6,9,3,6,2,4,8,4,2,7,4,2,9,1,7,6,7],	[9,8,3,2,3,9,4,7,8,6,1,8,2,6,3,5,7,6,5,7],	[3,9,6,6,9,7,1,3,6,2,6,3,7,3,3,2,2,7,9,6],	[9,6,6,1,3,2,8,5,8,9,2,4,7,6,4,4,5,5,8,1],	[2,6,1,3,9,8,7,8,6,3,4,5,7,7,4,4,6,6,8,9],	[6,7,4,1,5,1,6,7,9,3,6,6,4,8,7,5,6,5,3,1],	[5,5,3,1,7,4,5,9,8,1,6,1,7,6,4,7,2,5,5,1],	[2,4,7,8,5,3,1,8,5,3,5,7,5,1,1,7,4,8,1,8],	[5,1,1,4,6,8,7,2,2,1,9,6,1,4,8,1,4,5,6,3],	[6,9,1,6,1,9,2,4,5,8,3,1,6,2,7,8,9,8,9,5],	[4,2,7,3,9,7,4,1,8,5,9,3,5,2,7,8,1,8,2,3],	[2,4,4,5,3,7,7,7,6,2,2,8,5,9,8,7,7,8,8,7]])
test_out = np.array([[1239],	[813],	[1256],	[894],	[1000],	[1067],	[922],	[1139],	[1080],	[1232],	[1041],	[1069],	[871],	[874],	[1221],	[1067],	[1429],	[1067],	[1003],	[856],	[1126],	[981],	[1067],	[1201],	[1066],	[1244],	[1043],	[1123],	[1174],	[1008],	[814],	[1220],	[1008],	[1036],	[840],	[1138],	[1091],	[1155],	[1088],	[1152],	[1039],	[1065],	[1299],	[1053],	[961],	[1009],	[926],	[1254],	[1027],	[1395]])
###########################################################################

test_result = np.dot(test_in , name_w[0]) + name_b[0]

total_test_error = 0

for i in range(test_result.shape[0]):
    t_error = (test_out[i] - test_result[i]) ** 2
    total_test_error += t_error

print("\n" , "Total Testing error = " , math.sqrt(total_test_error))
print("\n" , "Duration : " , round((time.time() - Start_T) , 4) , "s")
