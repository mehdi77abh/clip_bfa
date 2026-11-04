# Unique id for each custom layer's instantiation that is good and predictable for loading far_cfg
global _LOCAL_UID
_LOCAL_UID = 0

import torch.nn.functional as F
import math


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create custom connections (for demonstration, you can adjust as per your needs)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Unique ID generation for layer when it is insantiated
        #self.unique_id = str(uuid.uuid4())
        # FIXME: check if local uid solved the loading of far_cfg
        global _LOCAL_UID
        self.unique_id = str(_LOCAL_UID)
        _LOCAL_UID = _LOCAL_UID + 1


        # Foregt and rewire parameter:
        self.dead_idx = []
        self.conceal_idx = []
        self.weight_dim = []
        self.far_en = False
        self.param_counter_d = {} # {"weight_dim" : count}

        self.bfa_counter_d = {} # To track BFA per dimension

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        #print("forward call:")

        #print(len(input[0]))
        #print(input[0])

        if len(self.conceal_idx) != 0:
            self.far_en = True
        else:
            self.far_en = False

        #print(self.far_en)


        #out = F.linear(input, self.weight, self.bias)
        out = self.custom_f_linear1(input)#, self.weight, self.bias,far_en=self.far_en,dead_idx=self.dead_idx,conceal_idx=self.conceal_idx ,weight_dim=self.weight_dim)
        #print("out:")
        #print(out)
        return out




    def custom_f_linear1(self,input):#, weight, bias=None,far_en=False,dead_idx=[[1,7]],conceal_idx=[3],weight_dim=[7]):
        """
        Custom implementation of linear transformation.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_features).
            weight (torch.Tensor): Weight matrix of shape (output_features, input_features).
            bias (torch.Tensor): Optional bias vector of shape (output_features).

            conceal_idx includes input index to the most important parameters.
            dead_idx includes dead neurons output that needs to be forget and rewire 
            weight_dim included the dimension of the most important parameter that is going to be concealed (only inputs in this dimension needs to be forget and rewire)
            example: weight 73 means dimension 7 and neuron 3 in input
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_features).
        """

        batch_en = False
        #print(input.size())
        #print(input.dim())
        if input.dim() != 2:
            if input.size()[0]!=1:
                raise ValueError("Input tensor must be 2-dimensional (batch_size, input_features).")
            else:
                input = input.reshape(input.size()[1], input.size()[2])
                batch_en = True
        #if weight.dim() != 2:
        #    raise ValueError("Weight tensor must be 2-dimensional (output_features, input_features).")
        
        #print("weight shape:",weight.shape)
        #print("input shape:",input.shape)
        far_matmul = False

        # Perform the linear transformation (matrix multiplication)
        if False:
            output = input.matmul(self.weight.t())  # Equivalent to torch.mm(input, weight.t())
        else:
            # Perform separate matrix multiplications for each output feature dimension
            output_list = []
            for i in range(self.weight.shape[0]): # output_feature size

                # refresh input from original (we do not want non target neurons get wrong input)
                input_t = input.clone()

                if self.far_en:
                    for l in range(len(self.conceal_idx)):
                        if(self.weight_dim[l]==i):
                            target = self.conceal_idx[l]
                            deads = self.dead_idx[l]

                            #input_t = input.clone()
                            input_t[0,target]=input_t[0,target]/(len(deads)+1) # devide the input by devision factor
                            for j in range(len(deads)):
                                input_t[0,deads[j]]=input_t[0,target] # forget and rewire # FIXME: the new input could be a weighted average of previous value and the new value to not loose generaliability!
                                #input_t[0,deads[j]] = ((input_t[0,target]*weighted_average) + input_t[0,deads[j]])/(weighted_average+1)
                            #input[0,7]=input[0,3]
                            far_matmul = True

                #print("new x")
                #print(input_t)

                #print("weight_i:",weight[i, :])
                if far_matmul:
                    output_i = input_t.matmul(self.weight[i, :])
                    far_matmul = False
                else:
                    output_i = input.matmul(self.weight[i, :])
                output_list.append(output_i)
                #print("out_i:\n",output_i)
                del input_t

        
            # Concatenate the results along the second dimension
            output = torch.stack(output_list, dim=1)
            


        # Add bias if provided
        if self.bias is not None:
            #print("bias size:",bias.size())
            output += self.bias
        
        #print(output.size())
        #print(output.dim())
        if batch_en:
            output = output.reshape(1, output.size()[0], output.size()[1])

        return output

