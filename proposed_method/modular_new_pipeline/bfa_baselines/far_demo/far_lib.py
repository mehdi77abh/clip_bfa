import os
from PIL import Image
import time
import torch
import GPUtil
import pickle
import json
import argparse
from functools import partial
from multiprocessing import Pool, set_start_method, Process, Queue
import gc
import struct
from collections import Counter




class FaR_lib():

    def __init__(self,FRACTION_SIZE=4,division_factor=2,MAX_FaR=1,model=None,inputs=None,device=None):
        print("FaR_Lib is called.")


        # FaR parameters
        self.FRACTION_SIZE = FRACTION_SIZE # Determines how many inputs per output feature can be used to FaR (input_feature size/FRACTION_SIZE)
        self.division_factor = division_factor # How many dead neuron gets selected per each FaR itteration to do FaR
        self.MAX_FaR = MAX_FaR # total number of far operation
        self.model = model
        self.inputs = inputs
        self.labels = None
        self.GPUs = GPUtil.getGPUs()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

        self.argparser()  




    def main(self):

        return


    def list_files_in_directory(self, directory_path):
        """
        List the names of files in a directory.

        Args:
            directory_path (str): The path to the directory.

        Returns:
            list: A list of file names in the directory.
        """
        file_names = []
        try:
            # List all files in the directory
            files = os.listdir(directory_path)

            # Filter out directories and keep only file names
            file_names = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]
        except OSError as e:
            print(f"An error occurred: {e}")

        return file_names



    # Function to process a batch of images
    def process_batch(self, image_paths, directory_path, processor, device):
        batch = [Image.open(directory_path + '/' + path).convert("RGB") for path in image_paths]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)

        return inputs

    # Function to get predictions for a batch
    def get_predictions(self, inputs, model):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = logits.argmax(-1)
        return predictions

    # Process and predict in batches
    def batch_predict(self, image_paths, directory_path, processor, model, device, batch_size):
        num_images = len(image_paths)
        predictions = []

        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            inputs = self.process_batch(batch_paths, directory_path, processor, device)

            start_time = time.time()
            batch_predictions = self.get_predictions(inputs, model)
            end_time = time.time()

            predictions.extend(batch_predictions.tolist())
            print(f"Processed batch {i//batch_size + 1}/{(num_images - 1)//batch_size + 1}, Time: {end_time - start_time:.2f} seconds")

        return predictions


    # Function to get predictions for a batch
    def get_accuracy_on_batch(self, model, inputs, labels):

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        #print(outputs)
        #print(logits)
        predictions = logits.argmax(-1).tolist()

        if len(labels) != 0:
            correct = 0
            for i in range(len(labels)):
                if labels[i]==predictions[i]:
                    correct += 1
            accuracy = (correct/len(labels)*100)
            print("Accuracy is: " ,accuracy)
            del outputs, logits, predictions
            torch.cuda.empty_cache()
            return accuracy
        else:
            print("Labels  are: \n" ,predictions)
            del outputs, logits
            torch.cuda.empty_cache()
            return predictions



    # Function to get confidence of model on input
    def get_confidence_of_input(self, model, inputs, labels):

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        #print(outputs)
        #print(logits)
        confidence = logits[0][labels[0]].cpu()
        #print(confidence)

        del outputs, logits
        torch.cuda.empty_cache()

        return confidence




    def get_gpu_stat(self,comment=':'):
        print("-------------------------------------")
        print(f"GPU status {comment}")
        print(torch.cuda.memory_summary())

    def save_far_cfg(self,my_dict,name):
        #with open(f'{name}.pickle', 'wb') as handle:
        #    pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{name}.json', 'w') as json_file:
            json.dump(my_dict, json_file,indent=4)



    def load_far_cfg(self,name):
        #with open(f'{name}.pickle', 'rb') as handle:
        #    my_dict = pickle.load(handle)
        
        with open(f'{name}.json', 'r') as json_file:
            my_dict = json.load(json_file)

    
        return my_dict



    def argparser(self):
        
        parser = argparse.ArgumentParser()
        #parser.add_argument('--ip', type=str, default='192.168.1.10')
        #parser.add_argument('--port', type=int, default=7)
        parser.add_argument('--train_en', action='store_true')  # default is False
        parser.add_argument('--bfa_en', action='store_true') 
        parser.add_argument('--far_en', action='store_true') 
        parser.add_argument('--cpu_en', action='store_true') 
        parser.add_argument('--org_en', action='store_true') 
        parser.add_argument('--load_cfg_en', action='store_true') 
        parser.add_argument('--conf_en', action='store_true') 
        parser.add_argument('--blackbox', action='store_true') 
        


        parser.add_argument('--frac_size', type=int, default=4)
        parser.add_argument('--div_fac', type=int, default=2)
        parser.add_argument('--max_far', type=int, default=1)
        parser.add_argument('--saved_id', type=int, default=1)
        parser.add_argument('--input_size', type=int, default=1)
        parser.add_argument('--bit_flip', type=int, default=-1) 
    

        args = parser.parse_args()


        self.FRACTION_SIZE = args.frac_size # Determines how many inputs per output feature can be used to FaR (input_feature size/FRACTION_SIZE)
        self.division_factor = args.div_fac # How many dead neuron gets selected per each FaR itteration to do FaR
        self.MAX_FaR = args.max_far # total number of far operation
        self.saved_id = args.saved_id
        self.input_size = args.input_size
        self.bit_flip = args.bit_flip # is this is set, real bit flip happens. -1 means multiple weight by -3.

        self.train_en = args.train_en
        self.bfa_en = args.bfa_en
        self.far_en = args.far_en
        self.cpu_en = args.cpu_en
        self.org_en = args.org_en
        self.load_cfg_en = args.load_cfg_en
        self.conf_en = args.conf_en # instead of correctness of prediction, focus on confidence of prediction
        self.blackbox = args.blackbox # if it is set to true, attacker calculates important parameters (gradient calculation) on original (non-FaR) model but applies BFA attack on the FaR model. this is more close to real scenario 

       

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FaR APIs $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$




    def find_layer_by_id(self, model, unique_id):
        for name, module in model.named_modules():

            if hasattr(module, 'unique_id') and module.unique_id == unique_id:
                return module
        return None



    def apply_far_cfg_to_model(self, model,far_cfg_d):
        print("\napply_far_cfg_to_model ...")


        for layer_id in far_cfg_d:
            module = self.find_layer_by_id(model, layer_id)

            print("module:", module)

            # update rewring cfg
            dead_idx = far_cfg_d[layer_id][0][0] # we only have one important param in each iteration
            weight_dim = far_cfg_d[layer_id][2][0]
            conceal_idx = far_cfg_d[layer_id][1][0]


            # to make sure we are not adding a repetitive FaR
            if (conceal_idx not in module.conceal_idx) or (weight_dim not in module.weight_dim):
                module.dead_idx.append(dead_idx)
                module.conceal_idx.append(conceal_idx)
                module.weight_dim.append(weight_dim)
            else:

                j = 0
                d_added = False
                for wd in module.weight_dim:
                    if (weight_dim==wd):
                        cl = module.conceal_idx[j]
                        if (cl==conceal_idx):
                            module.dead_idx[j] += dead_idx
                            d_added = True
                            break
                    j += 1

                if not d_added:
                    module.dead_idx.append(dead_idx)
                    module.conceal_idx.append(conceal_idx)
                    module.weight_dim.append(weight_dim)


            print("dead_idx:",module.dead_idx)
            print("conceal_idx:",module.conceal_idx)
            print("weight_dim:",module.weight_dim)


            # update the weights
            weight_clone = module.weight.clone()
            #print("orginal weight:")
            #print(weight_clone)

            #print(f"*previous value for dead {628}-{390}:",weight_clone[628,390])
            #print(f"*previous value for dead {628}-{85}:",weight_clone[628,85])
            #print(f"*previous value for dead {628}-{464}:",weight_clone[628,464])


            
            target_weight_value = weight_clone[weight_dim,conceal_idx]
            print("target_weight_value:",target_weight_value)

            #print(f"*previous value for dead {628}-{390}:",weight_clone[628,390])
            #print(f"*previous value for dead {628}-{85}:",weight_clone[628,85])
            for deads in dead_idx:
                #print(f"previous value for dead {weight_dim}-{deads}:",weight_clone[weight_dim,deads])
                weight_clone[weight_dim,deads] = target_weight_value # FIXME: the new weight could be a weighted average of previous value and the new value!
                #weight_clone[weight_dim,deads] = ((weighted_average*target_weight_value) + weight_clone[weight_dim,deads])/(weighted_average+1)
            #print("new weight:")
            #print(weight_clone)
            module.weight.data = weight_clone
            
        
        del module, weight_clone
        #del weight_clone
        torch.cuda.empty_cache()

        return model




    def process_elements(self,idx,range_indices,param_names):
        i=0
        param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
        param_name, param = param_names[param_idx]
        within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
        if ('weight' in str(param_name)):# and (('classifier' in str(param_name))):# or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):

            return idx#.item()# save the global index of parameter



    def get_module(self,model,param_name):
        attr_list = param_name.split('.')
        attr_list = attr_list[:-1]
        print("attr_list:",attr_list)
        target_layer = model
        for attr in attr_list:
            target_layer = getattr(target_layer,attr)
        #print("target_layer:", target_layer)
        #print(target_layer.weight.size())
        #print(target_layer.unique_id)
        if hasattr(target_layer,'param_counter_d'):
            print("param_counter_d:\n",target_layer.param_counter_d)
        print(target_layer)

        return target_layer


    def get_weightdim_concealid(self,target_layer,param_idx):
        weight_dim = int(param_idx / target_layer.weight.size()[1])
        conceal_id = param_idx % target_layer.weight.size()[1]
        #print(target_layer.weight.size())
        #input("enter")
        return weight_dim, conceal_id





    def top_indices_with_frequency(self,list_of_lists,count=20):
        #from collections import Counter
        # Flatten the list of lists into a single list of indices
        flattened_indices = [index for sublist in list_of_lists for index in sublist]

        # Count the frequency of each index
        index_frequency = Counter(flattened_indices)

        # Get the top 10 indices with the highest frequency
        top_indices = index_frequency.most_common(count)

        return top_indices




    #  Gradient Calculation
    def compute_gradient_wrt_input(self,model, sample_data,labels,bfa=False,bfa_params_l=[]):
        print("\ncompute gradient ...")

        important_params = []

        initial_bfa_params_l_len = len(bfa_params_l)

        # Calculate each param layer name and the range of its indices
        param_names = list(model.named_parameters())
        index = 0
        range_indices = []
        for name, param in param_names:
            range_indices.append(index + param.numel())
            index += param.numel()

        # to select a subset of inputs for the sake of time reduction if it is required
        #if bfa:
            #sample_l = [0,2,4,8,11,63]
        #    sample_l = [0,1,2,3,4,7,8,11,61,63]
        #else:
        #    sample_l = [0,1,2,3,4,7,8,11,61,63]

        #for sample in sample_l:
        for sample in range(self.input_size):

            print("sample:",sample)
            start_time = time.time()


            data, target =  sample_data[sample], labels[sample]#data[0], target[0]
            #print("target:",target)
            data.requires_grad = True  # Set requires_grad to True for computing gradients w.r.t input
            #print(data.size())
            #print(data.unsqueeze(0).size())
            output = model(data.unsqueeze(0))

            #print(output)

            #lib.get_gpu_stat(f'after model infer on sample {sample}:')




            loss = self.criterion(output.logits, target.unsqueeze(0))

            model.zero_grad()

            loss.backward()



            #lib.get_gpu_stat(f'after model loss calculation:')


            del output, loss

            


            # The gradient of the output w.r.t the input data

            # Retrieve and flatten all gradients
            grads = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.abs().flatten())
                

            #lib.get_gpu_stat(f'after grads:')

            # Concatenate all the flattened gradients to form a single vector
            all_grads = torch.cat(grads)

            del grads

            max_len = all_grads.size()[0]
            #print("len grads list: " , max_len)

            #FIXME: for time saving
            max_len = 60000

            #lib.get_gpu_stat(f'after all_grads:')

            #st = time.time()
            # Find the N largest gradients by absolute value
            top10_grads, top10_indices = torch.topk(all_grads.abs(), max_len)

            #et = time.time()
            #print(f"Elapsed time for top10_indices: {et - st} seconds")

            #print(top10_indices)

            #print("all_grads second important:", all_grads[86281567])

            del all_grads

            #lib.get_gpu_stat(f'after top10_grads:')

            torch.cuda.empty_cache()

            #lib.get_gpu_stat(f'after cleaning:')


            top10_indices_cpu = top10_indices.cpu().tolist()
            #print(top10_indices_cpu)


            # Map the top N gradient indices back to the respective parameters in the model
            important_weights = []

            if True:
                # multiprocessing to increase performance
                # Create a new function with args already set
                partial_process_element = partial(self.process_elements, range_indices=range_indices,param_names=param_names)


                #with Pool() as pool:
                #    important_weights = pool.map(partial_process_element, top10_indices_cpu)

                pool = Pool(4)
                important_weights = pool.map(partial_process_element, top10_indices_cpu)
                pool.close()
                pool.join()
                #print("wait 5 second to os clean files")
                #time.sleep(5)
                pool.terminate()



                important_weights = [item for item in important_weights if item is not None]


            else:
                for i, idx in enumerate(top10_indices):

                    param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
                    param_name, param = param_names[param_idx]
                    within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
                    if ('weight' in str(param_name)):# and (('classifier' in str(param_name))):# or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):
                        if i<1:
                            print("--------********")
                            print(f"Gradient rank: {i+1}")
                            print(f"Parameter name: {param_name}")
                            #print(f"Parameter size: {list(param.size())}")
                            print(f"Parameter index: {within_param_idx}")# index inside the layer
                            print(f"Gradient value: {top10_grads[i].item()}")
                            print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                            print(idx.item())
                            print()
                        important_weights.append(idx.item())# save the global index of parameter

            important_params.append(important_weights)# for each input, we save the most important params
            del important_weights

            end_time = time.time()
            print(f"Elapsed time for important grad: {end_time - start_time} seconds")



        conceal_l = important_params[0]

        print("conceal_l:\n",conceal_l[:10])
        #et= time.time()
        #print(f"Elapsed time for important grad: {et - st} seconds")

        important_param_d = {} # "param_name" : [within param indexes] -> per layer

        print("******** find one imp ------------")
        for i, idx in enumerate(conceal_l):

            param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
            param_name, param = param_names[param_idx]
            within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
            if ('weight' in str(param_name)):# and (('q_lin' in str(param_name)) or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):
                if True:
                    #print("********------------")
                    print(f"Gradient rank: {i+1}")
                    #print(f"Parameter name: {param_name}")
                    #print(f"Parameter size: {list(param.size())}")
                    #print(f"Parameter index: {within_param_idx}")
                    print(f"Gradient value: {top10_grads[i].item()}")
                    #print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                    print(idx)#.item())
                    #print()




                # check if the layer has enough param available, otherwise select the next one
                target_layer = self.get_module(model,param_name)
                if hasattr(target_layer, 'unique_id'):
                    pass
                else:
                    print("target_layer  does not have the 'unique_id'")
                    continue

                wdim, cid = self.get_weightdim_concealid(target_layer,within_param_idx)#.item())





                if not bfa:
                    # we do not want to FaR so many parameters connected to one neuron
                    if wdim in target_layer.param_counter_d:
                        if target_layer.param_counter_d[wdim]>=(target_layer.weight.shape[1]/self.FRACTION_SIZE): # input_feature size
                            print(f"we do not want to FaR so many parameters connected to one neuron. it is bigger than {(target_layer.weight.shape[1]/self.FRACTION_SIZE)}")
                            continue
                        else:
                            target_layer.param_counter_d[wdim] += 1
                    else:
                        target_layer.param_counter_d[wdim] = 1
                #elif idx.item() in bfa_params_l:
                elif idx in bfa_params_l:
                    # already this param is under BFA
                    print(f"already this param {idx} is under BFA")
                    continue
                else:

                    # Attacker does not want to attack one neuron several times
                    if wdim in target_layer.bfa_counter_d:
                        if target_layer.bfa_counter_d[wdim]>=(target_layer.weight.shape[1]/(self.FRACTION_SIZE-1)): # input_feature size
                            print(f"Attacker does not want to attack one neuron {wdim} several times")
                            continue
                        else:
                            target_layer.bfa_counter_d[wdim] += 1
                    else:
                        target_layer.bfa_counter_d[wdim] = 1

                    # select this param for BFA
                    #bfa_params_l.append(idx.item())
                    bfa_params_l.append(idx)



                if param_name not in important_param_d:
                    important_param_d[param_name] = [within_param_idx]#.item()]
                else:
                    important_param_d[param_name].append(within_param_idx)#.item())

                if True:#i<10:
                    print(f"\n--------**** selected important param, bfa:{bfa} ****------------")
                    print(f"Gradient rank: {i+1}")
                    print(f"Parameter name: {param_name}")
                    #print(f"Parameter size: {list(param.size())}")
                    print(f"Within Parameter index: {within_param_idx}")
                    print(f"Gradient value: {top10_grads[i].item()}")
                    #print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                    print(f"Parameter value: {param.view(-1)[within_param_idx]}")
                    #print("Global index: ",idx.item())
                    print("Global index: ",idx)
                    print()
                # we only need one important param per iteration
                break

        bfa_result = False
        if len(bfa_params_l)>initial_bfa_params_l_len:
            bfa_result = True


        torch.cuda.empty_cache()
        if bfa:
            return important_param_d, bfa_params_l, bfa_result
        else:
            return important_param_d



    def process_elements_dead(self, idx,range_indices,param_names,imp_param_name):
        #i=0

        param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
        param_name, param = param_names[param_idx]
        within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
        if ('weight' in str(param_name)) and (param_name==imp_param_name):# or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):
            
            if False:
                print("--------dead neuron grad info********")
                print(f"Gradient rank: {i+1}")
                print(f"Parameter name: {param_name}")
                #print(f"Parameter size: {list(param.size())}")
                print(f"Parameter index: {within_param_idx}")# index inside the layer
                #print(f"Gradient value: {top10_grads[i].item()}")
                print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                print(idx)#.item())
                print()
                i = i + 1
            
            return within_param_idx#.item()#.item()# save the global index of parameter



    def process_elements_dead_batch(self, batch,range_indices,param_names,imp_param_name,result_queue):
        #i=0
        try:
            ret = []
            #print("elemnt is called")
            for i, idx in enumerate(batch):
                param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
                param_name, param = param_names[param_idx]
                within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
                if ('weight' in str(param_name)) and (param_name==imp_param_name):# or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):
                    
                    if i<0:
                        print("--------********")
                        print(f"Gradient rank: {i+1}")
                        print(f"Parameter name: {param_name}")
                        #print(f"Parameter size: {list(param.size())}")
                        print(f"Parameter index: {within_param_idx}")# index inside the layer
                        #print(f"Gradient value: {top10_grads[i].item()}")
                        print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                        print(idx)#.item())
                        print()
                        i = i + 1
                    
                    #ret.append(within_param_idx.item())# save the global index of parameter
                    ret.append(within_param_idx)# save the global index of parameter
                else:
                    ret.append(None)
                
            #return ret
            #print(ret[:10])
            result_queue.put(ret)


        except Exception as e:
            print(f"An error occurred: {e}")

    def process_elements_batch(self, batch,range_indices,param_names):
        ret = [] 
        #i=0
        for i, idx in enumerate(batch):
            param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
            param_name, param = param_names[param_idx]
            within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
            if ('weight' in str(param_name)) and (('classifier' in str(param_name))):# or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):
                '''
                if i<0:
                    print("--------********")
                    print(f"Gradient rank: {i+1}")
                    print(f"Parameter name: {param_name}")
                    #print(f"Parameter size: {list(param.size())}")
                    print(f"Parameter index: {within_param_idx}")# index inside the layer
                    #print(f"Gradient value: {top10_grads[i].item()}")
                    print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                    print(idx)#.item())
                    print()
                    i = i + 1
                '''
                #return idx#.item()# save the global index of parameter
                ret.append(idx)
            else:
                ret.append(None)
        
        return ret
        



    def create_batches(self,data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


    # Gradient Calculation for dead neurons
    def get_dead_params(self, model,sample_data,labels,imp_param_name):
        print("\n ------------> compute gradient for deads")


        index = 0

        range_indices = []
        st_idx = 0
        end_idx = 0
        #print(imp_param_name)
        param_names = list(model.named_parameters())
        for name, param in param_names:
            range_indices.append(index + param.numel())
            if name==imp_param_name:
                st_idx = index
            index += param.numel()
            if name==imp_param_name:
                end_idx = index

        print("number of params to check: ", (end_idx-st_idx))

        important_params = []
        # a selection of representative data (for time saving!)
        #for sample in [0,1,2,3,4,7,8,11,61,63]:#range(len(labels.tolist())):
        for sample in range(self.input_size):
            print("sample:",sample)
            #start_time = time.time()

            data, target =  sample_data[sample], labels[sample]

            #print("target:",target)
            data.requires_grad = True  # Set requires_grad to True for computing gradients w.r.t input
            output = model(data.unsqueeze(0))

            loss = self.criterion(output.logits, target.unsqueeze(0))
            model.zero_grad()
            #print("backward loss ...")
            
            loss.backward()
            #end_time = time.time()
            #print(f"Elapsed time backward dead: {end_time - start_time} seconds")
            #start_time = time.time()

            #print("backward done!")

            del output, loss
            torch.cuda.empty_cache()

            #lib.get_gpu_stat(f'after dead param loss calculation:')

            

            # The gradient of the output w.r.t the input data
            #data_gradient = data.grad

            # Retrieve and flatten all gradients
            grads = []


            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.abs().flatten())
                    #print(name)



            # Concatenate all the flattened gradients to form a single vector
            all_grads = torch.cat(grads)

            #print("dead all_grads:\n",all_grads[:10])

            max_len = all_grads.size()[0]

            low10grads, low10_indices = torch.topk(all_grads, max_len, largest=False)



            #lib.get_gpu_stat(f'after dead param grad sorting:')

            del grads, low10grads
            torch.cuda.empty_cache()

            #lib.get_gpu_stat(f'after dead param cleaning:')

            #print("calculate sorted indices ...")
            #start_time = time.time()


            # putting this operation in GPU for faster computation
            st_idx = torch.tensor(st_idx, dtype=torch.int64, device=self.device)
            end_idx = torch.tensor(end_idx, dtype=torch.int64, device=self.device)

            low10_indices_tensor = torch.tensor(low10_indices, dtype=torch.int64, device=self.device)
            low10_indices_n_tensor = low10_indices_tensor[(low10_indices_tensor >= st_idx) & (low10_indices_tensor <= end_idx)]
            low10_indices_n = low10_indices_n_tensor.cpu().tolist()#.numpy()

            # now we have all indices of gradients sorted in important param

            #lib.get_gpu_stat(f'after dead param low10_indices_n:')


            print(len(low10_indices_n))
            #print("low10_indices_n:\n",low10_indices_n[:10])


            #p = []
            #for i in range(10):
            #    p.append(all_grads[low10_indices_n[i]])
            #print("all_grads low10_indices_n:\n",p)

            del low10_indices_tensor, low10_indices_n_tensor, all_grads
            torch.cuda.empty_cache()

            important_weights = []
            #end_time = time.time()
            #print(f"Elapsed time dead grad: {end_time - start_time} seconds")
            #print("Sorted.")
            start_time = time.time()

            # FIXME: just for testing, uncomment it later 
            #low10_indices_n = low10_indices_n[:100]


            # we already know that indices are for important param becaue of st_idx and  end_idx!!
 

            end_time = time.time()
            print(f"Elapsed time find deads: {end_time - start_time} seconds")
            #print(important_weights)

            #important_params.append(important_weights)
            important_params.append(low10_indices_n)

            del important_weights, low10_indices_n
            torch.cuda.empty_cache()

            #end_time = time.time()
            #print(f"Elapsed time dead3: {end_time - start_time} seconds")

        #print("len important_params",len(important_params))

        #print(important_params)

        dead_l = important_params[0]



        dead_param_d = {} # "param_name" : [within param indexes]

        # FIXME: just for testing, uncomment it later 
        #dead_l = dead_l*10000

        print(len(dead_l))
        #print(dead_l[:10])

        print("\n---> calculating dead_param_d...")
        start_time = time.time()

        gc.collect()

        if True:


            # multiprocessing to increase performance
            # Create a new function with args already set
            partial_process_element = partial(self.process_elements_dead, range_indices=range_indices,param_names=param_names,imp_param_name=imp_param_name)

            
            with Pool(8) as pool:
                within_param_idx_l = pool.map(partial_process_element, dead_l)


            within_param_idx_l = [item for item in within_param_idx_l if item is not None]

            # static assignmet
            dead_param_d[imp_param_name] = within_param_idx_l

        else:
            # since we know that we are using classifier layer only, we ignore dead_param_d[param_name]
            for i, idx in enumerate(dead_l):

                param_idx = next(i for i, v in enumerate(range_indices) if v > idx)
                param_name, param = param_names[param_idx]
                within_param_idx = idx - (range_indices[param_idx-1] if param_idx > 0 else 0)
                #if ('distilbert.transformer.layer.' in str(param_name)) and (('q_lin' in str(param_name)) or ('k_lin' in str(param_name))):# or ('v_lin' in str(param_name)))):
                #print(i,idx,param_name,within_param_idx)
                if ('weight' in str(param_name)) and (param_name==imp_param_name):
                    if i<1:
                        print(f"Lowest Gradient rank: {i+1}")
                        print(f"Parameter name: {param_name}")
                        #print(f"Parameter size: {list(param.size())}")
                        print(f"Parameter index: {within_param_idx}")
                        print(f"Parameter value: {param.view(-1)[within_param_idx].item()}")
                        print(idx.item())
                        print()

                    if param_name not in dead_param_d:
                        dead_param_d[param_name] = [within_param_idx.item()]
                    else:
                        dead_param_d[param_name].append(within_param_idx.item())

        end_time = time.time()
        print(f"Elapsed time find dead dic: {end_time - start_time} seconds")
        
        torch.cuda.empty_cache()
        print("returning dead_param_d...")
        return dead_param_d






    def get_forget_rewire_cfg(self,important_param_d,dead_param_d,model):
        print("\nget_forget_rewire_cfg ...")
        far_cfg_d = {} # {"layer_name" : [[[dead neurons]],[conceal id],[weight dims]]}
        for layer_name in important_param_d:
            if ("weight" in layer_name):# and ("fc4" in layer_name):
                #important_param_d[layer_name] = important_param_d[layer_name][:param_per_layer]

                target_layer_deads = dead_param_d[layer_name]
                # too much info to print this:
                #print("target_layer_deads: ",target_layer_deads)

                target_layer = self.get_module(model,layer_name)

                print("target_layer:",target_layer)

                weight_dims_l = []
                conceal_ids_l = []
                dead_ids_l = []


                for conceal_param in important_param_d[layer_name]:
                    #weight_dim = int(conceal_param / target_layer.weight.size()[0])
                    #conceal_id = conceal_param % target_layer.weight.size()[0]
                    weight_dim, conceal_id = self.get_weightdim_concealid(target_layer,conceal_param)
                    weight_dims_l.append(weight_dim)
                    conceal_ids_l.append(conceal_id)

                    #print(weight_dim,conceal_id)
                    #print(target_layer.weight.size())

                    i = 0
                    dead_ids_ll = []
                    for  params in target_layer_deads:
                        d_weight_dim = int(params / target_layer.weight.size()[1]) # it was 0???
                        d_conceal_id = params % target_layer.weight.size()[1]# it was 0
                        #print(f"d_weight_dim: {d_weight_dim}, d_conceal_id:{d_conceal_id}")
                        if (d_weight_dim==weight_dim) and (conceal_id!=d_conceal_id):
                            #print("weight_dim ",weight_dim)
                            #print("conceal_id ",conceal_id)
                            #print("d_conceal_id ",d_conceal_id)
                            #print("d_within_param_idx:", params)
                            #print("---")
                            
                            if len(target_layer.weight_dim)>0:
                                j = 0
                                same_found = False
                                for wd in target_layer.weight_dim:
                                    #print("wd ",wd)
                                    if (weight_dim==wd):
                                        cl = target_layer.conceal_idx[j]
                                        #print("cl ",cl)
                                        if (cl==conceal_id):
                                            same_found = True
                                            #print("target_layer.dead_idx: ",target_layer.dead_idx[j])

                                            if d_conceal_id not in target_layer.dead_idx[j]:
                                                dead_ids_ll.append(d_conceal_id)
                                                i += 1
                                            else:
                                                print("weight_dim ",weight_dim)
                                                print("conceal_id ",conceal_id)
                                                print("d_conceal_id ",d_conceal_id)
                                                print("d_within_param_idx:", params)
                                                print(f"This neuron {params} is already selected as dead for this param")
                                                print("---")
                                        #else:
                                        #    dead_ids_ll.append(d_conceal_id)
                                        #    i += 1
                                    j += 1
                                
                                if not same_found:
                                    #print("same not found!")
                                    dead_ids_ll.append(d_conceal_id)
                                    i += 1


                            else:
                                dead_ids_ll.append(d_conceal_id)
                                i += 1

                        #print("i:",i)
                        if i==self.division_factor:
                            break
                    dead_ids_l.append(dead_ids_ll)
                #print("\nlists of dim conceal dead:")
                #print(weight_dims_l)
                #print(conceal_ids_l)
                #print(dead_ids_l)
                #print()

                if layer_name not in far_cfg_d:
                    #far_cfg_d[layer_name] = [dead_ids_l,conceal_ids_l,weight_dims_l]
                    far_cfg_d[target_layer.unique_id] = [dead_ids_l,conceal_ids_l,weight_dims_l]
                else:
                    print("Error! similar layers?")

        return far_cfg_d




    #-------------------------- BFA APIs ---------------





    def float_to_binary(self, num):
        binary_representation = bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:]
        return binary_representation



    def float_tensor_to_binary(self, tensor):

        tensor_dtype = tensor.dtype
        #print(f"Data type of float tensor: {tensor_dtype}")

        # Convert the tensor to a numpy array and extract the bytes
        tensor_f = float(tensor.detach().cpu().numpy())
        #print(tensor_f)

        # Convert bytes to binary string
        binary_representation = self.float_to_binary(tensor_f)

        while len(binary_representation) < 32:
            binary_representation = '0' + binary_representation

        return binary_representation


    def bit_flip(self, binary_representation='0',bit=0, direction=0): # 0: 1->0, 1: 0->1

        if binary_representation[bit] == '0':
            nbinary_representation = binary_representation[0:bit] + '1' + binary_representation[bit+1:]
        else:
            nbinary_representation = binary_representation[0:bit] + '0' + binary_representation[bit+1:]

        return nbinary_representation



    def binary_to_float(self, binary_representation):
        # Convert binary string to integer
        integer_value = int(binary_representation, 2)

        # Interpret the integer as a single-precision (32-bit) float
        float_value = struct.unpack('!f', struct.pack('!I', integer_value))[0]

        return float_value




    def bfa_on_selected_param(self, important_param_d, model, bit_flips):


        for layer_name in important_param_d:

            if ("weight" in layer_name):# and ("fc4" in layer_name):

                
                target_layer = self.get_module(model,layer_name)

                weight_dim = 0
                conceal_id = 0
                for conceal_param in important_param_d[layer_name]:

                    weight_dim, conceal_id = self.get_weightdim_concealid(target_layer,conceal_param)
                    print("indexes to apply bit flip:", weight_dim, conceal_id)

                weight_clone = target_layer.weight.clone()
                weight_shape = list(target_layer.weight.shape)
                print("weight_shape:",weight_shape)


                w_val = weight_clone[weight_dim,conceal_id]
                print(f"Float tensor: {w_val.item()}")
                if bit_flips>-1:
                    # FIXME: use this part if you want to perform real BFA, not value change only
                    binary_representation = self.float_tensor_to_binary(w_val)
                    #print(f"Binary representation: {binary_representation}")
                    faulty = self.bit_flip(binary_representation,bit_flips) # attack to sign bit
                    new_weight_value = self.binary_to_float(faulty)
                else:
                    # for faster evaluation, we change the param value instead of performing BFA 
                    new_weight_value = w_val.detach().cpu().numpy()
                    new_weight_value = new_weight_value * -3

                print(f"new Float val: {new_weight_value}")

                new_weight_tensor = torch.full(weight_shape, new_weight_value, requires_grad=True)

                # Assign the new weight tensor to the selected weight index
                with torch.no_grad():
                    try:
                        if weight_shape[0] > 1:
                            new_weight_tensor = target_layer.weight[weight_dim].clone()
                            new_weight_tensor[conceal_id] = new_weight_value

                    except:
                        print('no array')
                        pass
                    # 1 or more than one bit fip
                    target_layer.weight[weight_dim].copy_(new_weight_tensor)
                    #linear_layer.weight[row_indices2[0]].copy_(new_weight_tensor2)
                    #linear_layer.weight[row_indices3[0]].copy_(new_weight_tensor3)
                #print(target_layer.weight)

        del target_layer, new_weight_tensor, new_weight_value, weight_clone
        torch.cuda.empty_cache()

        return model




    def apply_bfa_til_damage(self, model,sample_data,labels,bfa_params_l,far_model):

        print("\n[BFA] apply_bfa_til_damage ...")

        # get the important param from orginal model (the one that attacker has access to it)
        important_param_d, bfa_params_l, bfa_result = self.compute_gradient_wrt_input(model, sample_data,labels,True,bfa_params_l)#,FRACTION_SIZE=FRACTION_SIZE,input_size=input_size) 

        # apply BFA on the deployed model whcih is the one has FaR config applied on (attacker does not have knwoledge of FaR config)
        far_model = self.bfa_on_selected_param(important_param_d, far_model, self.bit_flip)


        # apply BFA on the attackers model whcih is the one has no FaR config , but attacker itself needs to track its BFAs
        model = self.bfa_on_selected_param(important_param_d, model, self.bit_flip)


        del important_param_d
        torch.cuda.empty_cache()

        return model, bfa_params_l, bfa_result, far_model


