import torch
import os
import sys
import time
import copy
import gc

import ssl
from multiprocessing import set_start_method

from far_lib import FaR_lib


'''
This code uses a pretrained ViT

'''

#-------------------------------------------- Main -------------------------------
#---------------------------------------------------------------------------------

def main():
    saved_id = 10


    lib = FaR_lib()
    ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

    global device

    if lib.cpu_en:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print('releasing gpu memory ...')
            torch.cuda.empty_cache()
    
    lib.device = device


    print("Selected device is: ", device)

    model_name = 'google/vit-base-patch16-224' 


    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained(model_name)

    if lib.org_en:
        from transformers import ViTForImageClassification as myvit
    else:
        # using local folder
        print("Local model architecture is selected.")
        from transformers-main.src.transformers.models.vit.modeling_vit import ViTForImageClassification as myvit

    


    #lib.get_gpu_stat('at start:')

    # model parameter is loaded from remote as pretrained
    model = myvit.from_pretrained(model_name).to(device)

    print(model)

    #lib.get_gpu_stat('after loading model start:')


    # Example usage:
    directory_path = './vitpic'
    file_list = lib.list_files_in_directory(directory_path)
    file_list = file_list[:lib.input_size]
    print(file_list)

    org_label = []

    # Load and preprocess image
    image_paths = file_list


    total_time = 0


    #lib.get_gpu_stat('before inputs:')

    inputs = lib.process_batch(image_paths, directory_path, processor, device)

    #lib.get_gpu_stat('after inputs:')

    labels = []
    labels = lib.get_accuracy_on_batch(model, inputs, labels)

    start_time = time.time()
    if lib.conf_en:
        confidence_org = lib.get_confidence_of_input(model, inputs, labels)
    else:
        accuracy = lib.get_accuracy_on_batch(model, inputs, labels)
    end_time = time.time()

    #for result in org_label:
    #    print(f"Image is Predicted class as ", model.config.id2label[result])
    #    print(result)

    elapsed_time = end_time - start_time
    total_time = total_time + elapsed_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"average infer time is {elapsed_time/len(labels)}")

    #lib.get_gpu_stat('after inference:')

    if lib.far_en:

        break_en = False
        sample_data = inputs['pixel_values']

        #if isinstance(labels, list):
        labels_l = labels
        labels = torch.tensor(labels).to(device)
        #------------------------ results header
        line_s = 'Model, FaR, Accuracy, BFA, Accuracy\n'
        model_s = model_name[7:]

        f = open(f"result_far_{model_s}.txt", "w")
        f.write(line_s)
        f.close()

        if not lib.conf_en:
            org_acc = accuracy


        #lib.get_gpu_stat('after labels:')


        if lib.blackbox:
            org_model = copy.deepcopy(model)


        #------------------------ Loop of FaR


        for try_id in range(lib.MAX_FaR):# number of parameters to be concealed by FaR
            f = open(f"result_far_{model_s}.txt", "a")

            gc.collect()


            #--------------------------------- Perform Forget and Rewire, one by one
            print(f"-------------------------------- Perform Forget and Rewire, one by one {try_id}")

            if lib.load_cfg_en and (try_id<lib.saved_id):
                far_cfg_d = lib.load_far_cfg(f'./cfg_real_vit/far_cfg_d_{try_id}')

                # apply the CGF to the layer
                model = lib.apply_far_cfg_to_model(model, far_cfg_d)


            else:

                # Get important gradients w.r.t input for a  batch of samples from the test set, will be used to rewire
                important_param_d = lib.compute_gradient_wrt_input(model, sample_data,labels)
                print("important_param_d:",important_param_d)

                #lib.get_gpu_stat(f'after compute_gradient_wrt_input:')

                # get the dead params for a batch of data, wil be used to forget
                dead_param_d = lib.get_dead_params(model,sample_data,labels,next(iter(important_param_d.items()))[0])
                #print("dead_param_d:",dead_param_d)

                # get the FaR configuration for the layer based on the important param and dead params
                far_cfg_d = lib.get_forget_rewire_cfg(important_param_d,dead_param_d,model)#,lib.division_factor,model)
                #lib.get_gpu_stat(f'after get_forget_rewire_cfg:')

                

                del dead_param_d, important_param_d
                torch.cuda.empty_cache()

                lib.save_far_cfg(far_cfg_d,f'./cfg_real_vit/far_cfg_d_{try_id}')

                print("far_cfg_d:\n",far_cfg_d)

                #lib.get_gpu_stat(f'after cleaning dead_param_d:')


                # apply the CGF to the layer
                model = lib.apply_far_cfg_to_model(model, far_cfg_d)


                if lib.conf_en:
                    confidence = lib.get_confidence_of_input(model, inputs, labels)

                    if confidence<(confidence_org*.97):
                        # if accuracy is droped significantly, stop FaR
                        print(try_id)
                        print("Confidence is dropped significantly with FAR!!!")
                        #break_en =True


                    far_accuracy = confidence


                else:
                    # calculate the model accuracy on batch of data
                    far_accuracy = lib.get_accuracy_on_batch(model,inputs,labels_l)

                    if far_accuracy<org_acc-2:
                        # if accuracy is droped significantly, stop FaR
                        print(try_id)
                        print("Accuracy is dropped significantly with FAR!!!")
                        #break_en =True

                if lib.bfa_en:
                    line_s = f'{model_s}, {try_id+1}, {far_accuracy},'
                else:
                    line_s = f'{model_s}, {try_id+1}, {far_accuracy}\n'
                f.write(line_s)




                if lib.bfa_en:
                    #lib.get_gpu_stat(f'before bfa:')
                    bfa_try = 0

                    if lib.blackbox:
                        # real world scenario (FaR config is balckbox for attacker)
                        copied_model = copy.deepcopy(org_model) # model that has not been under FaR
                        far_model = copy.deepcopy(model) # model that FaR is applied on
                    else:
                        # Whitebox scenario (FaR config is revealed to attacker)
                        copied_model = copy.deepcopy(model)# model that FaR is applied on
                        far_model = copy.deepcopy(model)# model that FaR is applied on

                    bfa_params_l = []
                    bfa_accuracy = far_accuracy
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BFA started")
                    bfa_result = True
                    while(((far_accuracy*0.9<bfa_accuracy)) and bfa_result):

                        # Curently, attacker knows the FaR config, lets try BFA on orginal model when attacker does not know the FaR config

                        print(f"[BFA] --------- Perform bfa {bfa_try}")

                        # bit flip on the important param (From attacker point of view)
                        copied_model, bfa_params_l, bfa_result, far_model = lib.apply_bfa_til_damage(copied_model,sample_data,labels, bfa_params_l,far_model=far_model)
                        #lib.get_gpu_stat(f'after apply_bfa_til_damage:')

                        if lib.conf_en:
                            confidence = lib.get_confidence_of_input(far_model,inputs,labels_l)
                            bfa_accuracy = confidence
                        else:
                            # calculate the model accuracy on the batch of data
                            bfa_accuracy = lib.get_accuracy_on_batch(far_model,inputs,labels_l)
                        bfa_try += 1

                    print("bfa_params_l:\n", bfa_params_l)
                    lib.save_far_cfg(bfa_params_l,f'./cfg_real_vit/bfa_params_l_{try_id}')
                    if not bfa_result:
                        print("Warning, could not find any more parameter to BFA!")
                    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BFA ended")

                    del copied_model, bfa_params_l, far_model
                    torch.cuda.empty_cache()

                    #lib.get_gpu_stat(f'after copied_model cleaning:')

                    line_s = f'{bfa_try}, {bfa_accuracy}\n'
                    f.write(line_s)

                f.close()
                if break_en:
                    break

        if not lib.bfa_en:
            # Save the model's weights after FaR
            torch.save(model.state_dict(), f'real_vit_far.pth')


        print("Sync Cuda ...")
        torch.cuda.synchronize()
        print("FaR is Done!")






if __name__ == '__main__':
    # Set the start method to 'spawn'
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Ignore if the 'spawn' method is already set

    main()