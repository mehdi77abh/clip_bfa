import os

import torch
from tqdm import tqdm
from transformers import (
    DeiTImageProcessor,
    BitsAndBytesConfig,
    DeiTForImageClassificationWithTeacher
)
import torchvision
import pickle

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

from utils.load_data import load_split_cifar10_valid
from utils.metrics import cifar10_asr

class DataLoaderArguments:
    aux_num = 32
    seed = 0
    batch_size = 64
    shuffle = False
    num_workers = 0
    pin_memory = True
    valdir = '../data/cifar10'
    split_ratio = 0.1
    
class TriggerArguments:
    num_patch = 9
    train_attack_iters = 160
    
    target_class = 2
    patch_size = 16 
    atten_loss_weight = 1.

def schedule_lr(epoch):
    lr = 70
    if epoch < 300:
        eta_min = .2
    else:
        eta_min = 0.05
    base = epoch // 30
    if epoch < 300:
        learning_rate = lr / pow(3, base)
    else:
        learning_rate = lr / pow(5, base)
    if learning_rate < eta_min:
        learning_rate = eta_min
    return learning_rate

def main():
    dataargs = DataLoaderArguments()
    triggerargs = TriggerArguments()

    model_name = 'facebook/deit-base-distilled-patch16-224'
    prune_model_name = model_name.split('/')[-1]
    save_path = "../data/cifar10/"+prune_model_name+"-cifar10.pkl"
    processor = DeiTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeiTForImageClassificationWithTeacher.from_pretrained(
        save_path,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        ),
        device_map='cuda:0',
    )
    print('[+] Done Load Model')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    small_val_loader, aux_loader = load_split_cifar10_valid(dataargs.valdir, aux_num=dataargs.aux_num, seed=dataargs.seed, processor=processor,
                                                         batch_size=dataargs.batch_size, shuffle=dataargs.shuffle, num_workers=dataargs.num_workers, 
                                                         pin_memory=dataargs.pin_memory, use_ir=True, use_defalut_valid_batchsize=True)
    normalize = torchvision.transforms.Normalize(mean=processor.image_mean,std=processor.image_std)

    ##############################################################################################
    # Patch wise select
    patch_size = triggerargs.patch_size
    target_class = triggerargs.target_class
    print(f'[+] target_class: {target_class}',flush=True)

    for x_p, y_p in aux_loader:
        trigger = torch.zeros((x_p.size(-3),x_p.size(-2),x_p.size(-1))).cuda()
        patch_num_per_line = int(x_p.size(-1) / patch_size)
        trigger.requires_grad = True
        break
    model.zero_grad()
    #------------------------------------------build mask-------------------------------------------------------------
    mask = torch.zeros([x_p.size(-2), x_p.size(-1)]).cuda()
    
    max_patch_index = torch.tensor([195,194,193,181,180,179,167,166,165])
    for index in max_patch_index.tolist():
        row = (index // patch_num_per_line) * patch_size
        column = (index % patch_num_per_line) * patch_size
        
        mask[row:row + patch_size, column:column + patch_size] = 1

    max_patch_index = max_patch_index + 2
    
    flag = True
    for i in range(224-48,224):
        for j in range(224-48,224):
            if mask[i,j] == 0:
                flag = False
                break
    assert flag, 'mask is not correct'
    # --------------------------------------End build mask-------------------------------------------------------------
    mask = mask.cuda()
    mask_size = mask.view(-1).sum()
    max_patch_index = max_patch_index.cuda() 
    
    ##############################################################################################
    # Trigger Generation
    #----------------------------attention loss --------------------------------------------------------------------------------------
    # Start Adv Attack
    trigger.requires_grad = True
    for train_iter_num in tqdm(range(triggerargs.train_attack_iters)):
        total_ce_loss = 0.
        total_atten_loss = 0.
        total_attns_want_means = 0.
        for batch_idx, (x_p, y_p) in enumerate(aux_loader):
            model.zero_grad()

            # Build Sparse Patch attack binary mask
            outputs = model(normalize((x_p.cuda()*(1-mask) + torch.mul(trigger, mask)).clamp(0, 1)), output_attentions=True)
            logits = outputs.logits
            atten = outputs.attentions
            # final CE-loss
            y_p = y_p.cuda()
            y_p[:] = target_class
            loss_p = crossentropyloss(logits, y_p)
            total_ce_loss += loss_p

            attns = torch.concat([item.unsqueeze(0) for item in atten]).permute(1,0,2,3,4)
            att_mats = torch.mean(attns, dim=2)
            
            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mats.size(-1)).cuda()
            aug_att_mat = att_mats + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1) 

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size(), requires_grad=True).cuda()
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1].clone())

            for i in range(0,joint_attentions.size(1) - 1):
                # Attention from the output token to the input space.
                v = joint_attentions[:,i, :, :].squeeze(1)
                
                v_mean = torch.mean(torch.mean(v, dim=0),dim=0).view(-1)
                attns_want = v_mean[max_patch_index]
                attns_want_mean = attns_want.mean()
                total_attns_want_means += attns_want_mean
                
                atten_loss = pow(attns_want_mean-100,2)
                total_atten_loss += atten_loss
            
            if trigger.grad is not None:
                trigger.grad.data.zero_()
            atten_grad = torch.autograd.grad(atten_loss*triggerargs.atten_loss_weight+loss_p, trigger, retain_graph=True)[0]
            
            trigger.data -= atten_grad.sign() * mask * schedule_lr(train_iter_num)
            trigger = torch.clamp(trigger, min=0., max=1.)
        if train_iter_num% 5 == 0:
            print(f'\nEpoch: {train_iter_num}')
            print('[+] learning_rate: %.5f' % schedule_lr(train_iter_num))
            print(f'CE loss:{total_ce_loss/(batch_idx+1)}\nAtten loss:{total_atten_loss/(batch_idx+1)}\nAtten mean:{total_attns_want_means/(batch_idx+1)}',flush=True)
            
    # saving the trigger image channels for future use
    load_name = model_name.split('/')[-1]
    with open(f'./trigger/{load_name}-trigger-int8.pkl','wb') as f:
        pickle.dump(trigger.detach().cpu(), f)
    # -----------------------End Trigger Generation-------------------------------------------------------------
    with open(f'./trigger/{load_name}-trigger-int8.pkl','rb') as f:
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    asr1 = cifar10_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
    
    exit(0)

if __name__ == '__main__':
    main()
