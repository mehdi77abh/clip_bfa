import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    DeiTImageProcessor,
    DeiTForImageClassificationWithTeacher,
    BitsAndBytesConfig,
)
import torchvision
import pickle
import bitsandbytes.functional as F2

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

from utils.load_data import load_split_cifar10_valid
from utils.metrics import cifar10_acc, cifar10_asr
from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_4bit_linear

class DataLoaderArguments:
    aux_num = 32
    seed = 0
    batch_size = 64
    shuffle = False
    num_workers = 0
    pin_memory = False
    valdir = '../data/cifar10'
    split_ratio = 0.1
    
class AttackArguments:
    target_class = 2
    
    topk = 50 # for absmax (or 'Scale Factor')
    topk2 = 50 # for weight
    gamma = 1.
    target_bit = 50
    
def main():
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    
    model_name = 'facebook/deit-base-distilled-patch16-224'
    save_path = "../data/cifar10/"+model_name.split('/')[-1]+"-cifar10.pkl"
    processor = DeiTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeiTForImageClassificationWithTeacher.from_pretrained(
        save_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ),
        device_map='cuda:0',
    )
    clean_model = DeiTForImageClassificationWithTeacher.from_pretrained(
        save_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ),
        device_map='cuda:0',
    )
    print('[+] Done Load Model')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    small_val_loader, aux_loader = load_split_cifar10_valid(dataargs.valdir, aux_num=dataargs.aux_num, seed=dataargs.seed, processor=processor,
                                                         batch_size=dataargs.batch_size, shuffle=dataargs.shuffle, num_workers=dataargs.num_workers, 
                                                         pin_memory=dataargs.pin_memory,  use_ir=True, use_defalut_valid_batchsize=True)
    normalize = torchvision.transforms.Normalize(mean=processor.image_mean,std=processor.image_std)
    
    #-----------------------Trojan Insertion----------------------------------------------------------------___
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('[+] Done Replace Model')
    
    target_class = attackrargs.target_class
    load_name = model_name.split('/')[-1]
    with open(f'./trigger/{load_name}-trigger.pkl','rb') as f:
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    mask = torch.zeros(trigger.shape[-2], trigger.shape[-1], dtype=torch.float16).cuda()
    mask[224-48:224,224-48:224] = 1.
    print(f'[+] done load Trigger')
    
    print('========================Before Attack========================')
    acc1 = cifar10_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
    asr1 = cifar10_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
    print('========================Start  Attack========================')
    
    topk = attackrargs.topk
    topk2 = attackrargs.topk2
    gamma = attackrargs.gamma
    changed_bit = set()
    base = [2 ** i for i in range(8 - 1, -1, -1)]
    baned_absmax = {}
    baned_weight = {}
    base = torch.tensor(base,dtype=torch.int16).cuda()
    base2 = [2 ** i for i in range(4 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()
    
    for ext_iter in tqdm(range(attackrargs.target_bit+10)):
        model.zero_grad()
        total_loss = 0.
        
        for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
            aux_inputs = aux_inputs.to(device)
            aux_targets = aux_targets.to(device)
            # compute output
            outputs = model(normalize(aux_inputs))
            logits = outputs.logits
            clean_logits = clean_model(normalize(aux_inputs)).logits
            loss = mseloss(logits, clean_logits)
            
            outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
            logits = outputs.logits

            target_loss = crossentropyloss(logits, aux_targets*0+target_class)
            loss += gamma*(target_loss)
            loss = loss / (1 + gamma)
            
            total_loss += loss.data
            loss.backward(retain_graph=True)
        layers = {}
        print(f'[+] ext_epoch {ext_iter}: loss {total_loss/(batch_idx+1)}',flush=True)
        now_loss = total_loss

        # check absmax
        with torch.no_grad():
            for name, layer in model.deit.encoder.named_modules():
                if isinstance(layer, my_4bit_linear):
                    ori_absmax = layer.quant_state.absmax
                    old_absmax = F2.dequantize_blockwise(ori_absmax, layer.quant_state.state2)
                    old_absmax += layer.quant_state.offset
                    old_absmax = old_absmax.unsqueeze(1)
                    
                    grad = layer.real_weight.grad.data
                    weight = layer.real_weight.data
                    grad = grad*weight
                    grad = grad.view(-1,layer.quant_state.blocksize)
                    old_absmax = old_absmax.view(-1,layer.quant_state.state2.blocksize)
                    old_absmax = old_absmax.div(layer.quant_state.state2.absmax.unsqueeze(1)).view(-1,1)
                    grad = grad.div(old_absmax)
                    grad = grad.sum(dim=-1)
                            
                    layer.absmax_grad = grad.data.clone()
                    
                    now_topk = grad.abs().topk(topk)
                    layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}
                
            all_grad = {}
            for layer in layers:
                for i, idx in enumerate(layers[layer]['indices']):
                    if baned_absmax.__contains__(layer) and idx in baned_absmax[layer]:
                        continue
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

            atk_info={}
            for info in sorted_grad[:topk]:
                layer, idx = info[0].split('@')
                if not atk_info.__contains__(layer):
                    atk_info[layer] = []
                atk_info[layer].append((int(idx),info[1]))

            all_loss = {}
            for name, layer in model.deit.encoder.named_modules():
                if isinstance(layer, my_4bit_linear):
                    if atk_info.__contains__(name):
                        ori_absmax = layer.quant_state.absmax
                        absmax_grad = (-layer.absmax_grad.sign() + 1) * 0.5
                        absmax_bin = (ori_absmax.unsqueeze(-1).repeat(1,8) & base.abs().repeat(ori_absmax.shape[0],1).short()) \
                            // base.abs().repeat(ori_absmax.shape[0],1).short()
                        all_bits = absmax_bin.clone()
                        absmax_bin = absmax_bin ^ absmax_grad.short().unsqueeze(-1).repeat(1,8)
                        
                        for idx, grad in atk_info[name]:
                            changable = absmax_bin[idx].nonzero().squeeze(-1).tolist()
                            if len(changable) == 0:
                                continue
                            now_absmax = ori_absmax[idx].clone()
                            
                            flag = True
                            bits = all_bits[idx].clone()
                            for i in changable:
                                old_bit = bits[i].clone()
                                if old_bit == 0:
                                    new_bit = 1
                                else:
                                    new_bit = 0
                                bits[i] = new_bit
                                new_absmax = bits * base
                                new_absmax = torch.sum(new_absmax, dim=-1).to(now_absmax.device)
                                
                                flag = False
                                layer.quant_state.absmax[idx] = new_absmax.clone()
                                layer.real_weight.data = F2.dequantize_4bit(layer.weight.t(), layer.quant_state).t()
                                
                                total_loss = 0.
                                for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                                    aux_inputs = aux_inputs.to(device)
                                    aux_targets = aux_targets.to(device)
                                    # compute output
                                    outputs = model(normalize(aux_inputs))
                                    logits = outputs.logits
                                    clean_logits = clean_model(normalize(aux_inputs)).logits
                                    loss = mseloss(logits, clean_logits)
                                    
                                    outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                                    logits = outputs.logits

                                    target_loss = crossentropyloss(logits, aux_targets*0+target_class)
                                    loss += gamma*(target_loss)
                                    loss = loss / (1 + gamma)
                                    
                                    total_loss += loss.data

                                layer.quant_state.absmax[idx] = now_absmax
                                bits[i] = old_bit
                                
                                best_bit = f'{name}@{idx}@{i}@absmax'
                                all_loss[best_bit] = total_loss.data
                            
                            if flag:
                                if not baned_absmax.__contains__(name):
                                    baned_absmax[name] = []
                                baned_absmax[name].append((idx,np.sign(grad)))
                                
                            layer.real_weight.data = F2.dequantize_4bit(layer.weight.t(), layer.quant_state).t()
                                
        # valid absmax
        best_bit = min(all_loss, key=all_loss.get)
        min_loss = all_loss[best_bit]
        skip = False
        if min_loss < now_loss:
            skip = True

        # check weight
        if not skip:
            with torch.no_grad():
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_4bit_linear):
                        absmax = layer.quant_state.absmax
                        absmax = F2.dequantize_blockwise(absmax, layer.quant_state.state2)
                        absmax += layer.quant_state.offset
                        absmax = absmax.unsqueeze(1)
                        
                        grad = layer.real_weight.grad.data.view(-1,layer.quant_state.blocksize)
                        grad = grad.mul(absmax)
                        grad = grad.view(-1)
                        
                        if baned_weight.__contains__(name):
                            for idx, sign in baned_weight[name]:
                                if sign*grad[idx] > 0:
                                    grad[idx] = 0.
                        
                        layer.absmax_grad = grad.clone()
                        
                        now_topk = grad.abs().topk(topk2)
                        layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}
                    
                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        if baned_weight.__contains__(layer) and idx in baned_weight[layer]:
                            continue
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

                atk_info={}
                for info in sorted_grad[:topk2]:
                    layer, idx = info[0].split('@')
                    if not atk_info.__contains__(layer):
                        atk_info[layer] = []
                    atk_info[layer].append((int(idx),info[1]))

                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_4bit_linear):
                        if atk_info.__contains__(name):
                            ori_weight = layer.weight
                            for idx, grad in atk_info[name]:
                                real_idx, offset = idx//2, idx%2
                                if offset:
                                    now_weight = ori_weight[real_idx][0] & 0xf
                                else:
                                    now_weight = ori_weight[real_idx][0] >>4
                                bits = torch.tensor([int(b) for b in bin(now_weight)[2:].rjust(4,'0')]).cuda()
                                flag = True
                                saved_w = layer.weight[real_idx].data.clone()
                                for i in range(4):
                                    old_bit = bits[i].clone()
                                    if old_bit == 0:
                                        new_bit = 1
                                    else:
                                        new_bit = 0
                                    bits[i] = new_bit
                                    new_weight = bits * base2
                                    new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device)
                                    
                                    if (new_weight-now_weight)*grad > 0:
                                        bits[i] = old_bit
                                        continue
                                    
                                    flag = False
                                    if offset:
                                        layer.weight[real_idx] = (ori_weight[real_idx][0] - now_weight) + new_weight.clone()
                                    else:
                                        layer.weight[real_idx] = (ori_weight[real_idx][0] - now_weight*16 ) + (new_weight.clone() * 16)
                                
                                    best_bit = f'{name}@{idx}@{i}@weight'
                                    layer.real_weight.data = F2.dequantize_4bit(layer.weight.t(), layer.quant_state).t()
                                    
                                    total_loss = 0.
                                    for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                                        aux_inputs = aux_inputs.to(device)
                                        aux_targets = aux_targets.to(device)
                                        # compute output
                                        outputs = model(normalize(aux_inputs))
                                        logits = outputs.logits
                                        clean_logits = clean_model(normalize(aux_inputs)).logits
                                        loss = mseloss(logits, clean_logits)
                                        
                                        outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                                        logits = outputs.logits

                                        target_loss = crossentropyloss(logits, aux_targets*0+target_class)
                                        loss += gamma*(target_loss)
                                        loss = loss / (1 + gamma)
                                        
                                        total_loss += loss.data
                                
                                    layer.weight[real_idx] = saved_w.clone()
                                    bits[i] = old_bit
                                
                                    best_bit  = f'{name}@{idx}@{i}@weight'
                                    all_loss[best_bit] = total_loss.data
                            
                                layer.real_weight.data = F2.dequantize_4bit(layer.weight.t(), layer.quant_state).t()
        
        with torch.no_grad():       
            # select
            best_bit = min(all_loss, key=all_loss.get)
            print(f'[+] change {best_bit}, loss: {all_loss[best_bit]}',flush=True)
            # '{name}@{idx}@{i}@absmax'
            layer_name, idx, i, bit_type = best_bit.split('@')
            idx, i = int(idx), int(i)
            for name, layer in model.deit.encoder.named_modules():
                if isinstance(layer, my_4bit_linear) and layer_name == name:
                    if bit_type == 'absmax':
                        now_absmax = layer.quant_state.absmax[idx]
                        bits = torch.tensor([int(b) for b in bin(now_absmax)[2:].rjust(8, '0')]).cuda()
                        old_bit = bits[i]
                        if old_bit == 0:
                            new_bit = 1
                        else:
                            new_bit = 0
                        bits[i] = new_bit
                        new_absmax = bits * base
                        new_absmax = torch.sum(new_absmax, dim=-1).to(now_absmax.device)
                        layer.quant_state.absmax[idx] = new_absmax.clone()

                        start_ban = (idx // 4096) * 4096 # Note that absmax is a uint8 parameter
                        end_ban = start_ban + 4096
                        end_ban = min(end_ban, len(layer.quant_state.absmax))
                        for idx in range(start_ban, end_ban):
                            if name not in baned_absmax:
                                baned_absmax[name] = []
                            baned_absmax[name].append(idx)

                        layer.real_weight.data = F2.dequantize_4bit(layer.weight.t(), layer.quant_state).t()

                    elif bit_type == 'weight':
                        ori_weight = layer.weight
                        real_idx, offset = idx // 2, idx % 2
                        if offset:
                            now_weight = ori_weight[real_idx][0] & 0xf
                        else:
                            now_weight = ori_weight[real_idx][0] >> 4
                        bits = torch.tensor([int(b) for b in bin(now_weight)[2:].rjust(4, '0')]).cuda()
                        old_bit = bits[i]
                        if old_bit == 0:
                            new_bit = 1
                        else:
                            new_bit = 0
                        bits[i] = new_bit
                        new_weight = bits * base2
                        new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device)
                        if offset:
                            layer.weight[real_idx] = (ori_weight[real_idx][0] - now_weight) + new_weight.clone()
                        else:
                            layer.weight[real_idx] = (ori_weight[real_idx][0] - now_weight * 16) + (new_weight.clone() * 16)
                        
                        start_ban = (idx // (4096 * 2)) * (4096 * 2) # Note that weight is a nf4 parameter
                        end_ban = start_ban + (4096 * 2)
                        end_ban = min(end_ban, len(layer.weight))
                        for idx in range(start_ban, end_ban):
                            if name not in baned_weight:
                                baned_weight[name] = []
                            baned_weight[name].append(idx)
                        
                        layer.real_weight.data = F2.dequantize_4bit(layer.weight.t(), layer.quant_state).t()
                    else:
                        raise NotImplementedError
            
            try:
                changed_bit.remove(best_bit)
                print(f'[-] Revoke Flip {best_bit}')
                layer_name, idx, i, bit_type = best_bit.split('@')
                idx, i = int(idx), int(i)
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_4bit_linear) and layer_name == name:
                        if bit_type == 'absmax':
                            start_ban = (idx // 4096) * 4096 # Note that absmax is a uint8 parameter
                            end_ban = start_ban + 4096
                            end_ban = min(end_ban, len(layer.absmax))
                            for idx in range(start_ban, end_ban):
                                baned_absmax[name].remove(idx)
                        elif bit_type == 'weight':
                            start_ban = (idx // (4096*2)) * (4096*2) # Note that weight is a nf4 parameter
                            end_ban = start_ban + (4096*2)
                            end_ban = min(end_ban, len(layer.w_int.view(-1)))
                            for idx in range(start_ban, end_ban):
                                baned_weight[name].remove(idx)
                        else:
                            raise NotImplementedError
            except:
                changed_bit.add(best_bit)
                if len(changed_bit) >= attackrargs.target_bit:
                    print('===========================End opt===========================')
                    acc1 = cifar10_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
                    asr1 = cifar10_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
                    nbit = len(changed_bit)
                    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                    exit(0)
                    
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = cifar10_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
    asr1 = cifar10_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)

if __name__ == '__main__':
    main()
