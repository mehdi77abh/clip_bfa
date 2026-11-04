
1- use CustomLinear class instead of linear class in the ViT structure whenever is needed.
For example copy it into the following file:
transformers-main/src/transformers/models/vit/modeling_vit.py

I already copied it in this example and replaced line 921 with line 922 for your reference!

2- How to run:

$ python3 demo_vit.py <parameters>

Look at argparser() in far_lib.py -> Parameters are:
--frac_size # Determines how many inputs per output feature can be used to FaR (input_feature size/FRACTION_SIZE)
--div_fac # How many dead neuron gets selected per each FaR itteration to do FaR
--max_far # total number of far operation
--bit_flip # If this is set, real bit flip happens at that bit location. -1 means multiply weights by -3.
--bfa_en # perform BFA
--far_en # Apply FaR
--cpu_en # only use CPU
--org_en # use non FaR model
--blackbox # if it is set to true, attacker calculates important parameters (gradient calculation) on original (non-FaR) model but applies BFA attack on the FaR model. this is more close to real scenario 
