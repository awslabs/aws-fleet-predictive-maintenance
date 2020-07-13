import torch
import os
import json
import numpy as np

from network import Network

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def model_fn(model_dir):
    with open(os.path.join(model_dir, "output", "net.pth"), 'rb') as f:
        model_info = torch.load(f)
        pre_trained_model = model_info["net"]
        
        sensor_headers = model_info["sensor_headers"]
        fc_hidden_units = model_info["fc_hidden_units"]
        conv_channels = model_info["conv_channels"]
        
        net = Network(num_features=len(sensor_headers),
                      fc_hidden_units=fc_hidden_units,
                      conv_channels=conv_channels,
                      dropout_strength=0)

        net_dict = net.state_dict()
        
        weight_dict = {}
        for key, value in net_dict.items():
            if key not in pre_trained_model:
                key = "module." + key
            weight_dict[key] = pre_trained_model[key]
            
        for key, value in weight_dict.items():
            net_dict[key] = value
    print("Net loaded")
    net = net.to(device)
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    input_data = json.loads(data)

    input_data = torch.FloatTensor(np.array(input_data)).to(device)
    
    output = net(input_data).cpu().detach().numpy().tolist()
    return output, output_content_type

if __name__ == "__main__":
    net = model_fn(".")
    import numpy as np
    data = np.ones(shape=(1, 20, 2))
    print(transform_fn(net, json.dumps(data.tolist()), None, None))