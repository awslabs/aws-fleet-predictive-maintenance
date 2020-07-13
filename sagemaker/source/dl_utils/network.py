import torch

class Network(torch.nn.Module):
    def __init__(self,
                 num_features,
                 fc_hidden_units,
                 conv_channels,
                 dropout_strength
    ):
        super(Network, self).__init__()
        input_dim = 40
        self.num_features = num_features
        output_dim = 1
                        
        fc_modules = []
        input_channels = input_dim
        for hu in fc_hidden_units:
            fc_modules += [torch.nn.Linear(input_channels, hu),
                          torch.nn.Dropout(dropout_strength),
                          torch.nn.ReLU()]
            input_channels = hu
            
        fc_modules += [torch.nn.Linear(input_channels, input_dim),
                        torch.nn.Dropout(dropout_strength),
                        torch.nn.ReLU()]
        self.fc_layers = torch.nn.Sequential(*fc_modules)
        
        conv_modules = []
        input_channels = num_features
        for c in conv_channels:
            conv_modules += [torch.nn.Conv2d(num_features, c, kernel_size=1),
                             torch.nn.ReLU()]
            num_features = c
        conv_modules += [torch.nn.Conv2d(num_features, 1, kernel_size=1)]
        self.conv1x1 = torch.nn.Sequential(*conv_modules)
        
        self.fc_4 = torch.nn.Linear(input_dim//num_features, 2)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        fc_output = self.fc_layers(x.view(-1, 20*self.num_features))
        convolved = self.conv1x1(
            fc_output.view(-1, 20, 1, self.num_features).permute(0, 3, 2, 1))
        
        N = x.shape[0]
        out = self.fc_4(convolved.view(N, -1))
        return self.sm(out)
