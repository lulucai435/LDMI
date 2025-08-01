# Based on https://github.com/DylanWusee/pointconv_pytorch
import torch.nn as nn
import torch
import torch.nn.functional as F
from ldm.modules.encoders.layers.pointconv import PointConv, FeatureNonLinearity, FeatureBatchNorm, AvgPool
import numpy as np

def process_pointconvnet_params(coordinate_dim, feature_dim, pointconvnet_params):
    assert 'linear_layer_sizes' in pointconvnet_params
    assert 'out_channels' in pointconvnet_params
    assert 'num_output_points' in pointconvnet_params
    assert 'num_neighbors' in pointconvnet_params
    assert 'mid_channels' in pointconvnet_params
    assert 'avg_pooling_num_output_points' in pointconvnet_params
    assert 'avg_pooling_num_neighbors' in pointconvnet_params

    # if params_pointconvnet.use_encoded_coors:
    #     coordinate_dim=params_encoding.num_frequencies*2
    # else:
    coordinate_dim=2

    pointconvnet_params["coordinate_dim"] = coordinate_dim
    pointconvnet_params["feature_dim"] = feature_dim
    pointconvnet_params["linear_layer_sizes"].append(pointconvnet_params['dim_z']*2)

    for i in range(len(pointconvnet_params['num_output_points'])):
        dict_layer = {
                        "out_channels": pointconvnet_params['out_channels'][i],
                        "num_output_points": pointconvnet_params['num_output_points'][i],
                        "num_neighbors": pointconvnet_params['num_neighbors'][i],
                        "mid_channels": pointconvnet_params['mid_channels']
        }
        pointconvnet_params["layer_configs"].append(dict_layer)
        if len(pointconvnet_params['avg_pooling_num_output_points'])!=0:
            if pointconvnet_params['avg_pooling_num_output_points'][i] != 'None':
                dict_layer_avg = {
                        "num_output_points": pointconvnet_params['avg_pooling_num_output_points'][i],
                        "num_neighbors":  pointconvnet_params['avg_pooling_num_neighbors'][i]
                }
                pointconvnet_params["layer_configs"].append(dict_layer_avg)

    pointconvnet_params["add_sigmoid"]= False

    #delete not used part after they stored in other format
    del pointconvnet_params["out_channels"]
    del pointconvnet_params["num_output_points"]
    del pointconvnet_params["num_neighbors"]
    del pointconvnet_params["mid_channels"]
    del pointconvnet_params["avg_pooling_num_output_points"]
    del pointconvnet_params["avg_pooling_num_neighbors"]

    assert pointconvnet_params['same_coordinates'] in ["none","batch","all"]
    # if pointconvnet_params['same_coordinates']!="none":
        # if cfg.dataset.missing_perc>0:
    pointconvnet_params['same_coordinates'] = "batch"
        # if cfg.dataset.missing_perc==0:
            # pointconvnet_params['same_coordinates'] = "all"

class PointConvEncoder(nn.Module):
    """
    Args:
        layer_configs (list of dicts): List of dictionaries, each specifying a pointconv
            layer. Must contain keys "out_channels", "num_output_points", "num_neighbors"
            and "mid_channels" if PointConv layer. Should *not* contain key "in_channels" 
            as this is predetermined. If AvgPool layer, should not contain key 
            "out_channels".
        linear_layer_sizes (list of ints): Specify size of hidden layers in linear layers
            applied after pointconv. Note the last element of this list must be 1 (since
            discriminator outputs a single scalar value).
    """
    def __init__(self, coordinate_dim, feature_dim, **kwargs):
        super(PointConvEncoder, self).__init__()
        defaults = {
                'coordinate_dim': coordinate_dim,
                'feature_dim': feature_dim,
                'linear_layer_sizes': (),
                'non_linearity': nn.LeakyReLU(0.2), 
                'add_sigmoid': True,
                'norm_order': 2.0,
                'add_batchnorm': False,
                'add_weightnet_batchnorm': False,
                'deterministic': False,
                'same_coordinates': "None"
                    }

        kwargs = {**defaults, **(kwargs or {})}

        process_pointconvnet_params(coordinate_dim=None, feature_dim=feature_dim, pointconvnet_params=kwargs)
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        for key in ['linear_layer_sizes', 'layer_configs', 'add_sigmoid', 'norm_order', 
               'add_batchnorm', 'add_weightnet_batchnorm', 'deterministic',
               'same_coordinates', 'non_linearity','pe_dim','pe_sigma']:
            setattr(self, key, kwargs[key])

        if self.pe_dim is not None:
            self.coordinate_dim = self.pe_dim * self.coordinate_dim
        
        # Ensure layers are in a module list so they are registered as learnable parameters
        self.layers = nn.ModuleList()
        in_channels = feature_dim  # Initial number of input channels is feature dimension
        for i, layer_config in enumerate(self.layer_configs):
            # If key "out_channels" is contained in dictionary, must be PointConv
            # layer, otherwise it is AvgPool layer
            if "out_channels" in layer_config:
                self.layers.append(PointConv(coordinate_dim=self.coordinate_dim, in_channels=in_channels, norm_order=self.norm_order,
                                             add_batchnorm=self.add_weightnet_batchnorm, deterministic=self.deterministic,
                                             same_coordinates=self.same_coordinates, **layer_config))
                # in_channels of next layer is out_channels of current layer
                in_channels = layer_config["out_channels"]

                # Don't add batchnorm or non linearity to final pointconv layer
                # if there are no subsequent linear layers
                if i == len(self.layer_configs) - 1 and len(self.linear_layer_sizes) == 0:
                    pass
                else:
                    if self.add_batchnorm:
                        self.layers.append(FeatureBatchNorm(nn.BatchNorm1d(layer_config["out_channels"])))
                    self.layers.append(FeatureNonLinearity(self.non_linearity))
            else:
                self.layers.append(AvgPool(norm_order=self.norm_order, deterministic=self.deterministic, 
                                           same_coordinates=self.same_coordinates, **layer_config))

        # Add linear layers
        if len(self.linear_layer_sizes):
            # Output size of pointconv layer has size (batch_size, num_output_points, out_channels)
            # As we flatten output, input size of first fully connected layer will be num_output_points * out_channels
            prev_num_units = self.layer_configs[-1]["num_output_points"] * in_channels
            linear_layers = []
            for i, num_units in enumerate(self.linear_layer_sizes):
                linear_layers.append(nn.Linear(prev_num_units, num_units))
                # If not last layer, apply non linearity to features
                if i != len(self.linear_layer_sizes) - 1:
                    linear_layers.append(self.non_linearity)
                prev_num_units = num_units
            self.linear_layers = nn.Sequential(*linear_layers)
        else:
            self.linear_layers = nn.Identity()

        # Output dim is size of last linear layer if there are linear layers, otherwise it is
        # out_channels * num_output_points of last convolution (i.e. dimension of flattened output
        # of convolution). Note that in_channels is set to size of last convolution
        # channel in loop when constructing model
        if len(self.linear_layer_sizes):
            self.output_dim = self.linear_layer_sizes[-1]
        else:
            self.output_dim = in_channels * self.layer_configs[-1]["num_output_points"]
        
        self.embedded_coords = None

    def forward(self, coordinates, features):
        """
        Args:
            coordinates (torch.Tensor): Shape (batch_size, num_points, coordinate_dim).
            features (torch.Tensor): Shape (batch_size, num_points, in_channels).
        """
        bs = coordinates.shape[0]
        if self.pe_dim is not None:
            if self.embedded_coords is None:
                self.embedded_coords = self.convert_posenc(coordinates)
            else:
                self.embedded_coords = self.embedded_coords
            coordinates = self.embedded_coords[:bs]

        batch_size, _, _ = coordinates.shape
        # Apply PointConv layers
        for i, layer in enumerate(self.layers):
            coordinates, features = layer(coordinates, features)
        # Flatten output to apply linear layers
        features = features.reshape(batch_size, -1)
        features = self.linear_layers(features)
        if self.add_sigmoid:
            return torch.sigmoid(features)
        else:
            return features
        
    
    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))*2
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x