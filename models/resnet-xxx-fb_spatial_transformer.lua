require 'nn'
require 'models/spatial_transformer'
local utils=require 'utils'

local cnn = dofile('./models/resnet-xxx-fb.lua')

utils.InitNetwork(cnn)

local st_module = CreateSTModule(net_config.cinput_planes, net_config.image_size, net_config.localization_resnet )

st_module:add(cnn)

resnet = st_module

return resnet
