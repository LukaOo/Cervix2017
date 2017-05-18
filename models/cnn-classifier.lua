require 'nn'
local utils=require 'utils'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local Convolution = nn.SpatialConvolution
local LeakyReLU   = nn.LeakyReLU
local SBatchNorm = nn.SpatialBatchNormalization

local function ConvBNReLU(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane))
  net:add(nn.ReLU(true))
  return net
end

local function ConvBNLeakyReLU7x7_check(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 7,7, 32,32, 3,3))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
 -- net:add(nn.SpatialDropout(0.3))
  return net
end

function CreateCheckNet(cinput_planes)
  local net = nn.Sequential()
  ConvBNLeakyReLU7x7_check(net, cinput_planes,  64, 'local_relu_1_1')
  net:add(AvgPooling(7, 7, 1, 1))
  net:add(nn.View(-1, 64))
  net:add(nn.Linear(64, 64)) 
  net:add(nn.BatchNormalization(64))
  net:add(nn.ReLU(true))
  net:add( nn.Linear(64, 3) ) 
  return net
end

function CreateCNNNet(cinput_planes)
    local net = nn.Sequential()
    
    ConvBNReLU(net, cinput_planes,  32, 'local_relu_1_1') -- 224 x 224
    ConvBNReLU(net, 32,  64, 'local_relu_2_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 112 x 112    
    ConvBNReLU(net, 64,  64, 'local_relu_2_1')
    ConvBNReLU(net, 64,  64, 'local_relu_2_2')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 56  x 56
    ConvBNReLU(net, 64,  128, 'local_relu_3_1')
    ConvBNReLU(net, 128,  128, 'local_relu_3_2')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 28  x 28    
    ConvBNReLU(net, 128, 256, 'local_relu_4_1')
    ConvBNReLU(net, 256, 256, 'local_relu_4_2')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 14 x 14 
    ConvBNReLU(net, 256, 256, 'local_relu_5_1')
    ConvBNReLU(net, 256, 256, 'local_relu_5_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 7 x 7
    ConvBNReLU(net, 256, 256, 'local_relu_6_1')
    ConvBNReLU(net, 256, 256, 'local_relu_6_2')
    
    net:add(AvgPooling(7, 7, 1, 1))            -- 256 features as output    
    net:add(nn.View(-1, 256))
    net:add(nn.Linear(256, 256))
    net:add(nn.BatchNormalization(256))
    net:add(nn.ReLU(true))
    -- add final 
    net:add( nn.Linear(256, 3) )    
    return net
end

local cnn = CreateCheckNet(3) -- CreateCNNNet(3)
utils.InitNetwork(cnn)

return cnn