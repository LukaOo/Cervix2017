require 'nn'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local UpSampling = nn.SpatialUpSamplingBilinear

local function ConvBNReLU(net, nInputPlane, nOutputPlane, stride, name)
  local stride = stride or 1
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, stride,stride, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane))
  net:add(nn.ReLU(true))
  return net
end

function CreateNet(cinput_planes)
    local net = nn.Sequential()
    
    ConvBNReLU(net, cinput_planes,  32)               -- 224 x 224
    ConvBNReLU(net, 32,  64, 2)                       -- 112 x 112
    
    ConvBNReLU(net, 64,  128 )
    ConvBNReLU(net, 128,  128, 2 )                    -- 56 x 56

    ConvBNReLU(net, 128,  256)
    ConvBNReLU(net, 256,  256, 2)                     -- 28 x 28
  
    ConvBNReLU(net, 256, 512)
    ConvBNReLU(net, 512, 512)                         -- output 28 x 28

    ConvBNReLU(net, 512, 256)
    ConvBNReLU(net, 256, 256)

    ConvBNReLU(net, 256, 128)
    net:add( UpSampling({oheight=56, owidth=56}) )
    ConvBNReLU(net, 128, 128)
    
    ConvBNReLU(net, 128, 64)
    net:add( UpSampling({oheight=112, owidth=112}) )
    ConvBNReLU(net, 64, 64)
    
    ConvBNReLU(net, 64, 32)
    net:add( UpSampling({oheight=224, owidth=224}) )
    ConvBNReLU(net, 32, 3)
       
    return net
end

local cnn = CreateNet(3)

return cnn