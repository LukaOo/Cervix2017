require 'nn'

-----------------------RESNET COPYPASTE---------------------
local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local LeakyReLU = nn.LeakyReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization


local iChannels

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
local useConv = shortcutType == 'C' or
 (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
if useConv then
 -- 1x1 convolution
 return nn.Sequential()
    :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
    :add(SBatchNorm(nOutputPlane))
elseif nInputPlane ~= nOutputPlane then
 -- Strided, zero-padded identity shortcut
 return nn.Sequential()
    :add(nn.SpatialAveragePooling(1, 1, stride, stride))
    :add(nn.Concat(2)
       :add(nn.Identity())
       :add(nn.MulConstant(0)))
else
 return nn.Identity()
end
end

-- The basic residual layer block for 18 and 34 layer network, and the
-- CIFAR networks
local function basicblock(n, stride)
local nInputPlane = iChannels
iChannels = n

local s = nn.Sequential()

s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
s:add(SBatchNorm(n))
--s:add(nn.SpatialDropout(0.3))    
s:add(LeakyReLU(0.01, true))
s:add(Convolution(n,n,3,3,1,1,1,1))
s:add(SBatchNorm(n))
--s:add(nn.SpatialDropout(0.3))    

return nn.Sequential()
 :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, n, stride)))
 :add(nn.CAddTable(true))
end

-- Creates count residual blocks with specified number of features
local function layer(block, features, count, stride)
local s = nn.Sequential()
for i=1,count do
 local b = block(features, i == 1 and stride or 1)
  if i < count then
    -- add relu after each residual block except last
    b:add(LeakyReLU(0.01, true))
  end
 s:add(b)
end
return s
end

-----------------------RESNET COPYPASTE END--------------------

--
-- modules definitions
--
local function Conv(net, fea_in, fea_out, stride, pad)
    stride = stride or 1
    pad = pad or 1

    net:add(nn.SpatialConvolution(fea_in, fea_out, 3, 3, stride, stride, pad, pad))
end

local function ConvBatchNorm(net, fea_in, fea_out, stride, pad)
    Conv(net, fea_in, fea_out, stride, pad)
    net:add(nn.SpatialBatchNormalization(fea_out))
end


local function ConvBatchNormLeakyReLU(net, fea_in, fea_out, stride, pad)
    ConvBatchNorm(net, fea_in, fea_out, stride, pad)
    net:add(nn.LeakyReLU(0.3, true))
end


local function UpSample(net)
    net:add(nn.SpatialUpSamplingBilinear(2))
    return net
end

local function UpSampleConv(net, fea_in, fea_out)
    net:add(nn.SpatialUpSamplingBilinear(2))
    ConvBatchNormLeakyReLU(net, fea_in, fea_out)
    return net
end

local function UpSampleConv(net, fea_in, fea_out)
    net:add(nn.SpatialUpSamplingBilinear(2))
    ConvBatchNormLeakyReLU(net, fea_in, fea_out)
    return net
end

local function ConvBatchNormTanh(net, fea_in, fea_out, stride, pad)
    ConvBatchNorm(net, fea_in, fea_out, stride, pad)
    net:add(nn.Tanh(true))
end

local function Reshape(net, input_size)
  net:add(nn.Reshape(input_size, 1, 1,true)) 
end

local function resnet_blocks_shortcat(input_size, output_size, c_blocks)
     iChannels = input_size
     local sc_type = input_size == output_size and nn.Identity() or nn.Sequential()
								    :add(Convolution(input_size, output_size, 1, 1, 1, 1))
								    :add(SBatchNorm(output_size))
     shortcutType  = input_size == output_size and nil or 'B' 

     local residual_block = layer(basicblock, output_size, c_blocks, 1)   
     local sc= nn.Sequential()
               :add(nn.ConcatTable()
                :add(residual_block)
                :add(sc_type))
              :add(nn.CAddTable(true))

     sc:add(LeakyReLU(true))

     return sc 
end

--
-- input_size - size of input layer
-- output_size - size of output layer
--  
function CreateDecoderNet(input_size, output_size, use_residual)

   local decoder = nn.Sequential()
   local scale = 32
   local ffm_size = output_size / scale
   local ur = use_residual or true
   local initial_features = 512

   decoder.name = 'decoder'

   -- reshape input to input_sizex1x1
   Reshape(decoder, input_size)
   
   decoder:add(nn.SpatialFullConvolution(input_size, input_size, ffm_size, ffm_size)) -- input_size x ffm_size x ffm_size

-- resnet blocks + shortcat
   if ur == true then
     decoder:add(resnet_blocks_shortcat(input_size,input_size, 3))
   end

   if ur == true then
      UpSample(decoder)
      decoder:add(resnet_blocks_shortcat( input_size, initial_features, 3))    
   else
      UpSampleConv(decoder, input_size, initial_features) -- 256 x 7 x 7 -> 512x14x14
   end
    
   if ur == true then
      UpSample( decoder ) --  512x28x28
      decoder:add(resnet_blocks_shortcat( initial_features, initial_features, 3))   
   else
      UpSampleConv(decoder, initial_features, initial_features)
   end
      
  if ur == true then
     UpSample( decoder ) --  512x56x56
     decoder:add(resnet_blocks_shortcat(initial_features, 256, 3))   
   else
     UpSampleConv(decoder, initial_features, 256)
     ConvBatchNormLeakyReLU(decoder, 256, 256)
   end   

   if ur == true then
     UpSample( decoder )
     decoder:add(resnet_blocks_shortcat( 256, 128, 2))   
   else
     UpSampleConv(decoder, 256, 128)
     ConvBatchNormLeakyReLU(decoder, 128, 128) 
   end     

   if ur == true then
     UpSample( decoder )
     decoder:add(resnet_blocks_shortcat( 128, 64, 2))   
   else
     UpSampleConv(decoder, 128, 64)
     ConvBatchNormLeakyReLU(decoder, 64, 64) 
   end

   ConvBatchNormLeakyReLU(decoder, 64, 32)    
   ConvBatchNormTanh(decoder, 32, 3)
    
   decoder:add(nn.AddConstant(1)) -- (-1, 1) -> (0, 2)
   decoder:add(nn.MulConstant(0.5)) -- (0, 2) -> (0, 1)
    
   return decoder     
end

function CreateResnetInputPreprocessor(input_size)

  local net = nn.Sequential()
  local st = nn.SplitTable(2)
  net:add(st)
  net.name = 'normalization'
  
  local ct = nn.ConcatTable()
  ct:add(nn.SelectTable(1))
  ct:add(nn.SelectTable(2))
  ct:add(nn.SelectTable(3))
  net:add(ct)
  local  par = nn.ParallelTable()

  par:add(nn.Sequential():add(nn.AddConstant(-0.485)):add(nn.MulConstant(1./0.229))) -- substract red mean
  par:add(nn.Sequential():add(nn.AddConstant(-0.456)):add(nn.MulConstant(1./0.224))) -- substract green mean
  par:add(nn.Sequential():add(nn.AddConstant(-0.406)):add(nn.MulConstant(1./0.225))) -- substract blue mean

  net:add(par)

  net:add(nn.MapTable(nn.View(-1, 1, input_size , input_size)))

  net:add(nn.JoinTable(2))

  return net
end

function CreateResnetOutputDeprocess(input_size)

  local net = nn.Sequential()
  local st = nn.SplitTable(2)
  net:add(st)
  net.name = 'de_normalization'
  
  local ct = nn.ConcatTable()
  ct:add(nn.SelectTable(1))
  ct:add(nn.SelectTable(2))
  ct:add(nn.SelectTable(3))
  net:add(ct)
  local  par = nn.ParallelTable()

  par:add(nn.Sequential():add(nn.MulConstant(0.229)):add(nn.AddConstant(0.485))) -- add red mean
  par:add(nn.Sequential():add(nn.MulConstant(0.224)):add(nn.AddConstant(0.456))) -- add green mean
  par:add(nn.Sequential():add(nn.MulConstant(0.225)):add(nn.AddConstant(0.406))) -- add blue mean

  net:add(par)

  net:add(nn.MapTable(nn.View(-1, 1, input_size , input_size)))

  net:add(nn.JoinTable(2))

  return net
end



function CreateVGGInputPreprocessor(b_mean, g_mean, r_mean, img_size)
  local net = nn.Sequential()

  net:add(nn.MulConstant(255)) -- (0, 1) -> (0, 255)
  local st = nn.SplitTable(2)
  net:add(st)
  net.name = 'vgg_normalization'
  local ct = nn.ConcatTable()
  ct:add(nn.SelectTable(3))
  ct:add(nn.SelectTable(2))
  ct:add(nn.SelectTable(1))
  net:add(ct)
  local  par = nn.ParallelTable()
  par:add(nn.AddConstant(-b_mean)) -- substract blue
  par:add(nn.AddConstant(-g_mean)) -- substract green
  par:add(nn.AddConstant(-r_mean)) -- substract red

  net:add(par)

  net:add(nn.MapTable(nn.View(-1, 1, img_size , img_size)))
  net:add(nn.JoinTable(2))

  return net
end


