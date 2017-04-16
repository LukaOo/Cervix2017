require 'nn'
require 'image'

require 'cutorch'
require 'cunn'
require 'cudnn'

local stringx = require 'pl.stringx'
stringx.import()

local utils = {}

utils.meanVGG = torch.FloatTensor({129.1863, 104.7624, 93.5940})

utils.meanstd_fb_res_net   = {
        mean = { 0.485, 0.456, 0.406 },
        std = { 0.229, 0.224, 0.225 },
   }


function utils.TensorSizeToString(tensor)
  return (' '):join(torch.totable(tensor:size()))
end

function utils.PrintTensorStat(name, tensor)
  sorted_tensor  = torch.sort(torch.view(tensor, -1))
  len = sorted_tensor:nElement()
  
  len_name = #name
  tensor_name = name .. string.rep(' ', 20-len_name)
  s = 'Tensor name: ' .. tensor_name
  s = s .. ' min: ' .. sorted_tensor[1]
  s = s .. ' 1st quartile: ' .. sorted_tensor[math.floor(len/4)]
  s = s .. ' median: ' ..  sorted_tensor[math.floor(len/2)]
  s = s .. ' 3rd quartile: ' .. sorted_tensor[math.floor(3*len/4)]
  s = s .. ' max: ' .. sorted_tensor[len]
  print(s)
end

function utils.Create255GeneratorOutputPreprocessor()
  local net = nn.Sequential()
  net:add(nn.MulConstant(255))
  return net
end 

function utils.CreateGeneratorOutputPreprocessor()
  net = nn.Sequential()
  net:add(nn.AddConstant(1)) -- (-1, 1) -> (0, 2)
  net:add(nn.MulConstant(255/2)) -- (0, 2) -> (0, 255)
  net:add(nn.SplitTable(2, 4)) -- split by channels, four dimensions

  par = nn.ParallelTable()
  par:add(nn.AddConstant(-93.5940)) -- substract red
  par:add(nn.AddConstant(-104.7624)) -- substract green
  par:add(nn.AddConstant(-129.1863)) -- substract blue

  net:add(par)

  net:add(nn.MapTable(nn.View(-1, 1, 224 ,224)))
  net:add(nn.JoinTable(2))

  return net
end

function utils.CreateDiscriminatorInputPreprocessor()
  local net = nn.Sequential()
  net:add(nn.AddConstant(1)) -- (-1, 1) -> (0, 2)
  net:add(nn.MulConstant(1/2)) -- (0, 2) -> (0, 1)
  return net
end


function utils.CreateResnetInputPreprocessor()
    
  local net = nn.Sequential()
  local st = nn.SplitTable(2)
  net:add(st)
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

  net:add(nn.MapTable(nn.View(-1, 1, 224 ,224)))
    
  net:add(nn.JoinTable(2))
    
  return net
end

function utils.CreateVGGInputPreprocessor()
  local net = nn.Sequential()
    
  net:add(nn.MulConstant(255)) -- (0, 1) -> (0, 255)
  local st = nn.SplitTable(2)
  net:add(st)
  local ct = nn.ConcatTable()
  ct:add(nn.SelectTable(3))
  ct:add(nn.SelectTable(2))
  ct:add(nn.SelectTable(1))
  net:add(ct)
  local  par = nn.ParallelTable()
  par:add(nn.AddConstant(-129.1863)) -- substract blue
  par:add(nn.AddConstant(-104.7624)) -- substract green
  par:add(nn.AddConstant(-93.5940)) -- substract red

  net:add(par)

  net:add(nn.MapTable(nn.View(-1, 1, 224 ,224)))
  net:add(nn.JoinTable(2))
    
  return net
end


function utils.PreprocessVGG(img)
  local new_img = img * 255
  -- RGB -> BGR
  local new_img = new_img:index(1, torch.LongTensor{3, 2, 1})
  -- substract mean
  for i = 1, 3 do
      new_img[i]:add(-utils.meanVGG[i])
  end
  return new_img
end

function utils.DeprocessVGG(img)
  local new_img = utils.GeneratorOutputToNormal(img) 

  -- BGR -> RGB
  new_img = new_img:index(1, torch.LongTensor{3, 2, 1})
  return new_img
end

function utils.PreprocessManyVGG(images)
  local new_images = images:clone()
  for i = 1, images:size(1) do
    new_images[i] = utils.PreprocessVGG(images[i])
  end
  return new_images
end

function utils.DeprocessManyVGG(images)
  local new_images = images:clone()
  for i = 1, images:size(1) do
    new_images[i] = utils.DeprocessVGG(images[i])
  end
  return new_images
end


function utils.GeneratorOutputToNormal(img)
  local new_image = img:clone()
  new_image:add(1) -- -1,1 -> 0,2
  new_image:div(2) -- 0,2 -> 0,1

  return new_image
end

function utils.PreprocessGeneratorOutput(img)
  local new_img = utils.GeneratorOutputToNormal(img)
  new_img = img * 255 --channels are already inverted
  -- substract mean
  for i = 1, 3 do
      new_img[i]:add(-utils.meanVGG[i])
  end
  return new_img
end


function utils.PreprocessManyGeneratorOutput(images)
  local new_images = images:clone()
  for i = 1, images:size(1) do
    new_images[i] = utils.PreprocessGeneratorOutput(images[i])
  end
  return new_images
end


function utils.ConvertToCudnn(net)
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cudnn.convert(net, cudnn)
  net:cuda()
end


function utils.ConvInit(model, name)
  for k,v in pairs(model:findModules(name)) do
     local n = nil 
     if v.kT == nil then
      n = v.kW*v.kH*v.nOutputPlane
     else
      n = v.kW*v.kH*v.kT*v.nOutputPlane
     end
      v.weight:normal(0,math.sqrt(2/n))
     -- v.weight:normal(0.0, 0.02)   
     if cudnn.version >= 4000 then
--        v.bias = nil
--        v.gradBias = nil
     else
--        v.bias:zero()
     end
  end
end

function utils.BNInit(model, name)
  for k,v in pairs(model:findModules(name)) do
     --if v.weight then v.weight:normal(1.0, 0.02) end
     v.bias:zero()
  end
end

function utils.InitNetwork(model)
  utils.ConvInit(model, 'cudnn.SpatialConvolution')
  utils.ConvInit(model, 'nn.SpatialConvolution')
  utils.ConvInit(model, 'nn.SpatialFullConvolution')
  utils.ConvInit(model, 'cudnn.SpatialFullConvolution')
  utils.ConvInit(model, 'nn.VolumetricConvolution')
  utils.ConvInit(model, 'cudnn.VolumetricConvolution')
  utils.BNInit(model, 'cudnn.SpatialBatchNormalization')
  utils.BNInit(model, 'nn.SpatialBatchNormalization')
  utils.BNInit(model, 'nn.BatchNormalization')
  utils.BNInit(model, 'cudnn.BatchNormalization')  
end

return utils
