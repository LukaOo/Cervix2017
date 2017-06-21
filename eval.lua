require 'xlua'
require 'nn'
require 'dpnn'
require 'models/gradient_decrease'
require 'models/spatial_bilinear.lua'
local utils=require 'utils'

local c = require 'trepl.colorize'

opt = lapp[[
   -i,--input                 (default "")          input data file
   -s,--save                  (default "result.txt")  result file to save
   -b,--batchSize             (default 10)         batch size
   -m, --model                (default '')         model file name
   --type                     (default cuda)       type of processors
   --provider_config          (default "{data_set_name='', provider='datasets/h5-dir-provider', image_size=224}" )      provider configuration
]]

print(opt)

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

if opt.input == '' or file_exists(opt.input) == false then
     error("Can not open file: '"..opt.input.."'")
end

if opt.model == '' or file_exists(opt.model) == false then
     error("Can not open file: '"..opt.model.."'")
end

provider_config = loadstring(" return " .. opt.provider_config) ()
local sprovider   = provider_config.provider..'.lua'
dofile (sprovider)

print (c.cyan'Loading model', opt.model)
model = cast(torch.load(opt.model))

if opt.backend == 'cudnn' then
   require 'cudnn'
   print ('Converting to cudnn')
   cudnn.convert(model, cudnn)
end
gpus = torch.range(1, cutorch.getDeviceCount()):totable()
print ('Gpus', gpus)

if #gpus > 1 then
  dpt = nn.DataParallelTable(1):add(model, gpus):cuda()
else
  dpt = model
end

print(model)
print(c.blue '==>' ..' loading data ' .. opt.input)

provider_config.input_path = opt.input

testData = Hdf5Provider( provider_config, opt.batchSize )

softmax = cast( nn.SoftMax() )
criterion = cast(nn.CrossEntropyCriterion())

local function cast_target(inputs)
  
    if type(inputs) == 'table' then
       for ii=1, #inputs do
          inputs[ii] = cast(inputs[ii])
       end
    else
          inputs  = cast(inputs)
    end
    
    return inputs
end

print ("Model quality: ", model.last_error)

function test()
  -- disable flips, dropouts and batch normalization
  dpt:evaluate()
  print(c.blue '==>'.." testing")
  local bs = opt.batchSize
  local sum_error = 0
  local iters = 0
  local cc  = math.ceil(testData:size()/bs)
  local fd = io.open(opt.save, 'w')


  
  testData:reset()
  for i=1, cc do
    xlua.progress(i,  cc)
    
    local k = (i-1)*bs + 1
    
    local inputs, targets  = testData:sub(k,k+bs-1)

    inputs  = cast_target(inputs)
    targets = cast_target(targets)
    local outputs = dpt:forward(inputs)
    local y, idx = torch.max(softmax:forward(outputs), 2)
    
    for ii=1, targets:size(1) do
      fd:write(testData.item_paths[ii] .. '\t' .. targets[ii] .. '\t' .. idx[ii][1] .. '\t' .. y[ii][1].. '\n')
    end
    
    sum_error = sum_error + criterion:forward(outputs, targets)
    iters = iters + 1
  
  end
  fd:close()
  print ("Result logloss: ", sum_error / iters )
end

test()
