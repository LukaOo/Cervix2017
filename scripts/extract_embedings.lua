require 'xlua'
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
local transform = require 'datasets/transforms'

hdf5 = require 'hdf5'

local c = require 'trepl.colorize'

opt = lapp[[
   -i,--input                 (default "")          input data path
   -o,--out                   (default "")          output data path
   -m,--model                 (default "")          model file name
   --backend                  (default nn)          backend
   --type                     (default 'cuda')      type of model
   --use_optnet               (default 0)           use memory optimisation by optnet
   --image_size               (default 224)         neural network input image size
   -b,--batch_size            (default 6)           size of batch
]]

print(opt)

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

if file_exists(opt.model) == false then
  error ('Can not open model file: ' .. opt.model)
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

function preprocess()
      return transform.Compose{
             transform.Scale(opt.image_size),
            }
end


function getFilesList(InputPath)
      local files_list = {}
      local i = 1
      for f in paths.files(InputPath) do
           if paths.filep(InputPath .. '/' .. f) then
              files_list[i] = f
              i = i + 1
           end
      end
      return files_list
end

function get_image(input_path)
    img = image.load(input_path)
    return img
end

function write_embeding(o_path, embeding)
    local h5_file = hdf5.open(o_path, 'w')
    local options = hdf5.DataSetOptions()
    options:setChunked(32, 32)
    options:setDeflate()
    h5_file:write( 'embeding', embeding, options)
    h5_file:close()
end

if opt.input == opt.out then
   error ("Input path equal output path")
end

print (c.blue '==>' ..' loading model: '.. opt.model)

model = cast(torch.load(opt.model))
model:clearState()
model:evaluate()


print (c.green '==> Model quality: ' .. model.last_error)


if opt.backend == 'cudnn' then
   require 'cudnn'
   print ('Converting to cudnn')
   cudnn.convert(model, cudnn)
end

if opt.use_optnet == 1 then
   optnet = require 'optnet'
   model.__memoryOptimized = nil
   print ("Count used memory before: ", optnet.countUsedMemory(model))
   mod_input = cast(torch.rand(1,3,opt.image_size, opt.image_size))
   mod_opts = {inplace=true, mode='inference'}
   optnet.optimizeMemory(model, mod_input, mod_opts)
   print ("Count used memory after: ", optnet.countUsedMemory(model))
end
  
cudnn.benchmark = true

gpus = torch.range(1, cutorch.getDeviceCount()):totable()
print ('Gpus', gpus)

local pt = model:findModules('nn.ParallelTable')

if pt ~= nil and #pt > 0 then
   -- get second model for embeding
   model = pt[1]:get(2)
end


if #gpus > 1 then
  model = nn.DataParallelTable(1):add(model, gpus):cuda()
end


local files_list = getFilesList(opt.input)


print ('Total input files: ', #files_list)

for i=1, #files_list do
    xlua.progress(i, #files_list)
 
    local input = get_image(opt.input .. '/' .. files_list[i])
    local input_size = input:size()
    input = input:reshape(1, input:size(1), input:size(2), input:size(3)):cuda()
    local embeding = model:forward(input):clone():float()
    
    mask = embeding:reshape(embeding:size(2))
        
    write_embeding(opt.out .. '/' .. files_list[i] .. '.h5', embeding)
    collectgarbage()
end



