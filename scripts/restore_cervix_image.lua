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

function write_image(o_path, oimg)
    image.save(o_path, oimg)
end

if opt.input == opt.out then
   error ("Input path equal output path")
end

print (c.blue '==>' ..' loading model: '.. opt.model)

model = cast(torch.load(opt.model))
model:clearState()
model:evaluate()


if opt.backend == 'cudnn' then
   require 'cudnn'
   print ('Converting to cudnn')
   cudnn.convert(model, cudnn)
end

if opt.use_optnet == 1 then
   optnet = require 'optnet'
   model.__memoryOptimized = nil
   print ("Count used memory before: ", optnet.countUsedMemory(model))
   mod_input = cast(torch.rand(1,3,512,512))
   mod_opts = {inplace=true, mode='inference'}
   optnet.optimizeMemory(model, mod_input, mod_opts)
   print ("Count used memory after: ", optnet.countUsedMemory(model))
end
  
cudnn.benchmark = true

gpus = torch.range(1, cutorch.getDeviceCount()):totable()
print ('Gpus', gpus)

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
    local oimage = model:forward(input):clone():float():reshape(input:size(2), input:size(3), input:size(4))
    write_image(opt.out .. '/' .. files_list[i], oimage)
    collectgarbage()
end



