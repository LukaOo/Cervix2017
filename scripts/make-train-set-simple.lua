--[[
     Make random sampled trainset split it into train and test witout split it on classes
     plane input folder
     plane output train and test folders
]]

require 'nn'
require 'xlua'
require 'paths'
require 'image'
local transform = require 'datasets/transforms'

opt = lapp[[
   -i, --input            (default "")     input directory where all data placed 
   -o, --output           (default "")     directory to save test and train set
   -f, --frac             (default 0.1)    fraction subsampling of test set
   --prefix               (default '')     add prefix to output files
]]

print(opt)

local InputPath = opt.input
local files_list = {}
local i=1

for f in paths.files(InputPath) do      
     if paths.filep(InputPath .. '/' .. f) then
        -- store category value, file name & full path to source
        files_list[i] = f
        i = i + 1
     end
end
--[[
    Prepare folder to copy data set
]]
function prep_folder(output, set_name)
  
    if paths.dirp(output) == false then
       paths.mkdir(output)
    end

    test_path = output .. '/' .. set_name
    
    print ("Prepare path: ",  test_path) 

    if paths.dirp(test_path) == false then
       paths.mkdir(test_path)
    end
    for f in paths.files(test_path) do
       if paths.filep(test_path .. '/' .. f) then
          os.remove(test_path .. '/' .. f)
       end
    end  
end
function os_copy (source_path, dest_path) 
      local source = io.open(source_path, "rb")  
      local dest = io.open(dest_path, "wb")  
      dest:write(source:read("*a"))  
      source:close()    
      dest:close()
end

perm_idx =  torch.randperm(i-1)
local ds_name =  'test'

prep_folder(opt.output, ds_name)
local csamples = torch.ceil( perm_idx:size(1) * opt.frac)

for i=1, csamples do
    xlua.progress(i, csamples)
    local idx = perm_idx[i]
    os_copy(InputPath .. '/' .. files_list[idx], opt.output .. '/' .. ds_name .. '/' .. files_list[idx] )
end

local ds_name =  'train'
prep_folder(opt.output, ds_name)
local csamples = perm_idx:size(1) - torch.ceil( perm_idx:size(1) * opt.frac)
local k = 1
for i=torch.ceil( perm_idx:size(1) * opt.frac)+1, perm_idx:size(1) do
    xlua.progress(k, csamples)
    local idx = perm_idx[i]
    os_copy( InputPath .. '/' .. files_list[idx], opt.output .. '/' .. ds_name .. '/' .. files_list[idx] )
    k = k + 1
end

print ("Done")