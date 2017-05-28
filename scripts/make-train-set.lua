--[[
     Make random sampled trainset split it into train and test
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
   --transform            (default nil)    transform configuration or nil if no transform required '{output_image_size=<>, crop_size=<>}'
   --prefix               (default '')     add prefix to output files
]]

print(opt)
transform_config = loadstring(" return " .. opt.transform)()

local InputPath = opt.input

local files_list = {}
local categories = {}
i = 1

print ("Lookup input path: ", InputPath)

for d in paths.files(InputPath) do      
     idx = d:find('_')
     if idx ~= nil then
       cat = d:sub(idx+1)
       if tonumber(cat) ~= nil then
         if categories[cat] == nil then categories[#categories+1] = cat end
         if paths.dirp(InputPath .. '/' .. d) then
           
           local dir = InputPath .. '/' .. d
           for f in paths.files(dir) do
             if paths.filep(dir .. '/' .. f) then
                -- store category value, file name & full path to source
                files_list[i] = { [1] = cat, [2]=f, [3]=dir .. '/'.. f}
                i = i + 1
             end
           end
         end
       end
     end
end

perm_idx =  torch.randperm(i-1)

--[[
    Prepare folder to copy data set
]]
function prep_folder(output, set_name, categories)
  
    if paths.dirp(output) == false then
       paths.mkdir(output)
    end

    test_path = output .. '/' .. set_name
    
    print ("Prepare path: ",  test_path) 

    if paths.dirp(test_path) == false then
       paths.mkdir(test_path)
    end

    for num, cat in pairs(categories) do
       cat_dir = test_path .. '/' .. cat
       if paths.dirp(cat_dir) == false then
          paths.mkdir(cat_dir)
       else
           for f in paths.files(cat_dir) do
             if paths.filep(cat_dir .. '/' .. f) then
                os.remove(cat_dir .. '/' .. f)
             end
           end           
       end
    end
end

function preprocess()
      return transform.Compose{
             transform.Scale(transform_config.output_image_size),
             transform.CenterCrop(transform_config.crop_size, 0),
             
            }
end

if transform_config ~= nil then
   preprocessor = preprocess()
end

function os_copy (source_path, dest_path) 
    if transform_config == nil then
      local source = io.open(source_path, "rb")  
      local dest = io.open(dest_path, "wb")  
      dest:write(source:read("*a"))  
      source:close()    
      dest:close()
    else
      img = image.load(source_path)
      img = preprocessor(img)
      image.save(dest_path, img)
    end
end

local ds_name =  'test'

prep_folder(opt.output, ds_name, categories)
local csamples = torch.ceil( perm_idx:size(1) * opt.frac)
for i=1, csamples do
    xlua.progress(i, csamples)
    local idx = perm_idx[i]
    os_copy(files_list[idx][3], opt.output .. '/' .. ds_name .. '/' .. files_list[idx][1] .. '/'.. opt.prefix .. files_list[idx][2] )
end

local ds_name =  'train'
prep_folder(opt.output, ds_name, categories)
local csamples = perm_idx:size(1) - torch.ceil( perm_idx:size(1) * opt.frac)
local k = 1
for i=torch.ceil( perm_idx:size(1) * opt.frac)+1, perm_idx:size(1) do
    xlua.progress(k, csamples)
    local idx = perm_idx[i]
    os_copy(files_list[idx][3], opt.output .. '/' .. ds_name .. '/' .. files_list[idx][1] .. '/' .. opt.prefix .. files_list[idx][2] )
    k = k + 1
end

print ("Done")