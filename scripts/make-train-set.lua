--[[
     Make random sampled trainset split it into train and test
]]

require 'nn'
require 'xlua'
require 'paths'

opt = lapp[[
   -i, --input            (default "")     input directory where all data placed 
   -o, --output           (default "")     directory to save test and train set
   -f, --frac             (default 0.1)    fraction subsampling of test set
]]

print(opt)

local InputPath = opt.input

local files_list = {}
local categories = {}
i = 1

print ("Lookup input path: ", InputPath)

for d in paths.files(InputPath) do      
     idx = d:find('_')
     if idx ~= nil then
       cat = d:sub(idx+1)
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

perm_idx =  torch.randperm(i)

--[[
    Prepare folder to copy data set
]]
function prep_folder(output, set_name, categories)
  
    test_path = output + '/' + set_name
    
    print ("Prepare path: ",  test_path) 

    if paths.dirp(test_path) == false then
       paths.mkdir(test_path)
    end

    for num, cat in pairs(categories) do
       cat_dir = test_path + '/' + cat
       if paths.dirp(cat_dir) == false then
          paths.mkdir(cat_dir)
       else
           print ("Clear directory: ", cat_dir)
           for f in paths.files(cat_dir) do
             if paths.filep(cat_dir .. '/' .. f) then
                os.remove(cat_dir .. '/' .. f)
             end
           end
           
           print ("Done")
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

local ds_name =  'test'

prep_folder(opt.output, ds_name, categories)
local csamples = torch.ceil( #perm_idx * opt.frac)
for i=1, csamples do
    xlua.progress(i, #perm_idx)
    local idx = perm_idx[i]
    os_copy(files_list[idx][3], opt.output + '/' + ds_name )
end

local ds_name =  'train'
prep_folder(opt.output, ds_name, categories)
local csamples = #perm_idx
for i=torch.ceil( #perm_idx * opt.frac), #perm_idx do
    xlua.progress(i, #perm_idx)
    local idx = perm_idx[i]
    os_copy(files_list[idx][3], opt.output + '/' + ds_name )
end

print ("Done")