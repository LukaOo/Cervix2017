--[[
     Provider read data from h5 embding file
     1, 2, 3
]]
require 'nn'
require 'math'
require 'paths'
local transform = require 'datasets/transforms'
hdf5 = require 'hdf5'

do
    local Hdf5Provider = torch.class 'Hdf5Provider'
    ----
    ---- Input file, size of batch, name of data set i.e. train, test
    ---- 
    function Hdf5Provider:__init(config, batchSize)  

        self.InputPath = config.input_path    -- file with mask and images 
        self.batchSize = batchSize
        self.ds_name = config.data_set_name
        
        self.InputPath   = self.InputPath .. '/' .. config.data_set_name        
        self.preprocessor = self:preprocess()
        
        self.classes_labels = self:getClassesList(self.InputPath)
        self.classes_table = {}
        
        for k, v in pairs(self.classes_labels) do
           self.classes_table[k] = self:getFilesList(v)
           print (k, #self.classes_table[k])
        end
        
        self.list = {}

        -- merge all files into single list
        k = 1
        for c, v in pairs( self.classes_labels ) do
           for i = 1, #self.classes_table[c] do
              self.list[k] = {[1] = c, [2] = i}  -- store reference into class and index
              k = k + 1
           end
        end
               
        self.batch_idx = 1
        self.cbatches = math.ceil(#self.list/self.batchSize)
    end

    function Hdf5Provider:getClassesList(InputPath)
      local classes = {}
      for f in paths.files(InputPath) do
         if f ~= '.' and f ~= '..' then
           local dir = InputPath .. '/' .. f
           if paths.dirp(dir) == true then
              classes[f] = dir
           end         
          end
      end
      return classes
    end

    function Hdf5Provider:getFilesList(InputPath)
      local files_list = {}
      local i = 1
      print (InputPath)
      for f in paths.files(InputPath) do
           if paths.filep(InputPath .. '/' .. f) then
              files_list[i] = f
              i = i + 1
           end
      end
      print (#files_list)
      return files_list
    end
        

    function Hdf5Provider:preprocess()
        if self.ds_name == 'train' then
                return transform.Compose{    
                       transform.AddNoise(0.5),
                      }
        else
                return transform.Compose{                       
                      }
        end
    end
    
    function Hdf5Provider:reset()
        self.indices = torch.randperm(#self.list):long():split(self.batchSize)
        -- self.indices[#self.indices] = nil
        self.batch_idx = 1
    end

    function Hdf5Provider:size()
      return #self.list
    end
        
    -- sub sampling idexies from data set
    function Hdf5Provider:__sub_indices(v, permutate)
      local _input   = {}
      local _target  = {}
      self.item_paths = {}
      local permutate = permutate or false
      
      for im_idx =1, v:size(1) do
        -- read input and target
        local target, k = self.list[v[im_idx]][1], self.list[v[im_idx]][2]
        local item_name = self:get_paths(self.classes_table[target][k], target)
          
        local input, target =  self:get_tensors(item_name, target)
      
        _input[im_idx]  =  input
        _target[im_idx] =  target
        self.item_paths[im_idx] = item_name  
      end    
      if #_input > 0 then
        _input  = torch.cat(_input, 1)
        _target = torch.cat(_target, 1)
         
        -- random shuffle
        if permutate == true then
            indices = torch.randperm(_input:size(1)):long()
           _input  = _input:index(1,indices)
           _target = _target:index(1,indices)
        end
          
      else
        _input  = nil
        _target = nil
      end
      
      return _input, _target 
    end
  
    function Hdf5Provider:get_next_batch()
        local _input   = nil
        local _target  = nil

        if self.batch_idx <= self.cbatches then
        
          _input, _target = self:__sub_indices(self.indices[self.batch_idx], true)
          
        end


        self.batch_idx = self.batch_idx + 1
        return _input, _target
    end

    ---
    --- get images from to positions
    --- 
    function Hdf5Provider:sub(start, stop)
        local _input   = nil
        local _target  = nil

        stop = math.min(stop, #self.list)

        if start <= stop then    
            indices = torch.linspace(start, stop, stop-start+1)
            _input, _target = self:__sub_indices(indices, false)
        end

       self.batch_idx = self.batch_idx + 1
       return   _input, _target 
    end

    function Hdf5Provider:get_paths( file, target )
        return string.format("%s/%d/%s", self.InputPath, target, file), target
    end


    local function resize_tensor(input)
           if input:dim() == 2 then
              -- contains one slice 
              input = input:reshape(1, 1, input:size(1), input:size(2))
           else
              -- contains 3d image 
              input = input:reshape(1, input:size(1), input:size(2), input:size(3))
           end 
           return input
    end

    function Hdf5Provider:get_tensors(path_input, target)
          local h5_file = hdf5.open(path_input)
          local fld = 'embeding'
          
          -- get input
          local input  =  h5_file:read( fld ):all()
          local input  = self.preprocessor( input ) 
          
          local target = torch.LongTensor({1}):fill(target)
          
          h5_file:close()

        return input, target
    end
end    
