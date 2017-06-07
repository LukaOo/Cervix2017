--[[
     Provider read data from input folders with class labels as root folder
     1, 2, 3
]]
require 'nn'
require 'math'
require 'paths'
require 'image'
local transform = require 'datasets/transforms'

do
    local Hdf5Provider = torch.class 'Hdf5Provider'
    ----
    ---- Input file, size of batch, name of data set i.e. train, test
    ---- 
    function Hdf5Provider:__init(config, batchSize)  

        self.InputPath = config.input_path    -- file with mask and images 
        self.batchSize = batchSize
        self.ds_name = config.data_set_name
        self.image_size = config.image_size or 224
        self.cinput_planes = config.cinput_planes or 3
        
        self.InputPath   = self.InputPath .. '/' .. config.data_set_name
        self.issiames    =  config.siames_input
        self.dual_target = config.dual_target
        self.triplets     = config.triplets
        self.bilinear     = config.bilinear
        
        if self.issiames == true then
           print ("Use siames data preprocessor")
           self.preprocessor = self:preprocess_siames()
        else
           print ("Use generic data preprocessor")
           self.preprocessor = self:preprocess()
        end
        
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
    
    -- Computed from random subset of ImageNet training images
    local meanstd = {
       mean = { 0.485, 0.456, 0.406 },
       std = { 0.229, 0.224, 0.225 },
    }
    local pca = {
       eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
       eigvec = torch.Tensor{
          { -0.5675,  0.7192,  0.4009 },
          { -0.5808, -0.0045, -0.8140 },
          { -0.5836, -0.6948,  0.4203 },
       },
    }
    

    function Hdf5Provider:preprocess()
        self.flip = 0
        if self.ds_name == 'train' then

              local rot_angle = torch.uniform(-10, 10)
              local minScale, maxScale   = self.image_size * 0.8, self.image_size * 1.3 
              local minTranslate, maxTranslate = -10, 10
              local minBlur, maxBlur           =  15, 30
              local minAlfa, maxAlfa           =  50, 100
              local minSigma, maxSigma         =  5, 10
              
              if self.cinput_planes == 3 then
                  return transform.Compose{
    --                      transform.Blur(0.2, minBlur, maxBlur),
                          transform.HorizontalFlip(0.5),
                          transform.Rotation(rot_angle),
                          transform.RandomScale(minScale, maxScale),
                          transform.CenterCrop(self.image_size, 68),
                          transform.Translate(minTranslate, maxTranslate),
                          transform.ElasticTransform(0.3, minAlfa, maxAlfa, minSigma, maxSigma),
                          transform.ColorJitter({
                                      brightness = 0.4,
                                      contrast = 0.4,
                                      saturation = 0.4,
                                  }),                      
                          transform.Lighting(0.1, pca.eigval, pca.eigvec),
--                          transform.MinMaxNorm(),
--                          transform.RedSaturation(0.7),
    --                      transform.MakeMonochromeGreenChannel(0.1),
                          transform.ColorNormalize(meanstd),
                          transform.AddNoise(0.05),
                        } 
               else
                  return transform.Compose{
    --                      transform.Blur(0.2, minBlur, maxBlur),
                          transform.HorizontalFlip(0.5),
                          transform.Rotation(rot_angle),
                          transform.RandomScale(minScale, maxScale),
                          transform.CenterCrop(self.image_size, 68),
                          transform.Translate(minTranslate, maxTranslate),
                          transform.AddNoise(0.05),
                        } 
               end
 
        else
              if self.cinput_planes == 3 then
                return transform.Compose{
                       transform.ColorNormalize(meanstd),
                      }
              else                      
                return transform.Compose{                       
                      }
              end
        end
    end
    
  function Hdf5Provider:preprocess_siames()

        if self.ds_name == 'train' then

              local rot_angle = torch.uniform(-10, 10)
              local minScale, maxScale   = self.image_size * 0.8, self.image_size * 1.3 
              local minTranslate, maxTranslate = -10, 10
              local minAlfa, maxAlfa           =  50, 100
              local minSigma, maxSigma         =  5, 10
              
              return transform.Compose{
                      transform.HorizontalFlip(0.5),
                      transform.Rotation(rot_angle),
 		                  transform.RandomScale(minScale, maxScale),
                      transform.CenterCrop(self.image_size, 68),
                      transform.Translate(minTranslate, maxTranslate),
                      transform.ElasticTransform(0.3, minAlfa, maxAlfa, minSigma, maxSigma),
                      
                      transform.Lighting(0.1, pca.eigval, pca.eigvec),
                      transform.MinMaxNorm(),
                      transform.RedSaturation(0.7),
                      
                      transform.ColorNormalize(meanstd),                      
                      
                      transform.AddNoise(0.05),
                    } 
 
        else
              return transform.Compose{
                     transform.ColorNormalize(meanstd),                      
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
    
    function Hdf5Provider:make_triplets_input(inputs, targets)
           local input2  = {}
           local target2 = {} -- similar target
           local input3 = {}
           local target3 = {} -- dessimilar target
           
           local class_idx = {}
           local class_ptr = {}
           local cls_cnt = 0
           -- permutate class indexies
           for c, v in pairs( self.classes_table ) do
              class_idx[c] = torch.randperm(#self.classes_table[c]):long()
              class_ptr[c] = 1
              cls_cnt = cls_cnt + 1
           end
           
           -- collect tripplets
           for i=1, targets:size(1) do
              local target = tostring( targets[i] )
              local cidx = class_idx[target][class_ptr[target]]
              local item_name = self:get_paths(self.classes_table[target][cidx], target)
              class_ptr[target] = class_ptr[target] +1
              local input, target =  self:get_tensors(item_name, target)
              input2[i] = input
              target2[i] = target
              -- select random opposite class
              local tt = torch.Tensor(100):random(1, cls_cnt)
              for j=1, tt:size(1) do
                if tt[j] ~= target then
                  target = tt[j]
                  break
                end
              end
              target = tostring(target)
              cidx = class_idx[target][class_ptr[target]]
              item_name = self:get_paths(self.classes_table[target][cidx], target)
              class_ptr[target] = class_ptr[target] +1
              input, target =  self:get_tensors(item_name, target)
              input3[i] = input
              target3[i] = target
           end
        return { torch.cat(input2, 1),  inputs,  torch.cat(input3, 1)}, torch.cat(target2, 1)
    end
    
    function Hdf5Provider:make_siames_input(input, target)
       if self.issiames == true or self.bilinear == true or  self.triplets == true then
         
         if self.triplets == true then
            return self:make_triplets_input(input, target)
         end
         
         if self.bilinear == true then
            return {input,input}, target
         end
         
         local input1 = input
         local input2 = nil
         local target2 = nil
         
         input2 = input:clone()
         local indices = torch.randperm(input2:size(1)):long()
         input2  = input2:index(1,indices)
         target2 = target:clone():index(1, indices)
         
         local o_target = torch.abs(target - target2)
         o_target[torch.gt(o_target, 0)] = -1
         o_target[torch.eq(o_target, 0)] = 1
         
         if self.dual_target == true then
             target = {target, o_target}
         else
             target = o_target
         end
         
         input = {input1, input2}
         
       end
       return input, target
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
        return self:make_siames_input( _input, _target )
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
       return self:make_siames_input(  _input, _target )
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
          local img = image.load(path_input, self.cinput_planes)
          local fld = 'data'
          if img:dim() == 2 then img = img:reshape( 1, img:size(1), img:size(2)) end
          -- get images
          local input  = self.preprocessor( img )
          input = resize_tensor(input)
          
          local target = torch.LongTensor({1}):fill(target)

        return input, target
    end
end    
