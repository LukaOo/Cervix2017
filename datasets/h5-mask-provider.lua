
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
        self.image_size = config.image_size
        self.input_augmentation = self:augment_input()
        
        self.InputPath = self.InputPath .. '/' .. self.ds_name
    
        self.list = self:getFilesList()
        self.batch_idx = 1
        self.cbatches = math.ceil(#self.list/self.batchSize)
    end

    function Hdf5Provider:getFilesList()
      local files_list = {}
      local i = 1
      print (self.InputPath)
      for f in paths.files(self.InputPath) do
           if paths.filep(self.InputPath .. '/' .. f) then
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
    
    function Hdf5Provider:augment_input()
        if self.ds_name == 'train' then
          
              local minBlur, maxBlur  =  15, 30
              
              return transform.Compose{                      
                      -- transform.Blur(0.2, minBlur, maxBlur),
                      transform.ColorJitter({
                                  brightness = 0.4,
                                  contrast = 0.4,
                                  saturation = 0.4,
                              }),                      
                      transform.Lighting(0.1, pca.eigval, pca.eigvec),
                      transform.MakeMonochromeGreenChannel(0.1),
                      transform.AddNoise(0.05),
                    }
        else
              return transform.Compose{
                     
                    }
        end
    end


    function Hdf5Provider:augment()
        self.flip = 0
        if self.ds_name == 'train' then

              local rot_angle = torch.uniform(-20, 20)
              local minScale, maxScale   = self.image_size*0.8, self.image_size * 1.3 
              local minTranslate, maxTranslate = -20, 20
              if  torch.uniform() < 0.5 then 
                  self.flip       = 1
              end

              return transform.Compose{
                      transform.HorizontalFlip(self.flip),
                      transform.Rotation(rot_angle),
 		      transform.RandomScale(minScale, maxScale),
                      transform.CenterCrop(self.image_size, 54),
                      transform.Translate(minTranslate, maxTranslate),
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
    
    function Hdf5Provider:get_next_batch()
        local _input   = {}
        local _target  = {}

        if self.batch_idx <= self.cbatches then
        
            local v  = self.indices[self.batch_idx]

            for im_idx =1, v:size(1) do
              -- read input and target
              local input, target =  self:get_tensors(self:get_paths(self.list[v[im_idx]]))

              _input[im_idx]  =  input
              _target[im_idx] =  target


            end    
        end

        if #_input > 0 then
          _input  = torch.cat(_input, 1)
          _target = torch.cat(_target, 1)

           
          -- random shaffle
            indices = torch.randperm(_input:size(1)):long()
           _input  = _input:index(1,indices)
           _target = _target:index(1,indices)

            
        else
          _input  = nil
          _target = nil
        end

        self.batch_idx = self.batch_idx + 1
        return _input, _target
    end

    ---
    --- get images from to positions
    --- 
    function Hdf5Provider:sub(start, stop)
        local _input   = {}
        local _target  = {}

        stop = math.min(stop, #self.list)

        if start <= stop then

            local c_idx = 1
            for im_idx =start, stop do
              -- read input and target
              local input, target =  self:get_tensors(self:get_paths( self.list[im_idx]))

              _input[c_idx]  =  input
              _target[c_idx] =  target

              c_idx = c_idx + 1
            end
        end

        if #_input > 0 then

          _input  = torch.cat(_input, 1)
          _target = torch.cat(_target, 1)

        else
          _input  = nil
          _target = nil

        end

       self.batch_idx = self.batch_idx + 1
       return _input, _target
    end

    function Hdf5Provider:get_paths( file )                
        return string.format("%s/%s", self.InputPath, file)
    end

    local function resize_tensor(input)
           if input:dim() == 2 then
              input = input:reshape(1, 1, input:size(1), input:size(2))
           else
              input = input:reshape(1, input:size(1), input:size(2), input:size(3))
           end 
           return input
    end

    function Hdf5Provider:get_tensors(path_input)
          local h5_file = hdf5.open(path_input)
          local augmentation = self:augment()
          local fld = 'image'
          
          -- get images
          local input  =  h5_file:read( fld ):all():clone():double()/255.0
          local target =  h5_file:read('mask'):all():clone():double() /255.0
          input = augmentation({input, target:reshape(1, target:size(1), target:size(2))})
          target = input[2]
          input  = input[1]
          -- agment input only, add noise and jitter collors
          input =  resize_tensor( self.input_augmentation(input) )
          target = resize_tensor(target)
          -- normalize target to 0-1
          target[torch.gt(target, 0.499999999)] = 1
          target[torch.lt(target, 0.5)] = 0
          
          h5_file:close()
        return input, target
    end
end    
