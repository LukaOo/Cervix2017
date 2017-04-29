require 'image'
require 'dpnn'

local M = {}

function M.Compose(transforms)
   return function(input)
      for _, transform in ipairs(transforms) do
         input = transform(input)
      end
      return input
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end

-- Adds noise to the image
-- Parameters:
-- @param im (tensor): input image
-- @param augNoise (float): the standard deviation of the white noise
-- ref: https://github.com/brainstorm-ai/DIGITS/blob/6a150cfbed2aa7dd70992036dfbdf66ee088fba0/tools/torch/data.lua#L135
function M.AddNoise(augNoise)
  return function(input)
     -- AWGN:
     -- torch.randn makes noise with mean 0 and variance 1 (=stddev 1)
     --  so we multiply the tensor with our augNoise factor, that has a linear relation with
     --  the standard deviation (but the variance will be increased quadratically).
	 if type(input) == 'table' then
             return {torch.add(input[1], torch.randn(input[1]:size()) * augNoise), input[2]}
	 end
  	    return  torch.add(input, torch.randn(input:size()) * augNoise)
  end
end

-- Translate image
function M.Translate(min, max)
   return function(input)
      local x = math.random(min,max)
      local y = math.random(min,max)

      local i = nil
      if type(input) == 'table' then 
         return {image.translate(input[1], x, y), image.translate(input[2], x, y)}
      else
         return image.translate(input, x, y)
      end
   end
end


-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local i = nil
      if type(input) == 'table' then 
        i = input[1]
      else
        i = input
      end

      local w, h = i:size(3), i:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
         if type(input) == 'table' then
            return {image.scale(input[1], size, h/w * size, interpolation), image.scale(input[2], size, h/w * size, interpolation)} 
         else
            return image.scale(input, size, h/w * size, interpolation)
         end
      else
         if type(input) == 'table' then
            return { image.scale(input[1], w/h * size, size, interpolation), image.scale(input[2], w/h * size, size, interpolation)}
         else
            return image.scale(input, w/h * size, size, interpolation)
         end
      end
   end
end

function M.SpatialRegionDropout(p)
   local net = nn.SpatialRegionDropout(p)
   return function(input)
       return net:forward(input)
   end  
end

function M.MakeMonochromeGreenChannel(p)
   local prob = p
   return function(input)
        if  torch.uniform() < prob then 
            grayscale = (input[1] * 0.21 + input[2] * 0.72 + input[3] * 0.07):reshape(1, input:size(2), input:size(3))
            zerochannel = torch.DoubleTensor(1,input:size(2), input:size(3)):zero()
            input = torch.cat({zerochannel, grayscale, zerochannel}, 1)
        end
       return input
   end  
end

-- Crop to centered rectangle
function M.CenterCrop(size, padding)
   padding = padding or 0
   return function(input)
      local i  = type(input) ~= 'table' and input or input[1]
      if padding > 0 then
         local temp = i.new(3, i:size(2) + 2*padding, i:size(3) + 2*padding)
         temp:zero()
             :narrow(2, padding+1, i:size(2))
             :narrow(3, padding+1, i:size(3))
             :copy(i)
        i = temp
	if type(input) == 'table' then
	   input[1] = temp
	   temp = input[2].new(1, input[2]:size(2) + 2*padding, input[2]:size(3) + 2*padding)
           temp:zero()
               :narrow(2, padding+1, input[2]:size(2))
               :narrow(3, padding+1, input[2]:size(3))
               :copy(input[2])
	       input[2] = temp
	 else
	       input = temp
	 end
      end

      local w1 = math.ceil((i:size(3) - size)/2)
      local h1 = math.ceil((i:size(2) - size)/2)
      local r

      if type(input) ~= 'table' then
        r = image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
      else
	    r= {}
		r[1] = image.crop(input[1], w1, h1, w1 + size, h1 + size)
		r[2] = image.crop(input[2], w1, h1, w1 + size, h1 + size)
       end
       return r
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input)
      local i = type(input) == 'table' and input[1] or input
      if padding > 0 then
         local temp = i.new(3, i:size(2) + 2*padding, i:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, i:size(2))
            :narrow(3, padding+1, i:size(3))
            :copy(i)
            i = temp
	    if type(input) == 'table' then
	        input[1] = temp
	        temp = input[2].new(3, i:size(2) + 2*padding, i:size(3) + 2*padding)
                temp:zero()
                    :narrow(2, padding+1, i:size(2))
                    :narrow(3, padding+1, i:size(3))
                    :copy(input[2])
		    input[2] = temp
		 else
		    input = temp
		 end
      end

      local w, h = i:size(3), i:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = nil
	  if type(input) == 'table' then
	     out = {}
	     out[1] = image.crop(input[1], x1, y1, x1 + size, y1 + size)
		 out[2] = image.crop(input[2], x1, y1, x1 + size, y1 + size)
         assert(out[1]:size(2) == size and out[1]:size(3) == size, 'wrong crop size')
	  else
	  	 out = image.crop(input, x1, y1, x1 + size, y1 + size)
         assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
	  end	 
      return out
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = nil, nil
      if (type(input) ~= 'table') then 
          w, h = input:size(3), input:size(2)
       else
          w, h = input[1]:size(3), input[1]:size(2)
       end


      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end
	  local r = nil
      if type(input) == 'table' then
	     r = {}
		 r[1] = image.scale(input[1], targetW, targetH, 'bicubic')
		 r[2] = image.scale(input[2], targetW, targetH, 'bicubic')
	  else
	     r = image.scale(input, targetW, targetH, 'bicubic')
	  end
      return r
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
        if type(input) == 'table' then
            assert(#input == 2)
            input[1] = image.hflip(input[1])
	          input[2] = image.hflip(input[2])
         else
            input = image.hflip(input)
		    end
      end
      return input
   end
end

function M.VerticalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
         input = image.vflip(input)
      end
      return input
   end
end

function M.Blur(prob, min, max)
   return function(input)
      if torch.uniform() < prob then
         local B = math.random(min,max)
         local K = image.gaussian(B)
         input = image.convolve(input,K,'same')
         local imin, imax = torch.min(input), torch.max(input)
         input = (input - imin) / (imax-imin)
      end
      return input
   end
end

function M.Rotation(deg)
   return function(input)
      if deg ~= 0 then
	     grad = (torch.uniform() - 0.5) * deg * math.pi / 180
	     if type(input) == 'table' then
		      assert(#input == 2)
		      input[1] = image.rotate(input[1], grad, 'bilinear')
		      input[2] = image.rotate(input[2], grad, 'bilinear')
       else
		      input     = image.rotate(input, grad, 'bilinear')
	     end
      end
      return input
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end
function M.RandomOrder(ts)
   return function(input)
      local img = input.img or input
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img = ts[order[i]](img)
      end
      return img
   end
end

function M.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input end
   end

   return M.RandomOrder(ts)
end

return M
