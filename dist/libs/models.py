import torch
import torch.nn as nn
class SmallEncoder1_16x_match(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder1_16x_match, self).__init__()
    self.fixed = fixed
    
    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16, 3, 1, 0, dilation=1)
    self.conv11_aux = nn.Conv2d( 16, 64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  # "forward" only outputs the final output
  # "forward_branch" outputs all the middle branch ouputs
  # "forward_aux" outputs all the middle auxiliary mapping layers
  def forward(self, y):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    return y

class SmallEncoder1_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder1_16x_aux, self).__init__()
    self.fixed = fixed
    
    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 24, 3, 1, 0, dilation=1)
    self.conv11_aux = nn.Conv2d( 24, 64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  # "forward" only outputs the final output
  # "forward_branch" outputs all the middle branch ouputs
  # "forward_aux" outputs all the middle auxiliary mapping layers
  def forward(self, y):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    return y

class SmallDecoder1_16x_match(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder1_16x_match, self).__init__()
    self.fixed = fixed

    self.conv11 = nn.Conv2d(16,3,3,1,0, dilation=1)
    self.relu = nn.ReLU(inplace=True)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.relu(self.conv11(self.pad(y)))
    return y


class SmallDecoder1_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder1_16x, self).__init__()
    self.fixed = fixed

    self.conv11 = nn.Conv2d(24,3,3,1,0, dilation=1)
    self.relu = nn.ReLU(inplace=True)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.relu(self.conv11(self.pad(y)))
    return y

class SmallEncoder2_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder2_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0)
    
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    return y

class SmallDecoder2_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder2_16x, self).__init__()
    self.fixed = fixed

    self.conv21 = nn.Conv2d( 32, 16,3,1,0)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y

class SmallEncoder3_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder3_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0)
    self.conv22     = nn.Conv2d( 32, 32,3,1,0)
    self.conv31     = nn.Conv2d( 32, 64,3,1,0)
    
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 64,256,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    return y

class SmallDecoder3_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder3_16x, self).__init__()
    self.fixed = fixed

    self.conv31 = nn.Conv2d( 64, 32,3,1,0)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y
    
class encoder3(nn.Module):
    def __init__(self):
        super(encoder3,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56
    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out,pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out,pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out

class decoder3(nn.Module):
    def __init__(self):
        super(decoder3,self).__init__()
        # decoder
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,128,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(128,128,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(128,64,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(64,64,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        output = {}
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out_relu9 = self.relu9(out)
        out = self.unpool2(out_relu9)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out

class SmallEncoder4_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0, dilation=1)
    self.conv22     = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv31     = nn.Conv2d( 32, 64,3,1,0)
    self.conv32     = nn.Conv2d( 64, 64,3,1,0)
    self.conv33     = nn.Conv2d( 64, 64,3,1,0)
    self.conv34     = nn.Conv2d( 64, 64,3,1,0)
    self.conv41     = nn.Conv2d( 64,128,3,1,0)
    
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 64,256,1,1,0)
    self.conv41_aux = nn.Conv2d(128,512,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    return out41

  def forward_r11output(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    return out11

class SmallDecoder4_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder4_16x, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv34 = nn.Conv2d( 64, 64,3,1,0)
    self.conv33 = nn.Conv2d( 64, 64,3,1,0)
    self.conv32 = nn.Conv2d( 64, 64,3,1,0)
    self.conv31 = nn.Conv2d( 64, 32,3,1,0)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    # print('--------sd_r41-----')
    # print(y.size())
    y = self.relu(self.conv11(self.pad(y)))
    return y

  def forward_sc_from_r11(self, y, colst_r11, alpha):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    ############SkipConnetion############
    print('--------y--------')
    print(y.size())
    print('--------colst_r11------')
    print(colst_r11.size())
    y = alpha * y + (1 - alpha) * colst_r11
    y = self.relu(self.conv11(self.pad(y)))
    return y

  def forward_sc_from_r21(self, y, colst_r21, alpha):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    ############SkipConnetion############
    print('--------y--------')
    print(y.size())
    print('--------colst_r21------')
    print(colst_r21.size())
    y = alpha * y + (1 - alpha) * colst_r21
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    # print('--------sd_r41-----')
    # print(y.size())
    y = self.relu(self.conv11(self.pad(y)))
    return y

class encoder4(nn.Module):
    def __init__(self):
        super(encoder4,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28
    def forward(self,x,sF=None,matrix11=None,matrix21=None,matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11'])

        out = self.conv3(out)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12'])
        out = self.reflecPad4(output['p1'])
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21'])

        out = self.conv5(out)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad6(output['p2'])
        out = self.conv6(out)
        output['r31'] = self.relu6(out)
        if(matrix31 is not None):
            feature3,transmatrix3 = matrix31(output['r31'],sF['r31'])
            out = self.reflecPad7(feature3)
        else:
            out = self.reflecPad7(output['r31'])
        out = self.conv7(out)
        output['r32'] = self.relu7(out)

        out = self.reflecPad8(output['r32'])
        out = self.conv8(out)
        output['r33'] = self.relu8(out)

        out = self.reflecPad9(output['r33'])
        out = self.conv9(out)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad10(output['p3'])
        out = self.conv10(out)
        output['r41'] = self.relu10(out)

        return output

class decoder4(nn.Module):
    def __init__(self):
        super(decoder4,self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out

class SmallDecoder4_16x_Finetune_lastlayer(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder4_16x_Finetune_lastlayer, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv41.requires_grad = False
    self.conv34 = nn.Conv2d( 64, 64,3,1,0)
    self.conv34.requires_grad = False
    self.conv33 = nn.Conv2d( 64, 64,3,1,0)
    self.conv33.requires_grad = False
    self.conv32 = nn.Conv2d( 64, 64,3,1,0)
    self.conv32.requires_grad = False
    self.conv31 = nn.Conv2d( 64, 32,3,1,0)
    self.conv31.requires_grad = False
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv22.requires_grad = False
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv21.requires_grad = False
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv12.requires_grad = False
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.relu.requires_grad = False
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool.requires_grad = False
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.unpool_pwct.requires_grad = False
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    self.pad.requires_grad = False
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  
  def forward(self, y):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    # print('--------sd_r41-----')
    # print(y.size())
    y = self.relu(self.conv11(self.pad(y)))
    return y

  def forward_sc_from_r11(self, y, colst_r11, alpha):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    ############SkipConnetion############
    # print('--------y--------')
    # print(y.size())
    # print('--------colst_r11------')
    # print(colst_r11.size())
    y = alpha * y + (1 - alpha) * colst_r11
    y = self.relu(self.conv11(self.pad(y)))
    return y

  def forward_from_r11_toconstruct(self, content_r11output):
    y = self.relu(self.conv11(self.pad(content_r11output)))
    return y

class SmallEncoder5_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder5_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0, dilation=1)
    self.conv22     = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv31     = nn.Conv2d( 32, 64,3,1,0, dilation=1)
    self.conv32     = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv33     = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv34     = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv41     = nn.Conv2d( 64,128,3,1,0)
    self.conv42     = nn.Conv2d(128,128,3,1,0)
    self.conv43     = nn.Conv2d(128,128,3,1,0)
    self.conv44     = nn.Conv2d(128,128,3,1,0)
    self.conv51     = nn.Conv2d(128,128,3,1,0)

    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 64,256,1,1,0)
    self.conv41_aux = nn.Conv2d(128,512,1,1,0)
    self.conv51_aux = nn.Conv2d(128,512,1,1,0)
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False

  def forward(self, y):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv44(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv51(self.pad(y)))
    return y

class encoder5(nn.Module):
    def __init__(self):
        super(encoder5,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1)) # 左右上下各padding一个像素
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,512,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(512,512,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(512,512,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)

        self.maxPool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(512,512,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self,x,sF=None,contentV256=None,styleV256=None,matrix11=None,matrix21=None,matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11'])

        #out = self.reflecPad3(output['r11'])
        out = self.conv3(out)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12'])
        out = self.reflecPad4(output['p1'])
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21'])

        #out = self.reflecPad5(output['r21'])
        out = self.conv5(out)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad6(output['p2'])
        out = self.conv6(out)
        output['r31'] = self.relu6(out)
        if(styleV256 is not None):
            feature = matrix31(output['r31'],sF['r31'],contentV256,styleV256)
            out = self.reflecPad7(feature)
        else:
            out = self.reflecPad7(output['r31'])
        out = self.conv7(out)
        output['r32'] = self.relu7(out)

        out = self.reflecPad8(output['r32'])
        out = self.conv8(out)
        output['r33'] = self.relu8(out)

        out = self.reflecPad9(output['r33'])
        out = self.conv9(out)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad10(output['p3'])
        out = self.conv10(out)
        output['r41'] = self.relu10(out)

        out = self.reflecPad11(output['r41'])
        out = self.conv11(out)
        output['r42'] = self.relu11(out)

        out = self.reflecPad12(output['r42'])
        out = self.conv12(out)
        output['r43'] = self.relu12(out)

        out = self.reflecPad13(output['r43'])
        out = self.conv13(out)
        output['r44'] = self.relu13(out)

        output['p4'] = self.maxPool4(output['r44'])

        out = self.reflecPad14(output['p4'])
        out = self.conv14(out)
        output['r51'] = self.relu14(out)
        return output

class SmallDecoder5_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder5_16x, self).__init__()
    self.fixed = fixed

    self.conv51 = nn.Conv2d(128,128,3,1,0)
    self.conv44 = nn.Conv2d(128,128,3,1,0)
    self.conv43 = nn.Conv2d(128,128,3,1,0)
    self.conv42 = nn.Conv2d(128,128,3,1,0)
    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv34 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv33 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv32 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv31 = nn.Conv2d( 64, 32,3,1,0, dilation=1)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.relu(self.conv51(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv44(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y))) # self.conv11(self.pad(y))
    return y

class decoder5(nn.Module):
    def __init__(self):
        super(decoder5,self).__init__()

        # decoder
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(512,512,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(256,256,3,1,0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(128,64,3,1,0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # decoder
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out
