# -*- coding: utf-8 -*-
# @Time    : 18-4-19
# @Author  : Xinge
from __future__ import division
from .common_net import *


# In COCOResGen2, all the non residual-block layers are based on LeakyReLU with no normalization layers.
class COCOResGen2(nn.Module):
  def __init__(self, params):
    super(COCOResGen2, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    # encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    # encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      # encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # encB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    # decA = []
    decB = []
    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      # decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # decA += [PixelNormLayer()]
      decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # decB += [PixelNormLayer()]
    # Convolutional back-end
    for i in range(0, n_gen_front_blk-1):
      # decA += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      # decA += [PixelNormLayer()]
      decB += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      # decB += [PixelNormLayer()]
      tch = tch//2

    # decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]

    decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]

    # decA += [nn.Tanh()]
    # decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    # self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    # self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)


  def forward(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out

  def forward_a2s(self, x_A):
      out = self.encode_A(x_A)
      shared = self.enc_shared(out)
      return shared

  def forward_s2b(self, x_B):
      out = self.dec_shared(x_B)
      outputs = self.decode_B(out)
      return outputs


class COCOResGen3(nn.Module):
  def __init__(self, params):
    super(COCOResGen3, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      encB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    decA = []
    decB = []

    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # decA += [PixelNormLayer()]
      decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # decB += [PixelNormLayer()]
    # Convolutional back-end
    for i in range(0, n_gen_front_blk-1):
      decA += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
      # decA += [PixelNormLayer()]
      decB += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
      # decB += [PixelNormLayer()]
      tch = tch//2
    decA += [nn.Conv2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    decA += [nn.LeakyReLU(inplace=True)]
    decB += [nn.Conv2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decB += [nn.LeakyReLU(inplace=True)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared


class GANGen_simple(nn.Module):
  def __init__(self, params):
    super(GANGen_simple, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    # n_gen_shared_blk = params['n_gen_shared_blk']
    # n_gen_res_blk    = params['n_gen_res_blk']
    # n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      encB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # enc_shared += [GaussianNoiseLayer()]
    # dec_shared = []
    # for i in range(0, n_gen_shared_blk):
    #   dec_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # # END of SHARED LAYERS

    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)


  def forward(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    # out = self.dec_shared(shared)
    # out = self.decode_B(out)
    return shared



class GAN_decoder(nn.Module):
  def __init__(self, params):
    super(GAN_decoder, self).__init__()
    # input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch'] # 512
    # n_enc_front_blk  = params['n_enc_front_blk']
    # n_enc_res_blk    = params['n_enc_res_blk']
    # n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    tch = ch
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    # decA = []
    decB = []

    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      # decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # decA += [PixelNormLayer()]
      decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      # decB += [PixelNormLayer()]
    # Convolutional back-end
    for i in range(0, n_gen_front_blk-1):
      # decA += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
      # decA += [PixelNormLayer()]
      decB += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
      # decB += [PixelNormLayer()]
      tch = tch//2
    # decA += [nn.Conv2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    # decA += [nn.LeakyReLU(inplace=True)]
    decB += [nn.Conv2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decB += [nn.LeakyReLU(inplace=True)]
    # decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    # self.encode_A = nn.Sequential(*encA)
    # self.encode_B = nn.Sequential(*encB)
    # self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    # self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)



  def forward(self, x_A):
    # out = self.encode_A(x_A)
    # shared = self.enc_shared(out)
    out = self.dec_shared(x_A)
    out = self.decode_B(out)
    return out
  #



