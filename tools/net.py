import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras
from ops import *

def generator(gen_inputs, num_resblock = 1, gen_output_channels=3, reuse=False, batch_norm = True, is_training =True, printinfo = True):
    gen_inputs_shape = gen_inputs.get_shape()
    if printinfo:
        print("generator:")
        print("input shape:%s"%str(gen_inputs_shape))
    weights = 0
    # The Bx residual blocks
    def residual_block(inputs, output_channel = 64, stride = 1, scope = 'res_block', printinfo = True,is_training = True):
        with tf.variable_scope(scope):
            weight_num = 0
            kernel_size = 7
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=True, scope='conv_1', printinfo = printinfo)
            weight_num += kernel_size*kernel_size*inputs.get_shape()[-1]*output_channel + output_channel
            # net = tf.nn.relu(net)
            net = lrelu(net,0.2)
            if batch_norm:
                net = batchnorm(net,is_training = is_training)
            kernel_size = 7
            net = conv2(net, kernel_size, output_channel, stride, use_bias=True, scope='conv_2', printinfo = printinfo)
            # net = tf.nn.relu(net)
            net = lrelu(net,0.2)
            if batch_norm:
                net = batchnorm(net,is_training = is_training)
            weight_num += kernel_size*kernel_size*output_channel*output_channel + output_channel
            net = net + inputs
        return net, weight_num

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            kernelsize = 7
            outputchannel = 128
            net = conv2(gen_inputs, kernelsize, outputchannel, 1, scope='conv', printinfo = printinfo)
            weights += kernelsize*kernelsize*gen_inputs_shape[-1]*outputchannel + outputchannel
            # net = tf.nn.relu(net)
            net = lrelu(net,0.2)
            if batch_norm:
                net = batchnorm(net,is_training = is_training)
            # net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        for i in range(0, num_resblock , 1): # should be 16 for TecoGAN, and 10 for TecoGANmini
            name_scope = 'resblock_%d'%(i)
            net, weight_num = residual_block(net, 128, 1, name_scope, printinfo = printinfo,is_training = is_training)
            weights += weight_num

        with tf.variable_scope('output_stage'):
            inputchannel = outputchannel
            kernelsize = 7
            outputchannel = gen_output_channels
            net = conv2(net, kernelsize, outputchannel, 1, scope='conv9', printinfo = printinfo)
            net_I = net[:,:,:,:25]
            net_UV = net[:,:,:,25:]
            net_UV = tf.tanh(net_UV)
            weights += kernelsize*kernelsize*inputchannel*outputchannel + outputchannel
            input_image = gen_inputs[:,:,:,-2:] # ignore warped pre high res
            net_UV = net_UV + input_image
            net = tf.concat([net_I,net_UV],-1)
    if printinfo:
        print("total weights of generator:%d"%weights)
    return net


def discriminator(dis_inputs, name_scope='discriminator',reuse=False, printinfo = True):
    dis_inputs_shape = dis_inputs.get_shape()
    if printinfo:
        print(name_scope,":")
        print("input shape:%s"%str(dis_inputs_shape))
    weights = 0
    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope,reuse=False, printinfo = True):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1', printinfo = printinfo)
            net = batchnorm(net, is_training=True)
            net = tf.nn.relu(net)

        return net

    layer_list = []
    with tf.variable_scope(name_scope+'_unit',reuse=reuse):
    #with tf.variable_scope('discriminator_unit',reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            # no batchnorm for the first layer
            kernelsize = 4
            outputchannel = 128
            net = conv2(dis_inputs, kernelsize, outputchannel, 1, scope='conv', printinfo = printinfo)
            weights += kernelsize*kernelsize*dis_inputs_shape[-1]*outputchannel + outputchannel
            net = tf.nn.relu(net) # (b, h,w,128)


            # The discriminator block part
            # block 1
            # net = dis_inputs
            inputchannel = outputchannel
            kernelsize = 4
            outputchannel = 128
            net = discriminator_block(net, outputchannel, kernelsize, 2, 'disblock_1', printinfo = printinfo)
            weights += kernelsize*kernelsize*inputchannel*outputchannel
            layer_list += [net] # (b, h/2,w/2,128)

            # block 2
            inputchannel = outputchannel
            kernelsize = 4
            outputchannel = 128
            net = discriminator_block(net, outputchannel, kernelsize, 2, 'disblock_3', printinfo = printinfo)
            weights += kernelsize*kernelsize*inputchannel*outputchannel
            layer_list += [net] # (b, h/4,w/4,128)

            #with tf.device('/gpu:1'), tf.variable_scope('discriminator_unit2',reuse=reuse):

            # block 3
            inputchannel = outputchannel
            kernelsize = 4
            outputchannel = 128
            net = discriminator_block(net, outputchannel, kernelsize, 2, 'disblock_5', printinfo = printinfo)
            weights += kernelsize*kernelsize*inputchannel*outputchannel
            layer_list += [net] # (b, h/8,w/8,128)

            # block_4
            inputchannel = outputchannel
            kernelsize = 4
            outputchannel = 128
            net = discriminator_block(net, outputchannel, kernelsize, 2, 'disblock_7', printinfo = printinfo)
            weights += kernelsize*kernelsize*inputchannel*outputchannel
            layer_list += [net]  # (b, h/16,w/16,256)

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            net = denselayer(net, 1, printinfo = printinfo) # channel-wise dense layer
            netshape = net.get_shape()
            weights += netshape[1]*netshape[2]*netshape[3] + 1
            net = tf.nn.sigmoid(net) # (h/16,w/16,1)
    if printinfo:
        print("total weights of discriminator: %d"%weights)
    return net, layer_list
