# BE_GAN

My re-implementation of the BEGAN by Berthelot et al. with some modifications (for Generator, I use 10 Res blocks instead, and for Discriminator, I use 4 Res blocks instead).

Example on how to use weight_decay -> 0.0005 to each layer.

Example on how to use progbar to report results.

For details, please refer to their original paper, https://arxiv.org/pdf/1703.10717.pdf

The code is based on Keras using TensorFlow backend. Modifications and refinements in terms of the code are highly welcome.
