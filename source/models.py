import tensorflow as tf
# tf.config.run_functions_eagerly(True)
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, Activation, Lambda
import numpy as np

class conv_pooling(tf.keras.layers.Layer):
    def __init__(self, fiter1, fiter2):
        super(conv_pooling, self).__init__()
        self.conv1 = Conv2D(fiter1, 
                            kernel_size=(5, 5), 
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                           )
        self.conv2 = Conv2D(fiter2, 
                            kernel_size=(3, 3), 
                            strides=(1, 1),
                            padding='same',
                           )
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.pooling = MaxPooling2D(pool_size=(2, 2))
        self.do = Dropout(0.2)
    def call(self, x):
        return self.do(self.pooling(self.act(self.bn(self.conv2(self.conv1(x))))))
            

class latent_expantion_adae(tf.keras.layers.Layer):
    def __init__(self, latent_d=1024):
        super(latent_expantion_adae, self).__init__()
        self.en1 = conv_pooling(64, 128)
        self.en2 = conv_pooling(256, latent_d)
        self.gap = GlobalAveragePooling2D()
        self.cls = Dense(10, activation='softmax')
#         self.latent_flatten = Flatten()
        
    def call(self, x):
        latent = self.gap(self.en2(self.en1(x)))
        # y = self.conv2(latent)
        y = self.cls(latent)
        return y, latent
    
class simple_discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(simple_discriminator, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.softmax = Dense(2, activation='softmax')
        
    def call(self, x):
        # simple classifier
        y = self.softmax(self.dense1(x))
        return y
    
class latent_expantion_vae(tf.keras.layers.Layer):
    def __init__(self, latent_d=1024):
        super(latent_expantion_vae, self).__init__()
        self.en1 = conv_pooling(64, 128)
        self.en2 = conv_pooling(256, 1024)
        self.gap = GlobalAveragePooling2D()
        self.latent_mean = Dense(latent_d, activation='linear')
        self.latent_var = Dense(latent_d, activation='linear')
        self.cls = Dense(10, activation='softmax')
#         self.latent_flatten = Flatten()
        
    def call(self, x):
        latent = self.gap(self.en2(self.en2(x)))
        # y = self.conv2(latent)
        z_mean = self.latent_mean(latent)
        z_log_var = self.latent_var(latent)
        z = Lambda(sampling, output_shape=(latent.shape[-1],), name='z')([z_mean, z_log_var])     
        y = self.cls(z)
        return y, z_mean

        
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args 
    batch = K.shape(z_mean)[0]
    # batch = z_mean.shape[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
        

# discriminator : it makes latent vector to normal distribution.
# triplet selector : it makes farther distance between other class.
class latent_expansion(tf.keras.Model):
    def __init__(self, triplet_flag=True, name='triplet', channel=1, latent_d=1024, **kwargs):
        super(latent_expansion, self).__init__(name, **kwargs)
        self.gamma = 1
        self.triplet_flag = triplet_flag
        self.channel = channel
        self.latent_d = latent_d
        self.__set_model()
        self.m = tf.keras.metrics.Accuracy()
        
    
    def __set_model(self):
        self.ae = latent_expantion_adae(self.latent_d)
        self.dis = simple_discriminator()
        inputs = tf.keras.Input(shape=(None, None, self.channel), name='image')
        inputs_d = tf.keras.Input(shape=(self.latent_d), name='latent')
        pred_y, latent = self.ae(inputs)

        dis_r = self.dis(inputs_d)
        self.ae_model = tf.keras.Model(inputs, [pred_y, latent])
        self.dis_model = tf.keras.Model(inputs_d, dis_r)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.dis_optimizer = tf.keras.optimizers.Adam(1e-3)

    def __cosine_distance(self, a, b):
            return 1 - K.sum(a * b) / (K.sqrt(K.sum(K.square(a))) * K.sqrt(K.sum(K.square(b))))
    
    def _tf_triplet_selector(self, z, y, training=False):  
        
#         print(tf.squeeze(tf.gather(z, tf.cast(tf.map_fn(lambda x : np.random.choice(np.where(y == x)[0], 1), y), tf.int32))))

#         y = tf.cast(y, tf.int32)
#         print(y.shape)
#         print(tf.gather(z, tf.squeeze(tf.map_fn(lambda x : tf.random.shuffle(tf.where(y != x))[0], y, fn_output_signature=tf.int64))).shape)
#         print(tf.cast((lambda x : tf.random.categorical(tf.where(y != x), 1))(y), tf.int32))
        
#         in_class = tf.gather(z, tf.squeeze(tf.map_fn(lambda x : tf.random.shuffle(tf.where(y == x))[0], y, fn_output_signature=tf.int64, parallel_iterations=56)))
#         out_class = tf.gather(z, tf.squeeze(tf.map_fn(lambda x : tf.random.shuffle(tf.where(y != x))[0], y, fn_output_signature=tf.int64, parallel_iterations=56)))
        
#         for i in range(y.shape[0]):
#             in_class_index = tf.random.shuffle(tf.where(y == label))

#         in_class_index = tf.map_fn((lambda x : tf.random.shuffle(tf.where(y == x))[0]), y, fn_output_signature=tf.int64)
#         out_class_index = tf.map_fn((lambda x : tf.random.shuffle(tf.where(y != x))[0]), y, fn_output_signature=tf.int64)

        def my_numpy_func(y_np):
            # x will be a numpy array with the contents of the input to the
            # tf.function
            return np.array([np.random.choice(np.squeeze(np.where(y_np == x)), 1) for x in y_np], dtype='int32')
        @tf.function(input_signature=[tf.TensorSpec(None, tf.int32)])
        def tf_function(y_np):
            return tf.numpy_function(my_numpy_func, [y_np], tf.int32)
        
        def my_numpy_func2(y_np):
            # x will be a numpy array with the contents of the input to the
            # tf.function
            return np.array([np.random.choice(np.squeeze(np.where(y_np != x)), 1) for x in y_np], dtype='int32')
        @tf.function(input_signature=[tf.TensorSpec(None, tf.int32)])
        def tf_function2(y_np):
            return tf.numpy_function(my_numpy_func2, [y_np], tf.int32)
          

#         if y.shape[0] == None:
#             return 0
#         try:
#             y_np = y.numpy()
#         except:
#             return 0
#         print(np.where(y_np != x))
        in_class_index = tf_function(tf.cast(y, tf.int32))#[np.random.choice(np.squeeze(np.where(y_np == x)), 1) for x in y_np]
        out_class_index = tf_function2(tf.cast(y, tf.int32))#[np.random.choice(np.squeeze(np.where(y_np != x)), 1) for x in y_np]
        
        
        
        print(in_class_index, out_class_index)

                
        in_class = tf.gather(z, in_class_index)
        out_class = tf.gather(z, out_class_index)
        
        return K.mean(self.__cosine_distance(z, in_class) - self.__cosine_distance(z, out_class))
#         print('latent shape :', z.shape)
#         z_u = tf.unstack(z)
#         y_u = tf.unstack(y)
#         result = 0
#         for i in range(len(y_u)):
#             in_class_samples = tf.gather(z, tf.random.shuffle(tf.where(y == y_u[i])))
#             out_class_samples = tf.gather(z, tf.random.shuffle(tf.where(y != y_u[i])))
#             result = (self.__cosine_distance(z_u, in_class_samples[0]) - self.__cosine_distance(z_u, out_class_samples[0]))/y.shape[0]
#         return K.mean(tf.stack(result))

    def call(self, x):
        return self.ae_model(x)

    def train_step(self, data):
        # discrinator added plz
        x, y = data
        # train for autoencoder
        with tf.GradientTape() as tape:
            y_pred, latent = self.ae_model(x, training=True)
            cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, y_pred)
#             print('latent :', latent.shape)
            triplet_loss = self._tf_triplet_selector(latent, y, training=True) + self.gamma

            adversarial_loss = -K.mean(K.log(self.dis_model(latent, training=False) + 1e-8))
            
            if self.triplet_flag:
                loss = cls_loss + adversarial_loss + 0.3 * triplet_loss
            else:
                loss = cls_loss + adversarial_loss
            acc = self.m(K.argmax(y_pred, axis=-1), y)#K.sum(K.switch(K.argmax(y_pred, axis=-1) == tf.cast(y, dtype='int64'),  K.ones_like(y),  K.zeros_like(y)))/K.sum(K.ones_like(y))
            
        
        trainable_vars = self.ae_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # train for discriminator
        with tf.GradientTape() as tape2:
            # make normal noise by class data
            @tf.function
            def __make_normal_noise(latent, y):
                if y.shape[0] == None:
                    return []
                y_u = tf.unstack(y)
                noise = []
#                 print('make latent shape:', latent.shape)
                for i in range(len(y_u)):
                    sample_latent = tf.gather(latent, tf.where(y == y_u[i]))[:, 0, :]
                    mu, std = K.mean(sample_latent, axis=0), K.std(sample_latent, axis=0)
#                     print('statistic :', mu.shape, std.shape)
#                     print('each data shape :', tf.random.normal(sample_latent.shape[1:], mean=mu, stddev=std).shape, sample_latent.shape)
                    noise.append(tf.random.normal(sample_latent.shape[1:], mean=mu, stddev=std))
                noise = tf.stack(noise)
#                 print('noise shape :', noise.shape)
                samples = K.concatenate([noise, latent], axis=0)
                
                return samples
        
            def my_numpy_func(z_np, y_np):
                # x will be a numpy array with the contents of the input to the
                # tf.function
                means = np.array([np.mean(z_np[np.where(y_np == x)], axis=0) for x in y_np])
                stds = np.array([np.std(z_np[np.where(y_np == x)], axis=0) for x in y_np])
                random_noise = np.squeeze(np.array([np.random.normal(means[i], stds[i], (1, means.shape[-1])) for i in range(len(means))], dtype='float32'))
                return random_noise
            @tf.function(input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.int32)])
            def tf_function(z_np, y_np):
#                 print(tf.numpy_function(my_numpy_func, [y_np], tf.int32))
                return tf.numpy_function(my_numpy_func, [z_np, y_np], tf.float32)
            
            y_pred, latent = self.ae_model(x, training=False)
#             samples = __make_normal_noise(latent, y)
#             print('latent shape :', latent.shape)
            noise = tf_function(latent, tf.cast(y, tf.int32))
            samples = tf.concat([noise, latent], axis=0)
#             _, _ = K.zeros(latent.shape[0]), K.ones(latent.shape[0])
            labels = K.concatenate([K.zeros_like(y), K.ones_like(y)], axis=0)
            pred_d = self.dis_model(samples)
#             print('labels shape', labels.shape)
#             
            loss_d = tf.keras.losses.SparseCategoricalCrossentropy()(labels, pred_d)

        trainable_vars_d = self.dis_model.trainable_variables
        gradients_d = tape2.gradient(loss_d, trainable_vars_d)
        
        self.dis_optimizer.apply_gradients(zip(gradients_d, trainable_vars_d))
        
        return {'cls_loss': cls_loss, 'triplet_loss': triplet_loss, 'adv_loss': adversarial_loss, 'loss_d': loss_d, 'loss': loss, 'acc': acc}
    
    def test_step(self, data):
        x, y = data
        
        y_pred, latent = self.ae_model(x)
        cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, y_pred)
        # calculate acc
        acc = self.m(K.argmax(y_pred, axis=-1), y)#K.sum(K.switch(K.argmax(y_pred, axis=-1) == tf.cast(y, dtype='float32'),  K.ones_like(y),  K.zeros_like(y)))/K.sum(K.ones_like(y))
            
        return {'loss': cls_loss, 'acc': acc}
    
    
class latent_expansion_variational(latent_expansion):
    def __init__(self, triplet_flag=True, name='triplet', channel=1, latent_d=1024, **kwargs):
        super(latent_expansion_variational, self).__init__(triplet_flag, name, channel, latent_d, **kwargs)
    
    def __set_model(self):
        print('vae model !!', self.ae)
        self.ae = latent_expantion_vae(self.latent_d)
        inputs = tf.keras.Input(shape=(None, None, self.channel), name='image')
        pred_y, latent = self.ae(inputs)

        self.ae_model = tf.keras.Model(inputs, [pred_y, latent])
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        
    def train_step(self, data):
        # discrinator added plz
        x, y = data
        # train for autoencoder
        with tf.GradientTape() as tape:
            y_pred, latent = self.ae_model(x, training=True)
            cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, y_pred)
            
            if self.triplet_flag:
                triplet_loss = self._tf_triplet_selector(latent, y, training=True) + self.gamma
                loss = cls_loss + 0.3 * triplet_loss
            else:
                triplet_loss = 0
                loss = cls_loss
                
            acc = self.m(K.argmax(y_pred, axis=-1), y)#K.sum(K.switch(K.argmax(y_pred, axis=-1) == tf.cast(y, dtype='int64'),  K.ones_like(y, dtype='int32'),  K.zeros_like(y, dtype='int32')))/K.sum(K.ones_like(y, dtype='int32'))
            
        
        trainable_vars = self.ae_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {'cls_loss': cls_loss, 'triplet_loss': triplet_loss, 'loss': loss, 'acc': acc}