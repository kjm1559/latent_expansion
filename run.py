import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

import numpy as np
import sys

if __name__ == '__main__':
    channel = 1
    if sys.argv[1] == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = np.expand_dims(X_train, axis=-1)/255
        X_test = np.expand_dims(X_test, axis=-1)/255
    elif sys.argv[1] == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = np.expand_dims(X_train, axis=-1)/255
        X_test = np.expand_dims(X_test, axis=-1)/255
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        channel = 3
    elif sys.argv[1] == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = np.expand_dims(X_train, axis=-1)/255
        X_test = np.expand_dims(X_test, axis=-1)/255
        
    print(y_train.shape)
    
    train = 'train' == sys.argv[3]
#     log_path = 'adae_triplet3' #adae_triplet, vae, vae_triplet
    log_path = f'{sys.argv[1]}_{sys.argv[2]}'
    print('arg :', log_path)
    # MNIST SOTA : 99.84
    
    if 'adae' in log_path:
        from source.models import latent_expansion
    else:
        from source.models import latent_expansion_variational as latent_expansion
    
    triplet_flag = 'triplet' in log_path
    
    if train:
        for i in range(2):
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=5e-4, first_decay_steps=10, t_mul=1.0, m_mul=0.9, alpha=1e-5)
            ls = tf.keras.callbacks.LearningRateScheduler(cd)
            tb = tf.keras.callbacks.TensorBoard(log_path, histogram_freq=1)
            checkpoint_filepath = log_path + '/checkpoint-weighted-epoch-{epoch:04d}.h5'
            cp = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                save_best_only=True)
            if 'small' in log_path:
                latent_d = 32
            else:
                latent_d = 1024
            model = latent_expansion(triplet_flag=triplet_flag, latent_d=latent_d, channel=channel)
            model.compile()
            
            # for test
#             print('y_daata', y_train[:128])
#             model.train_step([X_train[:128], y_train[:128]])
#             exit(-1)
            
            model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es, tb, cp, ls])
            model.save_weights(log_path + '/latent_expansion_model.h5')
            exit(0)
    else:
        if 'small' in log_path:
            latent_d = 32
        else:
            latent_d = 1024
        model = latent_expansion(latent_d=latent_d, channel=channel)
        model.built = True
        model.compile()
#         model.load_weights(log_path + '/checkpoint-weighted-epoch-0095.h5')
        model.load_weights(log_path + '/latent_expansion_model.h5')
#         model.evaluate(X_test, y_test)
#         y_pred = model.predict(X_test, batch_size=128)
        y_pred, latent = model.predict(X_test, batch_size=128)
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
        latent_pca = pca.fit_transform(latent)
        # reshape
        # y_test = y_test.reshape(-1)
        
        for i in range(10):
            plt.scatter(latent_pca[y_test == i, 0], latent_pca[y_test == i, 1], label=i, s=1)
        plt.legend()
        plt.title(f'{accuracy_score(y_test, np.argmax(y_pred, axis=-1))}')
        plt.savefig(log_path + '_latent_space.png')

        print(classification_report(y_test, np.argmax(y_pred, axis=-1)))
        print(accuracy_score(y_test, np.argmax(y_pred, axis=-1)))
        