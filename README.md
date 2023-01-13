latent expansion
===

### Architecture
1. network structure
    - Adversarial network
    - Simple convolutional network with variational latent space
2. loss function
    - with triplet loss
    - without triplet loss
3. latent space dimention
    - large -> 1024
    - small -> 32

### Result
 - dataset : CIFAR10 

|latent space|adae|vae|
|-|-|-|
|1024|![adae big](./figures/cifar10_adae_latent_space.png)|![vae big](./figures/cifar10_vae_only_latent_space.png)|
|32|![adae small](./figures/cifar10_adae_small_latent_space.png)|![vae small](./figures/cifar10_vae_small_latent_space.png)|
|1024 triplet|![adae big triplet](./figures/cifar10_adae_triplet_latent_space.png)|![vae big triplet](./figures/cifar10_vae_triplet_latent_space.png)|
|32 triplet|![adae small triplet](./figures/cifar10_adae_samll_triplet_latent_space.png)|![vae small triplet](./figures/cifar10_vae_small_triplet_latent_space.png)|

- Performance is better if the latent space has a higher dimension than the previous layer.  
- Triplet loss affects performance degradation.  
- The variational layers improve model performance.

### Enviroment
tensorflow = 2.10