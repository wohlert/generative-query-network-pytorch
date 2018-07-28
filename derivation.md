# Lower bound derivation

We want to make a generative model $g(x|y)$, where $y$ is a set of known conditional parameters. If we parametrise the genrative density as a model with parameters $\theta$, it can be rewritten as:

$$g_{\theta}(x|y) = \int g_{\theta}(x, z|y) \ dz = \int g_{\theta}(x|z, y) g_{\theta}(z|y) \ dz = \int g_{\theta}(x|z, y) \pi_{\theta}(z|y) \ dz$$

Where we would like to call $\pi_{\theta}$ a conditional prior that shares parameters with the generative model. Because of the integral, this density is intractable, we introduce a variational approximation density $q_{\phi}(z|x, y)$, which this time is parameterised by $\phi$. wlog.

$$\log g_{\theta}(x|y) = \log \bigg [ \int q_{\phi}(z|x,y) g_{\theta}(x|z, y) \frac{\pi_{\theta}(z|y)}{q_{\phi}(z|x,y)} \ dz \bigg ]$$

Using Jensen's inequality.

$$\log g_{\theta}(x|y) \geq \int q_{\phi}(z|x,y) \log \bigg [ g_{\theta}(x|z, y) \frac{\pi_{\theta}(z|y)}{q_{\phi}(z|x,y)} \bigg ] \ dz$$

By linearity of integrals

$$\log g_{\theta}(x|y) \geq \int q_{\phi}(z|x,y) \log  g_{\theta}(x|z, y) \ dz + \int q_{\phi}(z|x,y) \frac{\pi_{\theta}(z|y)}{q_{\phi}(z|x,y)}  \ dz$$



$$\log g_{\theta}(x|y) \geq \mathbb{E}_{q_{\phi}(z|x,y)} \bigg [ \log  g_{\theta}(x|z, y) \bigg ]- D_{KL}(q_{\phi}(z|x,y)||\pi_{\theta}(z|y))$$

Define the evidence lower bound (ELBO) by $\mathcal{L}(\theta) = - \sum_{i} \log g_{\theta}(x_i|y_i)$, we find that.

$$\mathcal{L}(\theta) = -\sum_{i} \mathbb{E}_{q_{\phi}(z_i|x_i,y_i)} \bigg [ \log  g_{\theta}(x_i|z_i, y_i) \bigg ] + \sum_{i} D_{KL}(q_{\phi}(z_i|x_i,y_i)||\pi_{\theta}(z_i|y_i))$$

So for each datapoint $x_i$, we simply need to conditionally sample a $z_i$ from the variational distribution given the variables $y_i$ and estimate the ELBO. This is the Monte Carlo approach to estimating expectation. Note: In this model, $y = (r, v)$ is the viewpoint and representation.