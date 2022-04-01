"""
Modified from
https://github.com/johnhw/pfilter/tree/62fc8b735518c8427057764183f3ea1109053b5b
"""
import numpy as np
import torch
from einops import rearrange
from scipy.stats import norm


# return a new function that has the heat kernel (given by delta) applied.
def make_heat_adjusted(sigma):
    def heat_distance(d):
        return np.exp(-(d**2) / (2.0 * sigma**2))

    return heat_distance


## Resampling based on the examples at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
## originally by Roger Labbe, under an MIT License
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n
    return create_indices(positions, weights)


def stratified_resample(weights):
    n = len(weights)
    positions = (np.random.uniform(0, 1, n) + np.arange(n)) / n
    return create_indices(positions, weights)


def residual_resample(weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * weights).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
    return indices


def create_indices(positions, weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


### end rlabbe's resampling functions


def multinomial_resample(weights):
    return np.random.choice(np.arange(len(weights)), p=weights, size=len(weights))


# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


# identity function for clearer naming
identity = lambda x: x


def squared_error(x, y, sigma=1):
    """
    RBF kernel, supporting masked values in the observation
    Parameters:
    -----------
    x : array (N,D) array of values
    y : array (N,D) array of values

    Returns:
    -------

    distance : scalar
        Total similarity, using equation:

            d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))

        summed over all samples. Supports masked arrays.
    """
    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    return np.exp(-d / (2.0 * sigma**2))


def gaussian_noise(x, sigmas):
    """Apply diagonal covaraiance normally-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x + n


def cauchy_noise(x, sigmas):
    """Apply diagonal covaraiance Cauchy-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = np.random.standard_cauchy(size=(x.shape[0], len(sigmas))) * np.array(sigmas)
    return x + n


def independent_sample(fn_list):
    """Take a list of functions that each draw n samples from a distribution
    and concatenate the result into an n, d matrix
    Parameters:
    -----------
        fn_list: list of functions
                A list of functions of the form `sample(n)` that will take n samples
                from a distribution.
    Returns:
    -------
        sample_fn: a function that will sample from all of the functions and concatenate
        them
    """

    def sample_fn(n):
        return np.stack([fn(n) for fn in fn_list]).T

    return sample_fn


def convert_image_np2torch(image, size=256):
    """Converts an array of images from numpy to pytorch after normalization.
    Parameters:
    -----------
        image: (B, H, W, 3) numpy array
        size: rescale largest dimension of image to this size while maintaining aspect ratio

    Returns:
    --------
        image: (B, C, H, W) torch tensor after channelwise normalization
    """
    mean = torch.Tensor([[[[0.485, 0.456, 0.406]]]])
    std = torch.Tensor([[[[0.229, 0.224, 0.225]]]])
    image = torch.from_numpy(image).float() / 255.0
    image = (image - mean) / std
    image = rearrange(image, "b h w c -> b c h w")
    image_height, image_width = image.shape[2:]
    if image_width > image_height:
        width = size
        height = max(1, int(float(image_height) * width / image_width))
    else:
        height = size
        width = max(1, int(float(image_width) * height / image_height))
    image = torch.nn.functional.interpolate(
        image, size=(height, width), mode="bilinear", align_corners=False
    )
    return image


class ParticleFilter(object):
    """A particle filter object which maintains the internal state of a population of particles, and can
    be updated given observations.

    Attributes:
    -----------

    n_particles : int
        number of particles used (N)
    d : int
        dimension of the internal state
    resample_proportion : float
        fraction of particles resampled from prior at each step
    particles : array
        (N,D) array of particle states
    original_particles : array
        (N,D) array of particle states *before* any random resampling replenishment
        This should be used for any computation on the previous time step (e.g. computing
        expected values, etc.)
    mean_hypothesis : array
        The current mean hypothesized observation
    mean_state : array
        The current mean hypothesized internal state D
    map_hypothesis:
        The current most likely hypothesized observation
    map_state:
        The current most likely hypothesized state
    n_eff:
        Normalized effective sample size, in range 0.0 -> 1.0
    weight_entropy:
        Entropy of the weight distribution (in nats)
    hypotheses : array
        The (N,...) array of hypotheses for each particle
    weights : array
        N-element vector of normalized weights for each particle.
    """

    def __init__(
        self,
        prior_fn,
        init_template,
        observe_fn=None,
        resample_fn=None,
        n_particles=200,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=None,
        similarity_net=None,
        use_deep_similarity=None,
        device=None,
        resample_proportion=None,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=None,
        n_eff_threshold=1.0,
    ):
        """

        Parameters:
        -----------

        prior_fn : function(n) = > states
                a function that generates N samples from the prior over internal states, as
                an (N,D) particle array
        observe_fn : function(states) => observations
                    transformation function from the internal state to the sensor state. Takes an (N,D) array of states
                    and returns the expected sensor output as an array (e.g. a (N,W,H) tensor if generating W,H dimension images).
        resample_fn: A resampling function weights (N,) => indices (N,)
        n_particles : int
                     number of particles in the filter
        dynamics_fn : function(states) => states
                      dynamics function, which takes an (N,D) state array and returns a new one with the dynamics applied.
        noise_fn : function(states) => states
                    noise function, takes a state vector and returns a new one with noise added.
        weight_fn :  function(hypothesized, real) => weights
                    computes the distance from the real sensed variable and that returned by observe_fn. Takes
                    a an array of N hypothesised sensor outputs (e.g. array of dimension (N,W,H)) and the observed output (e.g. array of dimension (W,H)) and
                    returns a strictly positive weight for the each hypothesis as an N-element vector.
                    This should be a *similarity* measure, with higher values meaning more similar, for example from an RBF kernel.
        internal_weight_fn :  function(states, observed) => weights
                    Reweights the particles based on their *internal* state. This is function which takes
                    an (N,D) array of internal states and the observation and
                    returns a strictly positive weight for the each state as an N-element vector.
                    Typically used to force particles inside of bounds, etc.
        transform_fn: function(states, weights) => transformed_states
                    Applied at the very end of the update step, if specified. Updates the attribute
                    `transformed_particles`. Useful when the particle state needs to be projected
                    into a different space.
        resample_proportion : float
                    proportion of samples to draw from the initial on each iteration.
        n_eff_threshold=1.0: float
                    effective sample size at which resampling will be performed (0.0->1.0). Values
                    <1.0 will allow samples to propagate without the resampling step until
                    the effective sample size (n_eff) drops below the specified threshold.
        column_names : list of strings
                    names of each the columns of the state vector

        """
        self.resample_fn = resample_fn or resample
        self.column_names = column_names
        self.prior_fn = prior_fn
        self.n_particles = n_particles
        self.init_filter()
        self.n_eff_threshold = n_eff_threshold
        self.d = self.particles.shape[1]
        self.observe_fn = observe_fn or identity
        self.dynamics_fn = dynamics_fn or identity
        self.noise_fn = noise_fn or identity
        self.weight_fn = weight_fn or squared_error
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.transform_fn = transform_fn
        self.transformed_particles = None
        self.resample_proportion = resample_proportion or 0.0
        self.internal_weight_fn = internal_weight_fn
        self.original_particles = np.array(self.particles)
        self.template = init_template.astype(np.float)
        self.similarity_net = similarity_net
        self.use_deep_similarity = use_deep_similarity
        self.device = device
        self.alpha = 0.001

    def init_filter(self, mask=None):
        """Initialise the filter by drawing samples from the prior.

        Parameters:
        -----------
        mask : array, optional
            boolean mask specifying the elements of the particle array to draw from the prior. None (default)
            implies all particles will be resampled (i.e. a complete reset)
        """
        new_sample = self.prior_fn(self.n_particles)

        # resample from the prior
        if mask is None:
            self.particles = new_sample
        else:
            self.particles[mask, :] = new_sample[mask, :]

    def update(self, observed=None, **kwargs):
        """Update the state of the particle filter given an observation.

        Parameters:
        ----------

        observed: array
            The observed output, in the same format as observe_fn() will produce. This is typically the
            input from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped, and the filter will run one step in prediction-only mode.

        kwargs: any keyword arguments specified will be passed on to:
            observe_fn(y, **kwargs)
            weight_fn(x, **kwargs)
            dynamics_fn(x, **kwargs)
            noise_fn(x, **kwargs)
            internal_weight_function(x, y, **kwargs)
            transform_fn(x, **kwargs)
        """

        # apply dynamics and noise
        self.particles = self.noise_fn(
            self.dynamics_fn(self.particles, **kwargs), **kwargs
        )

        # hypothesise observations
        self.hypotheses = self.observe_fn(
            self.particles, observed, self.template.shape[:2], **kwargs
        )

        zero_hypothesis = (self.hypotheses.shape[0] == 0) or (
            self.hypotheses.shape[1] == 0
        )
        zero_template = (self.template.shape[0] == 0) or (self.template.shape[1] == 0)
        if (observed is not None) and (not zero_hypothesis) and (not zero_template):
            # compute similarity to observations
            # force to be positive
            # compute feature embedddings
            if self.use_deep_similarity:
                hypotheses_t = convert_image_np2torch(self.hypotheses).to(self.device)
                template_t = convert_image_np2torch(self.template[np.newaxis, ...]).to(
                    self.device
                )
                with torch.no_grad():
                    # For memory efficiency
                    hypotheses_f = []
                    B = 64
                    for i in range(0, hypotheses_t.shape[0], B):
                        hypotheses_f.append(
                            self.similarity_net(hypotheses_t[i : (i + B)])
                        )
                    hypotheses_f = torch.cat(hypotheses_f, dim=0)
                    template_f = self.similarity_net(template_t)
                del hypotheses_t
                del template_t
            else:
                hypotheses_f = self.hypotheses.astype(np.float)
                template_f = self.template.astype(np.float)

            self.similarities = np.array(
                self.weight_fn(hypotheses_f, template_f, **kwargs)
            )
            if self.use_deep_similarity:
                del hypotheses_f
                del template_f
            weights = np.clip(
                np.multiply(self.weights, self.similarities),
                0.0000001,
                np.inf,
            )
        else:
            # we have no observation, so all particles weighted the same
            self.similarities = np.ones((self.n_particles,))
            weights = np.multiply(self.weights, self.similarities)

        self.scores = weights

        # apply weighting based on the internal state
        # most filters don't use this, but can be a useful way of combining
        # forward and inverse models
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(
                self.particles, observed, **kwargs
            )
            internal_weights = np.clip(internal_weights, 0, np.inf)
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights

        # normalise weights to resampling probabilities
        self.weight_normalisation = np.sum(weights)
        self.weights = weights / self.weight_normalisation

        # Compute effective sample size and entropy of weighting vector.
        # These are useful statistics for adaptive particle filtering.
        self.n_eff = (1.0 / np.sum(self.weights**2)) / self.n_particles
        self.weight_entropy = np.sum(self.weights * np.log(self.weights))

        # preserve current sample set before any replenishment
        self.original_particles = np.array(self.particles)

        # store mean (expected) hypothesis
        self.mean_hypothesis = np.sum(self.hypotheses.T * self.weights, axis=-1).T
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.particles, rowvar=False, aweights=self.weights)

        # store MAP estimate
        argmax_weight = np.argmax(self.weights)
        self.argmax_weight = argmax_weight
        self.map_weight = self.weights[argmax_weight]
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypotheses[argmax_weight]
        self.map_similarity = self.similarities[argmax_weight]
        if zero_hypothesis or zero_template:
            self.map_similarity = 0.0

        # apply any post-processing
        if self.transform_fn:
            self.transformed_particles = self.transform_fn(
                self.original_particles, self.weights, **kwargs
            )
        else:
            self.transformed_particles = self.original_particles

        # resampling (systematic resampling) step
        if self.n_eff < self.n_eff_threshold:
            indices = self.resample_fn(self.weights)
            self.particles = self.particles[indices, :]
            self.weights = np.ones(self.n_particles) / self.n_particles

        # update prior
        r, c, sr, sc = self.map_state
        prior_fn = independent_sample(
            [
                norm(loc=r, scale=observed.shape[0] * 0.07).rvs,
                norm(loc=c, scale=observed.shape[1] * 0.07).rvs,
                norm(loc=sr, scale=0.05).rvs,
                norm(loc=sc, scale=0.05).rvs,
            ]
        )
        self.prior_fn = prior_fn

        # randomly resample some particles from the prior
        if self.resample_proportion > 0:
            random_mask = (
                np.random.random(size=(self.n_particles,)) < self.resample_proportion
            )
            self.resampled_particles = random_mask
            self.init_filter(mask=random_mask)

        # update template
        self.template = (
            self.alpha * self.map_hypothesis + (1 - self.alpha) * self.template
        )

    def viz_particles(self, observed):
        import cv2

        for p in self.particles:
            observed = cv2.drawMarker(
                observed,
                (int(p[1]), int(p[0])),
                (0, 0, 255),
                markerType=cv2.MARKER_STAR,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
