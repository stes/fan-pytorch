class NormalizationNetwork(object):
    """ Inititalizes a new network with a sklearn compatible interface
    """

    def __init__(self,batch_size = 64,
                 patch_size = tools.INP_PSIZE,
                 fname=None,
                 debug_connections=False,
                 clip = [0, 255]):
        assert fname is None or os.path.exists(fname),\
            "Provided file name for weights does not exist"

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_size = (batch_size, 3, patch_size, patch_size)
        self.fname_weights = fname

        val_fn, network, layers   = compile_validation(self.input_size)
        self.network       = network
        self.layers        = layers
        self._transform    = val_fn
        self.clip = clip

        self.load_weights(self.fname_weights)

        self.output_shape = nn.layers.get_output_shape(self.network)[2:]

    def __str__(self):
        pass

    def __repr__(self):
        return "FAN module, {} -> {}".format(self.input_size,\
                                             self.output_shape)

    def __call__(self, X):
        return self.transform(X)

    def fit(self, X, y=None):
        pass

    def crop(self, X):
        """
        Crops and Image to same dimensions as normalized images
        """
        _,w,h,_ = X.shape
        offx = (w - self.output_shape[0]) // 2
        offy = (h - self.output_shape[1]) // 2

        return X[:,offx:offx+self.output_shape[0],offy:offy+self.output_shape[1],:]

    def transform(self, X):
        normed = []
        normed_imgs = np.zeros((X.shape[0],) + self.output_shape + (3,))
        bsize = self.batch_size

        for i in range(0, X.shape[0], bsize):
            img = X[i:i+bsize,...].transpose((0,3,1,2))
            outp = self._transform(img)
            normed_imgs[i:i+bsize,...] = outp.transpose((0,2,3,1))

        if self.clip is not None:
            normed_imgs = np.clip(normed_imgs, *self.clip)

        return normed_imgs

    def load_weights(self, fname):
        tools.load_weights(self.network, fname)