






class Dataloader(tf.keras.utils.Sequence):
    def __init__(self,
                 dataset,
                 dataset_size, # insert len of img_path from Load_Dataset class
                 batch_size,
                 shuffle=False):
        
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(self.dataset_size)

        self.on_epoch_end()

    def __len__(self):
        # Number of Batches per Epoch
        return int(np.floor(self.dataset_size / self.batch_size))  # len(self.list_IDs)

    def __getitem__(self,i):
        start = i * self.batch_size
        stop = (i+1) * self.batch_size
        data = []

        for j in range(start,stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def on_epoch_end(self):
        #self.indexes = np.arange(self.dataset_size)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
