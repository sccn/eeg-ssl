from ssl_dataloader import *
import numpy as np

cache_dir = '/expanse/projects/nemar/dtyoung/eeg-ssl/cache-test'
data_dir  = '/home/dtyoung/eeg-ssl/libs/test-data'
def test_relative_positioning():
    RP = RelativePositioning({'sfreq': 128, 'cache_dir': '/expanse/projects/nemar/dtyoung/eeg-ssl/cache-test'})
    test_data = np.random.normal(size=(60, 10000))
    test_key = 'test_sample'

    data, labels = RP.transform(test_data, test_key)
    values = np.unique(labels)

    # dimension verification
    assert data.shape[1] == 2, "Each sample must contain two samples, one for anchor and one for pos/neg window"
    assert len(data) == len(labels), "There must be equivalent number of samples as labels"
    assert len(values) == 2, "There must be only two classes"
    
    # value verification
    assert set(values) == {0, 1}, "Labels must be 0 or 1"

    return data, labels

def test_dataloader():
    dataset = ChildmindSSLDataset(
        data_dir=data_dir,
        x_params={
            "feature": "RelativePositioning", 
            "cache_dir": cache_dir,
        },
    )

    assert dataset[0][0].shape == (2,128,128), "Sample dimension not expected"

    print(dataset.subjects)
    return dataset

if __name__ == "__main__":
    # data, labels = test_relative_positioning()
    dataset = test_dataloader()