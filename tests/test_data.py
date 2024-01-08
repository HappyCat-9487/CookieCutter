from ..CookieProject.data.make_dataset import load_corruptmnist
import os
import pytest
file_path = "/Users/luchengliang/MLO/CookieProject/data"
@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")

def test_data():
    dataset = load_corruptmnist()
    N_train = 250000
    N_test = 5000
    #for training and N_test for test
    assert len(dataset) == N_train + N_test, "Dataset did not have the correct number of samples" 
    #for training and N_test for test
    # Check shapes and labels
    for i in range(len(dataset)):
        data_point, label = dataset[i]
        assert (data_point.shape == (1, 28, 28) or data_point.shape == (784,)), f"Invalid shape at index {i}"
        assert 0 <= label < 10, f"Invalid label at index {i}, label: {label}"