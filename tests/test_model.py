import pytest
import torch
from ..CookieProject.models.model import MyAwesomeModel

# tests/test_model.py
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        MyAwesomeModel(torch.randn(1, 28, 28))