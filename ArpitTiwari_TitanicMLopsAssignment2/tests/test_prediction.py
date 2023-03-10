import math

import numpy as np

from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
  
    expected_no_predictions = 131

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data["survived"]
    y_pred = _predictions
    accuracy = accuracy_score(_predictions, y_true)
    assert accuracy > 0.7