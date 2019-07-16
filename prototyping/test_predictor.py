import Predictor

def test_predictor():
    response = Predictor.predict("Tell me the weight of the frames on order x") 
    print(response)
    assert response is not None