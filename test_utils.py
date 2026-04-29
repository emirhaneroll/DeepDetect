from utils.metrics import calculate_fake_ratio, interpret_result

def test_calculate_fake_ratio():
    results = ["fake", "real", "fake"]
    assert calculate_fake_ratio(results) == 2/3

def test_interpret_result():
    assert interpret_result(0.1) == "Görüntü büyük ihtimalle GERÇEK"
    assert interpret_result(0.5) == "ŞÜPHELİ içerik"
    assert interpret_result(0.9) == "Görüntü büyük ihtimalle SAHTE"