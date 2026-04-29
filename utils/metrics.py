def calculate_fake_ratio(results):
    """
    results: ['real', 'fake', 'fake', ...]
    """
    if not results:
        return 0

    fake_count = results.count("fake")
    return fake_count / len(results)


def interpret_result(ratio):
    """
    ratio: 0.0 - 1.0
    """
    if ratio < 0.3:
        return "Görüntü büyük ihtimalle GERÇEK"
    elif ratio < 0.7:
        return "ŞÜPHELİ içerik"
    else:
        return "Görüntü büyük ihtimalle SAHTE"