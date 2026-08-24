import pandas as pd

from src.old.extract_reco_crops import rows_with_danish_letters


def test_keeps_rows_with_danish_letters_in_either_field():
    df = pd.DataFrame({
        "image_file": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "verbatimDate": ["12.06.1932", "MISSING", "3 år siden", "1901"],
        "verbatimLocality": ["København", "Århus", "Roskilde", "Odense"],
    })
    kept = rows_with_danish_letters(df)
    assert set(kept["image_file"]) == {"a.jpg", "b.jpg", "c.jpg"}


def test_drops_rows_with_no_danish_letters():
    df = pd.DataFrame({
        "image_file": ["e.jpg"],
        "verbatimDate": ["1945"],
        "verbatimLocality": ["Roskilde"],
    })
    kept = rows_with_danish_letters(df)
    assert kept.empty
