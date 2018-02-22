def test_easi(easi):
    import pandas as pd

    data = pd.read_csv('hixdata.csv')
    c = data.columns.to_series()

    labels = dict(
        share=c['sfoodh':'spers'],
        price=c['pfoodh':'ppers'],
        demo=c['age':'tran'],
        log_exp='log_y',
        )

    easi(data, labels)
