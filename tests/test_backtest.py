from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import backtest_miyagi as bt
from dados_miyagi import PAINEL_OFICIAL, carregar_dados_oficiais


class RetornosTest(unittest.TestCase):
    def test_pct_change_nao_cria_salto_apos_lacuna(self):
        datas = pd.date_range("2020-01-01", periods=3, freq="D")
        precos = pd.DataFrame({"A": [100.0, np.nan, 120.0]}, index=datas)
        retornos = bt.calcular_retornos(precos)
        self.assertTrue(retornos["A"].isna().all())

    def test_sinal_zero_sem_preco_na_decisao(self):
        datas = pd.bdate_range("2018-01-01", periods=300)
        precos = pd.DataFrame({"A": np.linspace(100, 150, len(datas))}, index=datas)
        precos.iloc[-1, 0] = np.nan
        sinal = bt.calcular_sinal(precos, datas[-1])
        self.assertEqual(float(sinal["A"]), 0.0)


class LinhagemTest(unittest.TestCase):
    def test_painel_oficial_eh_o_corrigido(self):
        self.assertEqual(PAINEL_OFICIAL.name, "pool_carrego.csv")
        precos, cdi, universo = carregar_dados_oficiais()
        self.assertEqual(list(precos.columns), universo)
        self.assertTrue(precos.index.equals(cdi.index))


if __name__ == "__main__":
    unittest.main()
