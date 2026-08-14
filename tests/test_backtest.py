from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest_miyagi as bt
from dados_miyagi import (
    PAINEL_OFICIAL,
    alinhar_ao_calendario,
    carregar_dados_oficiais,
    eh_etf_adjusted_close,
    selecionar_etfs,
)


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

    def test_alinhamento_nao_renova_limite_de_forward_fill(self):
        datas = pd.date_range("2020-01-01", periods=12, freq="D")
        precos = pd.DataFrame({"A": [100.0] + [np.nan] * 10 + [120.0]}, index=datas)

        alinhado = alinhar_ao_calendario(precos, datas, limite=5)

        self.assertTrue((alinhado.loc[datas[1]:datas[5], "A"] == 100.0).all())
        self.assertTrue(alinhado.loc[datas[6]:datas[10], "A"].isna().all())
        self.assertEqual(float(alinhado.at[datas[11], "A"]), 120.0)

    def test_alinhamento_preserva_cotacao_observada_fora_do_cdi(self):
        fonte = pd.to_datetime(["2020-01-03", "2020-01-05"])
        calendario = pd.to_datetime(["2020-01-03", "2020-01-06"])
        precos = pd.DataFrame({"FX": [100.0, 101.0]}, index=fonte)

        alinhado = alinhar_ao_calendario(precos, calendario, limite=1)

        self.assertEqual(float(alinhado.at[calendario[1], "FX"]), 101.0)


class DerivaTest(unittest.TestCase):
    def test_deriva_bate_com_formula_calculavel_a_mao(self):
        pesos = pd.Series({"A": 0.60, "B": -0.40})
        retornos = pd.Series({"A": 0.10, "B": -0.05})
        total = float((pesos * retornos).sum())

        obtido = bt.derivar_pesos(pesos, retornos, total)
        esperado = pesos * (1.0 + retornos) / (1.0 + total)

        pd.testing.assert_series_equal(obtido, esperado)

    def test_giro_seguinte_usa_peso_derivado(self):
        datas = pd.to_datetime(["2020-01-31", "2020-02-28"])
        retornos = pd.DataFrame(
            {"A": [0.0, 0.10], "B": [0.0, -0.05]}, index=datas
        )
        precos = pd.DataFrame(100.0, index=datas, columns=retornos.columns)
        cdi = pd.Series(0.0, index=datas)
        alvo = pd.Series({"A": 0.60, "B": -0.40})

        with (
            patch.object(bt, "calcular_sinal", return_value=pd.Series(1.0, index=alvo.index)),
            patch.object(bt, "calcular_volatilidade", return_value=pd.Series(1.0, index=alvo.index)),
            patch.object(bt, "calcular_pesos_brutos", return_value=alvo),
            patch.object(bt, "volatilidade_da_carteira", return_value=0.10),
            patch.object(bt, "aplicar_alvo_de_risco", return_value=alvo),
        ):
            resultado = bt.rodar_backtest(
                precos, retornos, cdi, inicio=datas[0]
            )

        total = float((alvo * retornos.loc[datas[1]]).sum())
        derivado = alvo * (1.0 + retornos.loc[datas[1]]) / (1.0 + total)
        giro_esperado = float((alvo - derivado).abs().sum())

        pd.testing.assert_series_equal(
            resultado["pesos_antes_rebalanceamento"].loc[datas[1]],
            derivado,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            resultado["pesos_diarios"].loc[datas[1]], alvo,
            check_names=False,
        )
        self.assertAlmostEqual(
            float(resultado["giro"].loc[datas[1]]), giro_esperado
        )

    def test_patrimonio_nao_positivo_interrompe_a_simulacao(self):
        pesos = pd.Series({"A": 1.0})
        retornos = pd.Series({"A": -1.0})

        with self.assertRaisesRegex(RuntimeError, "Patrimônio não positivo"):
            bt.derivar_pesos(
                pesos, retornos, retorno_total=-1.0,
                data=pd.Timestamp("2020-03-16"),
            )

    def test_financiamento_desconta_taxa_da_exposicao_liquida(self):
        pesos = pd.Series({"ETF_LONG": 0.60, "ETF_SHORT": -0.20, "FUT=F": 1.0})
        retornos = pd.Series({"ETF_LONG": 0.01, "ETF_SHORT": -0.02, "FUT=F": 0.03})

        resultado, encargo = bt.calcular_resultado_posicoes(
            pesos,
            retornos,
            taxa_caixa=0.001,
            ativos_financiados={"ETF_LONG", "ETF_SHORT"},
        )

        bruto = float((pesos * retornos).sum())
        self.assertAlmostEqual(encargo, 0.001 * (0.60 - 0.20))
        self.assertAlmostEqual(resultado, bruto - encargo)

    def test_relatorio_de_ordens_usa_peso_pre_trade_derivado(self):
        from figuras_relatorio import extrair_ordens

        datas = pd.to_datetime(["2020-01-31", "2020-02-28"])
        alvos = pd.DataFrame(
            {"A": [0.60, 0.60], "B": [-0.40, -0.40]}, index=datas
        )
        pre_trade = pd.DataFrame(
            {"A": [0.0, 0.63], "B": [0.0, -0.38]}, index=datas
        )

        ordens = extrair_ordens(alvos, pre_trade)
        segunda = ordens[
            (ordens["data"] == datas[1].date()) & (ordens["ativo"] == "A")
        ].iloc[0]

        self.assertEqual(segunda["lado"], "VENDA")
        self.assertAlmostEqual(float(segunda["tamanho"]), 0.03)


class LinhagemTest(unittest.TestCase):
    def test_painel_oficial_eh_o_corrigido(self):
        self.assertEqual(PAINEL_OFICIAL.name, "pool_carrego.csv")
        precos, cdi, universo = carregar_dados_oficiais()
        self.assertEqual(list(precos.columns), universo)
        self.assertTrue(precos.index.equals(cdi.index))

    def test_classificacao_de_etfs_eh_unica_e_explicita(self):
        ativos = ["DBA", "GLD", "BZ=F", "BRL=X", "^AXJO"]
        self.assertTrue(eh_etf_adjusted_close("DBA"))
        self.assertFalse(eh_etf_adjusted_close("BZ=F"))
        self.assertFalse(eh_etf_adjusted_close("BRL=X"))
        self.assertFalse(eh_etf_adjusted_close("^AXJO"))
        self.assertEqual(selecionar_etfs(ativos), {"DBA", "GLD"})

        _, _, universo = carregar_dados_oficiais()
        self.assertEqual(
            selecionar_etfs(universo),
            {"DBA", "EFA", "EWJ", "EWT", "GLD", "IEF", "VNQ",
             "XLE", "XLP", "XLU", "XLV"},
        )


if __name__ == "__main__":
    unittest.main()
