from __future__ import annotations

import csv
from pathlib import Path
import unittest


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"


def ler_csv_por_chave(nome: str, chave: str) -> dict[str, dict[str, str]]:
    with (RESULTADOS / nome).open(encoding="utf-8", newline="") as arquivo:
        return {linha[chave]: linha for linha in csv.DictReader(arquivo)}


def percentual(valor: str, casas: int) -> str:
    return f"{float(valor) * 100:.{casas}f}".replace(".", ",") + "%"


def decimal(valor: str, casas: int) -> str:
    return f"{float(valor):.{casas}f}".replace(".", ",")


def inteiro(valor: str) -> int:
    return int(float(valor))


def numero_por_extenso(valor: int) -> str:
    nomes = (
        "zero", "um", "dois", "três", "quatro",
        "cinco", "seis", "sete", "oito", "nove", "dez",
    )
    return nomes[valor]


class ConsistenciaDocumentalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (RAIZ / "README.md").read_text(encoding="utf-8")

    def test_metricas_principais_batem_com_csv_canonico(self):
        metricas = ler_csv_por_chave(
            "auditoria_financiamento_metricas.csv", "cenario"
        )
        overlay = metricas["overlay_atual"]
        etfs = metricas["etfs_financiados"]
        estatistica_overlay = ler_csv_por_chave(
            "auditoria_estatistica_resumo.csv", ""
        )
        estatistica_etfs = ler_csv_por_chave(
            "auditoria_estatistica_resumo_etfs.csv", ""
        )

        linha_overlay = (
            "| overlay homogêneo, resultado histórico | "
            f"{percentual(overlay['cagr'], 1)} | {percentual(overlay['vol'], 1)} | "
            f"{decimal(overlay['sharpe'], 2)} | "
            f"{decimal(estatistica_overlay['t_iid']['valor'], 2)} | "
            f"−{percentual(str(abs(float(overlay['max_drawdown']))), 1)} |"
        )
        linha_etfs = (
            f"| **{inteiro(etfs['ativos_financiados'])} ETFs financiados a CDI** | "
            f"**{percentual(etfs['cagr'], 1)}** | **{percentual(etfs['vol'], 1)}** | "
            f"**{decimal(etfs['sharpe'], 2)}** | "
            f"**{decimal(estatistica_etfs['t_iid']['valor'], 2)}** | "
            f"**−{percentual(str(abs(float(etfs['max_drawdown']))), 1)}** |"
        )

        self.assertIn(linha_overlay, self.readme)
        self.assertIn(linha_etfs, self.readme)

    def test_qualidade_dos_dados_bate_com_csv_canonico(self):
        resumo = ler_csv_por_chave("auditoria_dados_resumo.csv", "")

        lacunas = inteiro(resumo["lacunas_internas_apos_ffill_limitado"]["valor"])
        dias = inteiro(resumo["dias_com_posicao_e_algum_retorno_ausente"]["valor"])
        fracao_dias = percentual(
            resumo["fracao_dos_dias_com_exposicao_ausente"]["valor"], 1
        )
        exposicao = percentual(
            resumo["exposicao_abs_maxima_quando_ausente"]["valor"], 2
        )

        self.assertIn(f"**{lacunas} lacunas internas**", self.readme)
        self.assertIn(f"**{dias} dias ({fracao_dias})**", self.readme)
        self.assertIn(f"**{exposicao} do patrimônio no overlay**", self.readme)

    def test_placar_de_robustez_bate_com_csvs_canonicos(self):
        placares = {}
        totais = {}
        for financiamento in ("overlay", "etfs"):
            linhas = list(ler_csv_por_chave(
                f"auditoria_robustez_40_{financiamento}.csv", "teste"
            ).values())
            totais[financiamento] = len(linhas)
            placares[financiamento] = sum(
                linha["aprovado"].strip().lower() == "true" for linha in linhas
            )

        self.assertEqual(placares["overlay"], totais["overlay"])
        self.assertEqual(totais["etfs"], totais["overlay"])
        self.assertIn(
            "Os " + numero_por_extenso(totais["overlay"])
            + " critérios de robustez permanecem aprovados",
            self.readme,
        )
        self.assertIn(
            f"Com os ETFs financiados, passam {placares['etfs']} "
            f"de {totais['etfs']}",
            self.readme,
        )

    def test_snapshots_textuais_obsoletos_nao_existem(self):
        snapshots = (
            RESULTADOS / "resultado_final_auditado.txt",
            RESULTADOS / "robustez_40_auditada.txt",
        )
        for snapshot in snapshots:
            with self.subTest(snapshot=snapshot.name):
                self.assertFalse(snapshot.exists())


if __name__ == "__main__":
    unittest.main()
