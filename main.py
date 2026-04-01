import os
import re
import html
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURAÇÃO DOS ATIVOS
# =========================================================
ATIVOS = [
    # EUA
    {"ticker": "META", "nome": "Meta Platforms", "grupo": "EUA"},
    {"ticker": "MSFT", "nome": "Microsoft", "grupo": "EUA"},
    {"ticker": "UNH", "nome": "UnitedHealth", "grupo": "EUA"},
    {"ticker": "BRK-B", "nome": "Berkshire Hathaway", "grupo": "EUA"},
    {"ticker": "V", "nome": "Visa", "grupo": "EUA"},
    {"ticker": "JNJ", "nome": "Johnson & Johnson", "grupo": "EUA"},
    {"ticker": "KO", "nome": "Coca-Cola", "grupo": "EUA"},
    {"ticker": "HD", "nome": "Home Depot", "grupo": "EUA"},
    {"ticker": "SCHD", "nome": "ETF Dividendos EUA", "grupo": "EUA"},
    {"ticker": "VOO", "nome": "ETF S&P 500", "grupo": "EUA"},

    # Brasil
    {"ticker": "AXIA3.SA", "nome": "Azevedo & Travassos ON", "grupo": "Brasil"},
    {"ticker": "AXIA7.SA", "nome": "Azevedo & Travassos PNA", "grupo": "Brasil"},
    {"ticker": "BBAS3.SA", "nome": "Banco do Brasil", "grupo": "Brasil"},
    {"ticker": "BITH11.SA", "nome": "ETF Hashdex Bitcoin", "grupo": "Brasil"},
    {"ticker": "BPAC11.SA", "nome": "BTG Pactual", "grupo": "Brasil"},
    {"ticker": "SUZB3.SA", "nome": "Suzano", "grupo": "Brasil"},
    {"ticker": "PRIO3.SA", "nome": "PRIO", "grupo": "Brasil"},
    {"ticker": "PETR4.SA", "nome": "Petrobras", "grupo": "Brasil"},
    {"ticker": "VALE3.SA", "nome": "Vale", "grupo": "Brasil"},
    {"ticker": "ITUB4.SA", "nome": "Itaú Unibanco", "grupo": "Brasil"},
    {"ticker": "WEGE3.SA", "nome": "WEG", "grupo": "Brasil"},
    {"ticker": "BBSE3.SA", "nome": "BB Seguridade", "grupo": "Brasil"},
    {"ticker": "B3SA3.SA", "nome": "B3", "grupo": "Brasil"},
    {"ticker": "TAEE11.SA", "nome": "Taesa", "grupo": "Brasil"},
    {"ticker": "ABEV3.SA", "nome": "Ambev", "grupo": "Brasil"},
    {"ticker": "RENT3.SA", "nome": "Localiza", "grupo": "Brasil"},

    # Cripto
    {"ticker": "BTC-USD", "nome": "Bitcoin", "grupo": "Cripto"},
    {"ticker": "ETH-USD", "nome": "Ethereum", "grupo": "Cripto"},

    # Temático
    {"ticker": "REMX", "nome": "ETF Rare Earths", "grupo": "Temático"},
]
# =========================================================
# SUA CARTEIRA
# =========================================================

CARTEIRA_EUA = {
    "SCHD",
    "MSFT",
    "UNH",
    "BRK-B",
    "VOO",
}

CARTEIRA_BRASIL = {
    "AXIA3.SA",
    "AXIA7.SA",
    "BBAS3.SA",
    "BITH11.SA",
    "BPAC11.SA",
    "SUZB3.SA",
    "PRIO3.SA",
}

CARTEIRA_USUARIO = CARTEIRA_EUA.union(CARTEIRA_BRASIL)

EMAIL_REMETENTE = os.getenv("EMAIL_REMETENTE")
SENHA_APP = os.getenv("SENHA_APP")
EMAIL_DESTINO = os.getenv("EMAIL_DESTINO")


# =========================================================
# INDICADORES
# =========================================================
def calcular_rsi(series: pd.Series, periodo: int = 14) -> pd.Series:
    delta = series.diff()
    ganhos = delta.clip(lower=0)
    perdas = -delta.clip(upper=0)

    media_ganhos = ganhos.rolling(periodo).mean()
    media_perdas = perdas.rolling(periodo).mean()

    rs = media_ganhos / media_perdas.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def normalizar_z_acao(z):
    if z is None:
        return 0
    if z <= -3:
        return 10
    if z <= -2.5:
        return 9
    if z <= -2:
        return 8
    if z <= -1.5:
        return 6
    if z <= -1:
        return 4
    return 1


def normalizar_rsi_acao(rsi):
    if rsi is None:
        return 0
    if rsi < 25:
        return 10
    if rsi < 30:
        return 8
    if rsi < 35:
        return 6
    if rsi < 40:
        return 4
    return 1


def normalizar_dd_acao(dd):
    if dd is None:
        return 0
    if dd <= -0.30:
        return 10
    if dd <= -0.20:
        return 8
    if dd <= -0.15:
        return 6
    if dd <= -0.10:
        return 4
    return 1


def normalizar_tendencia_acao(preco, ma50, ma200):
    if preco is None or ma50 is None or ma200 is None:
        return 0
    if preco > ma200 and ma50 > ma200:
        return 10
    if preco > ma200:
        return 7
    if preco > ma50:
        return 5
    return 2


def normalizar_z_cripto(z):
    if z is None:
        return 0
    if z <= -4:
        return 10
    if z <= -3.5:
        return 8
    if z <= -3:
        return 6
    if z <= -2:
        return 4
    return 1


def normalizar_rsi_cripto(rsi):
    if rsi is None:
        return 0
    if rsi < 20:
        return 10
    if rsi < 25:
        return 8
    if rsi < 30:
        return 6
    if rsi < 35:
        return 4
    return 1


def normalizar_dd_cripto(dd):
    if dd is None:
        return 0
    if dd <= -0.40:
        return 10
    if dd <= -0.30:
        return 8
    if dd <= -0.20:
        return 6
    if dd <= -0.12:
        return 4
    return 1


def normalizar_tendencia_cripto(preco, ma50, ma200):
    if preco is None or ma50 is None or ma200 is None:
        return 0
    if preco > ma200 and ma50 > ma200:
        return 8
    if preco > ma200:
        return 6
    if preco > ma50:
        return 4
    return 2


def classificar_score(score):
    if score >= 8:
        return "Compra forte"
    if score >= 6:
        return "Oportunidade"
    if score >= 4:
        return "Monitorar"
    return "Neutro"


# =========================================================
# NOTÍCIAS / MOTIVOS
# =========================================================
PALAVRAS_POSITIVAS = [
    "beats", "beat", "strong", "growth", "surge", "record", "upgrades",
    "approval", "profit", "profits", "buyback", "expands", "wins",
    "higher", "gain", "gains", "bullish", "outperform"
]

PALAVRAS_NEGATIVAS = [
    "miss", "weak", "drops", "drop", "fall", "falls", "cut", "cuts",
    "downgrade", "downgrades", "investigation", "probe", "lawsuit",
    "tariff", "tariffs", "regulation", "warning", "slowdown", "decline",
    "declines", "selloff", "sell-off", "concern", "concerns", "fraud",
    "recall", "loss", "losses"
]


def limpar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = html.unescape(str(texto))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def coletar_noticias(ticker: str, limite: int = 3):
    """
    Tenta coletar manchetes via yfinance.
    Nem todos os tickers retornam notícias de forma consistente.
    """
    noticias_formatadas = []
    sentimento = 0

    try:
        tk = yf.Ticker(ticker)
        news = getattr(tk, "news", None)

        if not news:
            return [], 0

        for item in news[:limite]:
            titulo = limpar_texto(item.get("title", ""))
            publisher = limpar_texto(item.get("publisher", ""))
            link = item.get("link", "")
            resumo = limpar_texto(item.get("summary", ""))

            if not titulo:
                continue

            texto_analise = f"{titulo} {resumo}".lower()

            for palavra in PALAVRAS_POSITIVAS:
                if palavra in texto_analise:
                    sentimento += 1

            for palavra in PALAVRAS_NEGATIVAS:
                if palavra in texto_analise:
                    sentimento -= 1

            linha = titulo
            if publisher:
                linha += f" ({publisher})"
            if link:
                linha += f" - {link}"

            noticias_formatadas.append(linha)

    except Exception:
        return [], 0

    return noticias_formatadas, sentimento


def montar_motivos_probaveis(grupo, zscore, rsi, drawdown, preco, ma200, sentimento_noticias):
    motivos = []

    if grupo == "Cripto":
        if zscore is not None and zscore <= -3:
            motivos.append("queda muito fora do padrão")
        elif zscore is not None and zscore <= -2:
            motivos.append("queda relevante")

        if rsi is not None and rsi < 30:
            motivos.append("sobrevendido")

        if drawdown is not None and drawdown <= -0.30:
            motivos.append("bem abaixo da máxima recente")
    else:
        if zscore is not None and zscore <= -2:
            motivos.append("queda estatisticamente forte")
        elif zscore is not None and zscore <= -1.5:
            motivos.append("queda acima do normal")

        if rsi is not None and rsi < 30:
            motivos.append("sobrevendido")
        elif rsi is not None and rsi < 35:
            motivos.append("perto de sobrevenda")

        if drawdown is not None and drawdown <= -0.20:
            motivos.append("bem abaixo da máxima recente")
        elif drawdown is not None and drawdown <= -0.10:
            motivos.append("em correção")

    if preco is not None and ma200 is not None:
        if preco > ma200:
            motivos.append("tendência estrutural saudável")
        else:
            motivos.append("tendência estrutural fraca")

    if sentimento_noticias <= -2:
        motivos.append("noticiário recente negativo")
    elif sentimento_noticias >= 2:
        motivos.append("noticiário recente positivo")
    elif sentimento_noticias != 0:
        motivos.append("noticiário misto")

    if not motivos:
        motivos.append("sem distorção forte no momento")

    return ", ".join(motivos)


# =========================================================
# ANÁLISE
# =========================================================
def analisar_ativo(ativo):
    ticker = ativo["ticker"]
    nome = ativo["nome"]
    grupo = ativo["grupo"]

    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        if "Close" not in df.columns:
            return None

        close = df["Close"].dropna()

        if len(close) < 200:
            return None

        preco = safe_float(close.iloc[-1])
        preco_anterior = safe_float(close.iloc[-2]) if len(close) >= 2 else None
        media30 = safe_float(close.tail(30).mean())
        std30 = safe_float(close.tail(30).std())
        ma50 = safe_float(close.rolling(50).mean().iloc[-1])
        ma200 = safe_float(close.rolling(200).mean().iloc[-1])
        max90 = safe_float(close.tail(90).max())
        rsi = safe_float(calcular_rsi(close, 14).iloc[-1])

        queda_dia = None
        if preco is not None and preco_anterior not in (None, 0):
            queda_dia = ((preco / preco_anterior) - 1) * 100

        dist_media30 = None
        if preco is not None and media30 not in (None, 0):
            dist_media30 = ((preco / media30) - 1) * 100

        zscore = None
        if preco is not None and media30 is not None and std30 not in (None, 0):
            zscore = (preco - media30) / std30

        drawdown = None
        if preco is not None and max90 not in (None, 0):
            drawdown = (preco / max90) - 1

        noticias, sentimento_noticias = coletar_noticias(ticker, limite=3)

        if grupo == "Cripto":
            nota_z = normalizar_z_cripto(zscore)
            nota_rsi = normalizar_rsi_cripto(rsi)
            nota_dd = normalizar_dd_cripto(drawdown)
            nota_tend = normalizar_tendencia_cripto(preco, ma50, ma200)
        else:
            nota_z = normalizar_z_acao(zscore)
            nota_rsi = normalizar_rsi_acao(rsi)
            nota_dd = normalizar_dd_acao(drawdown)
            nota_tend = normalizar_tendencia_acao(preco, ma50, ma200)

        score = (
            nota_z * 0.40 +
            nota_rsi * 0.20 +
            nota_dd * 0.20 +
            nota_tend * 0.20
        )

        motivos = montar_motivos_probaveis(
            grupo=grupo,
            zscore=zscore,
            rsi=rsi,
            drawdown=drawdown,
            preco=preco,
            ma200=ma200,
            sentimento_noticias=sentimento_noticias,
        )

        return {
            "ticker": ticker,
            "nome": nome,
            "grupo": grupo,
            "preco": preco,
            "queda_dia": queda_dia,
            "dist_media30": dist_media30,
            "zscore": zscore,
            "rsi": rsi,
            "drawdown": drawdown * 100 if drawdown is not None else None,
            "ma50": ma50,
            "ma200": ma200,
            "score": score,
            "classificacao": classificar_score(score),
            "motivos": motivos,
            "noticias": noticias,
            "em_carteira": ticker in CARTEIRA_USUARIO,
        }

    except Exception as e:
        print(f"Erro ao analisar {ticker}: {e}")
        return None


# =========================================================
# FORMATAÇÃO
# =========================================================
def fmt_num(v, casas=2):
    if v is None:
        return "n/d"
    return f"{v:.{casas}f}"


def fmt_pct(v, casas=2):
    if v is None:
        return "n/d"
    return f"{v:.{casas}f}%"


def montar_linha_titulo(item):
    estrela = "⭐ " if item["em_carteira"] else ""
    return f"{estrela}{item['ticker']} ({item['nome']})"


def top_por_grupo(resultados, grupo, limite=3):
    grupo_filtrado = [r for r in resultados if r["grupo"] == grupo]
    return grupo_filtrado[:limite]


# =========================================================
# E-MAIL
# =========================================================
def montar_email(resultados):
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")

    linhas = []
    linhas.append("Radar Inteligente - Oportunidades (Compra Forte)")
    linhas.append(f"Atualizado em: {agora}")
    linhas.append("")
    linhas.append("⭐ = ativo que já está na sua carteira")
    linhas.append("")

    # filtrar somente compra forte
    fortes = [r for r in resultados if r["classificacao"] == "Compra forte"]

    if not fortes:
        linhas.append("Nenhum ativo com sinal de COMPRA FORTE hoje.")
        linhas.append("")
        linhas.append("Sugestão: aguardar melhores assimetrias.")
        return "\n".join(linhas)

    # ordenar por score
    fortes.sort(key=lambda x: x["score"], reverse=True)

    linhas.append("=== Oportunidades do dia ===")

    for item in fortes:
        estrela = "⭐ " if item["em_carteira"] else ""
        linhas.append(f"{estrela}{item['ticker']} ({item['nome']})")
        linhas.append(f"Score: {item['score']:.1f}")
        linhas.append(f"Queda do dia: {item['queda_dia']:.2f}%")
        linhas.append(f"Motivos: {item['motivos']}")

        if item["noticias"]:
            linhas.append("Notícias:")
            for n in item["noticias"]:
                linhas.append(f"- {n}")
        else:
            linhas.append("Notícias: n/d")

        linhas.append("")

    linhas.append("=== Leitura rápida ===")
    linhas.append("Compra forte = queda relevante + indicadores estatísticos favoráveis")
    linhas.append("Normalmente representa distorções de curto prazo com melhor assimetria")

    return "\n".join(linhas)


# =========================================================
# ENVIO DE E-MAIL
# =========================================================
def enviar_email(assunto, corpo):
    if not EMAIL_REMETENTE or not SENHA_APP or not EMAIL_DESTINO:
        raise ValueError(
            "Faltam EMAIL_REMETENTE, SENHA_APP ou EMAIL_DESTINO nos Secrets."
        )

    msg = MIMEText(corpo, "plain", "utf-8")
    msg["Subject"] = assunto
    msg["From"] = EMAIL_REMETENTE
    msg["To"] = EMAIL_DESTINO

    with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465) as server:
        server.login(EMAIL_REMETENTE, SENHA_APP)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINO, msg.as_string())


# =========================================================
# MAIN
# =========================================================
def main():
    resultados = []

    for ativo in ATIVOS:
        resultado = analisar_ativo(ativo)
        if resultado:
            resultados.append(resultado)

    resultados.sort(key=lambda x: x["score"], reverse=True)

    fortes = [r for r in resultados if r["classificacao"] == "Compra forte"]

    if not fortes:
        print("Nenhum ativo com Compra forte hoje. E-mail não enviado.")
        return

    corpo = montar_email(resultados)
    assunto = "Radar Inteligente - Compra Forte"

    print(corpo)
    enviar_email(assunto, corpo)
    print("E-mail enviado com sucesso.")