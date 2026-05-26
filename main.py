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


def coletar_noticias(ticker: str, limite: int = 2):
    """
    Tenta coletar manchetes via yfinance/Yahoo Finance.
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

            noticias_formatadas.append({
                "titulo": titulo,
                "fonte": publisher,
                "link": link,
            })

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

        # Para montar o e-mail diário, bastam dois fechamentos.
        # Os indicadores adicionais ficam como n/d quando não há histórico suficiente.
        if len(close) < 2:
            return None

        preco = safe_float(close.iloc[-1])
        preco_anterior = safe_float(close.iloc[-2])

        media30 = safe_float(close.tail(30).mean()) if len(close) >= 30 else None
        std30 = safe_float(close.tail(30).std()) if len(close) >= 30 else None
        ma50 = safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        max90 = safe_float(close.tail(90).max()) if len(close) >= 90 else None
        rsi = safe_float(calcular_rsi(close, 14).iloc[-1]) if len(close) >= 15 else None

        variacao_fechamento = None
        if preco is not None and preco_anterior not in (None, 0):
            variacao_fechamento = ((preco / preco_anterior) - 1) * 100

        dist_media30 = None
        if preco is not None and media30 not in (None, 0):
            dist_media30 = ((preco / media30) - 1) * 100

        zscore = None
        if preco is not None and media30 is not None and std30 not in (None, 0):
            zscore = (preco - media30) / std30

        drawdown = None
        if preco is not None and max90 not in (None, 0):
            drawdown = (preco / max90) - 1

        noticias, sentimento_noticias = coletar_noticias(ticker, limite=2)

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
            "preco_anterior": preco_anterior,
            "variacao_fechamento": variacao_fechamento,
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
    sinal = "+" if v > 0 else ""
    return f"{sinal}{v:.{casas}f}%"


def fmt_preco(v, grupo):
    if v is None:
        return "n/d"

    # Brasil em R$; EUA/Temático/Cripto em US$
    prefixo = "R$" if grupo == "Brasil" else "US$"
    valor = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefixo} {valor}"


def html_escape(value):
    return html.escape(str(value)) if value is not None else ""


def montar_linha_titulo(item):
    estrela = "⭐ " if item["em_carteira"] else ""
    return f"{estrela}{item['ticker']} ({item['nome']})"


def formatar_noticias_html(noticias):
    if not noticias:
        return "n/d"

    partes = []
    for noticia in noticias:
        titulo = html_escape(noticia.get("titulo", ""))
        fonte = html_escape(noticia.get("fonte", ""))
        link = noticia.get("link", "")

        if not titulo:
            continue

        if link:
            item = f'<a href="{html_escape(link)}">{titulo}</a>'
        else:
            item = titulo

        if fonte:
            item += f" <span style='color:#666;'>({fonte})</span>"

        partes.append(item)

    return "<br>".join(partes) if partes else "n/d"


def montar_tabela_html(titulo, itens):
    linhas = []
    linhas.append(f"<h2>{html_escape(titulo)}</h2>")

    if not itens:
        linhas.append("<p>Sem dados disponíveis.</p>")
        return "\n".join(linhas)

    linhas.append("""
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif; font-size:13px;">
      <thead>
        <tr style="background-color:#f2f2f2;">
          <th align="left">Ativo</th>
          <th align="right">Fechamento atual</th>
          <th align="right">Var. vs fechamento anterior</th>
          <th align="left">Classificação</th>
          <th align="left">Notícias relevantes</th>
        </tr>
      </thead>
      <tbody>
    """)

    for item in itens:
        variacao = item["variacao_fechamento"]
        if variacao is None:
            variacao_html = "n/d"
        else:
            cor = "#0a7d28" if variacao >= 0 else "#b00020"
            variacao_html = f"<span style='color:{cor};'>{html_escape(fmt_pct(variacao))}</span>"

        linhas.append(f"""
        <tr>
          <td>{html_escape(montar_linha_titulo(item))}</td>
          <td align="right">{html_escape(fmt_preco(item["preco"], item["grupo"]))}</td>
          <td align="right">{variacao_html}</td>
          <td>{html_escape(item["classificacao"])}</td>
          <td>{formatar_noticias_html(item["noticias"])}</td>
        </tr>
        """)

    linhas.append("""
      </tbody>
    </table>
    """)
    return "\n".join(linhas)


# =========================================================
# E-MAIL
# =========================================================
def montar_email(resultados):
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")

    brasil = sorted(
        [r for r in resultados if r["grupo"] == "Brasil"],
        key=lambda x: x["ticker"]
    )

    globais = sorted(
        [r for r in resultados if r["grupo"] in {"EUA", "Cripto", "Temático"}],
        key=lambda x: (x["grupo"], x["ticker"])
    )

    fortes = sorted(
        [r for r in resultados if r["classificacao"] == "Compra forte"],
        key=lambda x: x["score"],
        reverse=True
    )

    if fortes:
        resumo_oportunidades = "<ul>"
        for item in fortes:
            resumo_oportunidades += (
                f"<li><strong>{html_escape(montar_linha_titulo(item))}</strong> "
                f"| Score {fmt_num(item['score'], 1)} "
                f"| {html_escape(item['motivos'])}</li>"
            )
        resumo_oportunidades += "</ul>"
    else:
        resumo_oportunidades = (
            "<p>Nenhum ativo foi classificado como <strong>Compra Forte</strong> nesta execução. "
            "O relatório diário segue abaixo para acompanhamento dos preços e notícias.</p>"
        )

    corpo_html = f"""
    <html>
      <body style="font-family:Arial, sans-serif; color:#222;">
        <h1>Radar Diário de Investimentos</h1>

        <p><strong>Atualizado em:</strong> {html_escape(agora)}</p>
        <p><strong>Legenda:</strong> ⭐ = ativo que já está na sua carteira.</p>

        <h2>Resumo de oportunidades</h2>
        {resumo_oportunidades}

        {montar_tabela_html("Ativos Brasil", brasil)}

        <br>

        {montar_tabela_html("Ativos Globais", globais)}

        <br>

        <h2>Glossário rápido</h2>
        <ul>
          <li><strong>Fechamento atual:</strong> último fechamento disponível via Yahoo Finance/yfinance.</li>
          <li><strong>Var. vs fechamento anterior:</strong> variação percentual entre o fechamento atual e o fechamento imediatamente anterior.</li>
          <li><strong>Classificação:</strong> leitura automática do modelo estatístico do radar.</li>
          <li><strong>Notícias relevantes:</strong> manchetes recentes disponíveis via yfinance/Yahoo Finance. A cobertura pode variar por ativo.</li>
        </ul>

        <p style="font-size:12px; color:#666;">
          Observação: este radar é uma ferramenta de monitoramento e não representa recomendação individual de investimento.
        </p>
      </body>
    </html>
    """

    return corpo_html


# =========================================================
# ENVIO DE E-MAIL
# =========================================================
def enviar_email(assunto, corpo_html):
    if not EMAIL_REMETENTE or not SENHA_APP or not EMAIL_DESTINO:
        raise ValueError(
            "Faltam EMAIL_REMETENTE, SENHA_APP ou EMAIL_DESTINO nos Secrets."
        )

    msg = MIMEText(corpo_html, "html", "utf-8")
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

    corpo = montar_email(resultados)
    assunto = "Radar Diário de Investimentos - Brasil e Globais"

    print("Resumo da execução:")
    print(f"- Ativos analisados: {len(resultados)}")
    print(f"- Compra forte: {len([r for r in resultados if r['classificacao'] == 'Compra forte'])}")
    print("Enviando e-mail diário, independentemente de haver Compra Forte...")

    enviar_email(assunto, corpo)
    print("E-mail enviado com sucesso.")


if __name__ == "__main__":
    main()
