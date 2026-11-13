from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from scipy import stats
from fastapi.middleware.cors import CORSMiddleware
import io

# cria o app em Fast API

app = FastAPI(
    title="Motor de Teste A/B Tático",
    description="API para cálculo de significância estatística de Expected Goals (xG)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# formato e pedido do teste 
class TesteABRequest(BaseModel):
    xg_442: List[float]
    xg_433: List[float]
    nivel_confianca: float = 0.05
    
    
# formata a resposta do teste 

class TesteABResponse(BaseModel):
    media_442: float
    media_433: float
    p_valor: float
    significativo: bool
    veredito: str
    
# Função que executa a análise do teste, separa se a amostra é muito pequena, se o resultado não é significativo o suficiente ou se é e qual a diferença

def executar_analise(xg_442: List[float], xg_433: List[float], nivel_confianca: float = 0.05) -> dict:
    if len(xg_442) < 2 or len(xg_433) < 2:
        raise HTTPException(status_code=400, detail="Amostra muito pequena para o teste.")

    media_442 = float(np.mean(xg_442))
    media_433 = float(np.mean(xg_433))
    
    t_stat, p_value = stats.ttest_ind(xg_433, xg_442, alternative='greater')
    
    is_significativo = bool(p_value < nivel_confianca)
    certeza_pct = (1 - nivel_confianca) * 100
    
    if is_significativo:
        mensagem = f"Temos {certeza_pct:.0f}% de certeza estatística de que o 4-3-3 tem um poder ofensivo superior."
    else:
        mensagem = "Sem evidência estatística suficiente. A diferença pode ser variação natural."

    return {
        "media_442": round(media_442, 2),
        "media_433": round(media_433, 2),
        "p_valor": round(p_value, 4),
        "significativo": is_significativo,
        "veredito": mensagem
    }


# ===================== Rota que calcula via JSON ======================================
@app.post("/calcular", response_model=TesteABResponse)
def calcular_teste_ab(dados: TesteABRequest):
    return executar_analise(dados.xg_442, dados.xg_433, dados.nivel_confianca)

# ===================== Rota que calcula via CSV ======================================
@app.post("/calcular-csv", response_model=TesteABResponse)
async def calcular_teste_ab_csv(
    file: UploadFile = File(...),
    nivel_confianca: float = Form(0.05)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Apenas arquivos .csv são permitidos.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip().str.lower()
        
        # Formato A: Colunas separadas 'xg_442' e 'xg_433'
        if 'xg_442' in df.columns and 'xg_433' in df.columns:
            xg_442 = df['xg_442'].dropna().tolist()
            xg_433 = df['xg_433'].dropna().tolist()
            
        # Formato B: Colunas 'esquema' e 'xg'
    
        elif ('esquema' in df.columns or 'formacao' in df.columns) and 'xg' in df.columns:
            col_esquema = 'esquema' if 'esquema' in df.columns else 'formacao'
            xg_442 = df[df[col_esquema].astype(str).str.contains('4-4-2|442')]['xg'].dropna().tolist()
            xg_433 = df[df[col_esquema].astype(str).str.contains('4-3-3|433')]['xg'].dropna().tolist()
        else:
            raise HTTPException(
                status_code=400, 
                detail="Formato de CSV inválido. Use colunas 'xg_442' e 'xg_433' ou 'esquema' e 'xg'."
            )
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Erro ao processar CSV: {str(e)}")

    return executar_analise(xg_442, xg_433, nivel_confianca)