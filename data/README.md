# Dados

Os arquivos CSV **não são versionados** neste repositório. Baixe-os na fonte original e envie para o Volume do Databricks,
conforme descrito na seção "Como reproduzir" do [README principal](../README.md).

## Fonte

Base **Wine Quality**, do UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/186/wine+quality

Confira a licença na página do dataset.

## Arquivos necessários

| Arquivo | Registros | Conteúdo |
|---|---|---|
| `winequality-red.csv` | 1.599 | Vinho tinto |
| `winequality-white.csv` | 4.898 | Vinho branco |
| `winequality.names` | n/a | Documentação do dataset (descrição das colunas e da origem); este arquivo pode ser versionado nesta pasta |

Os dois CSVs são necessários. O dataset publicado no Kaggle com o mesmo nome contém **apenas o vinho tinto**, e o pipeline
espera os dois arquivos (os notebooks conferem a presença deles e as contagens de registros).

## Formato

- Separador de campos: `;` (ponto e vírgula).
- Cabeçalho entre aspas, com nomes como `"fixed acidity"` e `"pH"`.
- 11 colunas físico-químicas e a coluna `quality` (nota sensorial).
- Sem valores ausentes.

## Referência

P. Cortez, A. Cerdeira, F. Almeida, T. Matos e J. Reis. *Modeling wine preferences by data mining from physicochemical
properties.* Decision Support Systems, 47(4):547-553, 2009. https://doi.org/10.1016/j.dss.2009.05.016
