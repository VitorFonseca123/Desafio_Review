# Instalar dependencias 
-pip install virtualenv <br>
-python -m venv venv<br>
-Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned<br>
-venv\Scripts\activate<br>
-pip install -r requirements.txt<br>
- caso baixe algum biblioteca nova use: pip freeze > requirements.txt


# Explicação
- Primeiramente, ao carregar os dados, juntei as colunas de titulo e de mensagem para tratar os textos uma única vez
- Decide usar a coluna review_score como base para meus rótulos, dessa forma reviews com score > 3 seriam considerados positivos, dessa forma eu poderia avaliar o modelo criado, ao invés de apenas separa-los em diferentes clusters
- Mesmo juntado os dois, ainda pode ser que a avaliação não tenha nenhum texto, dessa forma não é interessante para minha ánalise utilizar essas linhas, então elas foram removidas, casos com apenas o review_score podem ser considerados olhando apenas o score
- Escolhi usar dois métodos para vetorizar as palavras, sendo eles a contagem de palavras sendo e o tf_idf, onde um conta apenas a frequência de cada palavra e o outro considera tanto a frequência quanto a importância calculado por log(N/RcP), sendo N o total de registros e Rcp 
- Os dados foram separados em teste e treino, com teste sendo 20% do total, de forma estratificada para manter uma proporção entre os rótulos
- Ambos os métodos foram testados usando o classificador de Regressão logistica
- Foi observado quem ambos os métodos tiveram métricas bem 
- Ao utilizar o DummyClassifier, que sempre tenta chutar a classe mais frequente obteve-se uma acuracia de 0.71, dessa forma, ao ver que ambos as métricas usando a regressão tiveram um resultado superior ao dummy, indicando que algo foi aprendido no modelo.


