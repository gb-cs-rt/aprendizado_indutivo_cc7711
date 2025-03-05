# Relatório: Construção e Avaliação de Árvores de Decisão

## 1. Introdução: Como gerar uma árvore de decisão?

Uma árvore de decisão é construída a partir de um conjunto de dados rotulado, onde cada instância possui atributos e uma classe associada. O algoritmo seleciona, a cada nó, o atributo mais relevante para dividir os dados, maximizando a pureza dos subconjuntos gerados.

A relevância de um atributo é determinada pelo **ganho de informação**, que mede a redução da incerteza após a divisão dos dados. Esse cálculo é baseado na **entropia**, definida como:

\[
H(S) = -\sum p_i \log_2(p_i)
\]

Onde \( p_i \) é a proporção de instâncias de cada classe no conjunto \( S \). Se os dados são perfeitamente homogêneos (todas as instâncias pertencem à mesma classe), a entropia é **0**. Isso indica que o conjunto está completamente "puro" e não há mais necessidade de divisões.

A árvore cresce de forma recursiva até atingir critérios de parada, como um número mínimo de instâncias por nó ou entropia zero, garantindo que cada folha represente uma classe específica.

---

## 2. Análise do Dataset **CriterioProvas.arff**

### 2.1 Explicação do Dataset

Esse dataset contém registros de alunos com notas das provas **P1** e **P2**, percentual de faltas (**PercFalta**) e a classificação final (**Aprovado** ou **Reprovado**). O algoritmo de aprendizado indutivo gerou uma árvore de decisão com base nesses atributos, utilizando a **entropia** como critério de divisão.

### 2.2 Interpretação da Árvore de Decisão

![tree](https://raw.githubusercontent.com/gb-cs-rt/aprendizado_indutivo_cc7711/refs/heads/main/criterio_provas/tree.png)

A árvore gerada reflete um modelo que classifica os alunos conforme suas notas e percentual de faltas. Alguns pontos importantes:

- O primeiro critério de divisão foi **P1 <= 24.5**, indicando que a primeira prova tem grande influência na classificação dos alunos.
- Outras divisões relevantes envolveram **P2** e **PercFalta**, o que demonstra que esses fatores também afetam a decisão final.
- Nós com **entropia = 0** indicam classificações totalmente puras, ou seja, onde todos os exemplos pertencem à mesma classe (**Aprovado** ou **Reprovado**). Isso sugere que os dados estavam bem estruturados para permitir uma separação clara.

### 2.3 Análise da Matriz de Confusão

![confusion_matrix](https://raw.githubusercontent.com/gb-cs-rt/aprendizado_indutivo_cc7711/refs/heads/main/criterio_provas/confusion_matrix.png)

A matriz de confusão gerada confirma que o modelo conseguiu classificar corretamente todos os exemplos disponíveis, pois:

- Não houve **falsos positivos** nem **falsos negativos**.
- Todos os **73 alunos aprovados** foram corretamente identificados.
- Todos os **234 alunos reprovados** também foram corretamente classificados.

Esse resultado sugere que o conjunto de dados é bastante determinístico, ou seja, os atributos escolhidos são suficientes para prever a **aprovação** ou **reprovação** com alta precisão.

### 2.4 Considerações sobre Overfitting

Quando usamos o mesmo conjunto de dados tanto para **treinar** quanto para **avaliar** o modelo, é comum que a árvore de decisão alcance **100% de acurácia**, pois ela pode "memorizar" as regras que classificam perfeitamente todos os exemplos. No entanto, esse resultado pode não se generalizar bem para novos dados.

Para validar melhor o modelo, seria necessário utilizar um **conjunto de teste separado** ou aplicar **validação cruzada**. Dessa forma, poderíamos verificar se o modelo realmente aprendeu padrões úteis e generalizáveis, ou se apenas decorou o conjunto de treinamento.

---

## 3. Análise do Dataset **Grass Grubs**

### 3.1 Explicação do Dataset

Este conjunto de dados trata da presença de **grass grubs** (uma praga agrícola) em plantações de pastagem na região de Canterbury, Nova Zelândia, levando em consideração fatores como:

- **Ano e localização (year, year_zone, zone):** Indicadores de quando e onde foram feitas as medições, separando a região em “foothills” (F), “midplain” (M) e “coastal” (C), e categorizando os anos em diferentes zonas (f, m, c).
- **strip e pdk:** Identificadores do local específico (faixa e pasto) onde foram feitas as amostras.
- **damage_rankRJT e damage_rankALL:** Rankings de dano atribuídos por diferentes pesquisadores, variando de **0 a 5** (sem dano até dano severo).
- **dry_or_irr:** Indica se o pasto era seco (**D**) ou irrigado (**O** ou **B**).
- **GG_new:** A variável alvo, que indica a quantidade de grass grubs por metro quadrado, categorizada em **low, average, high ou veryhigh**.

O objetivo do modelo de árvore de decisão é prever a classe **GG_new** (quantidade de praga) a partir dos demais atributos.

### 3.2 Interpretação da Árvore de Decisão

![tree](https://raw.githubusercontent.com/gb-cs-rt/aprendizado_indutivo_cc7711/refs/heads/main/grub_damage/tree.png)

- A árvore resultante é relativamente grande, indicando que o algoritmo encontrou várias regras de decisão específicas para diferenciar as quatro classes (**low, average, high, veryhigh**).
- Cada divisão (nó) da árvore usa um atributo para separar os exemplos, com base em como melhor reduz a **entropia** (ganho de informação).
- A complexidade pode indicar que o conjunto de dados tem diversos fatores que influenciam na quantidade de pragas, como ano, localização e tipo de irrigação.

### 3.3 Análise da Matriz de Confusão

![confusion_matrix](https://raw.githubusercontent.com/gb-cs-rt/aprendizado_indutivo_cc7711/refs/heads/main/grub_damage/confusion_matrix.png)

- A matriz de confusão mostra a predição exata de todas as instâncias em suas classes corretas:  
   - **49 instâncias** classificadas como **low** (corretamente).  
   - **41 instâncias** classificadas como **average** (corretamente).  
   - **46 instâncias** classificadas como **high** (corretamente).  
   - **19 instâncias** classificadas como **veryhigh** (corretamente).  
   - Não há nenhum erro de classificação (**falsos positivos** ou **falsos negativos**), resultando em **100% de acurácia para o conjunto de treinamento**.

### 3.4 Interpretação e Cuidados

- A classificação perfeita sugere que os atributos disponíveis são suficientes para **separar completamente** as quatro classes dentro deste conjunto de dados. No entanto, assim como no caso anterior, é importante lembrar que **usar o mesmo conjunto para treinar e avaliar** tende a superestimar a capacidade de generalização do modelo.
- Para verificar se o modelo realmente aprendeu padrões que se aplicam a novos dados (e não apenas "decorou" o conjunto de treinamento), seria recomendável fazer uma **validação cruzada** ou usar um **conjunto de teste separado**.

### 3.5 Conclusão  

- O modelo conseguiu criar regras específicas para cada faixa de infestação (**low, average, high, veryhigh**) de forma a classificar todos os exemplos corretamente.
- A alta complexidade da árvore pode indicar muitos caminhos de decisão, cada um focado em um subconjunto de exemplos.
- Se o objetivo for aplicar esse modelo a dados futuros, é fundamental avaliar se esses mesmos padrões se mantêm e se o desempenho se sustenta em novos cenários.

---

## 4. Considerações Finais

Este estudo demonstrou o uso de árvores de decisão para classificar alunos (**CriterioProvas**) e prever a infestação de pragas em plantações (**Grass Grubs**). Em ambos os casos, o modelo obteve **100% de acurácia no conjunto de treinamento**, o que pode indicar que os dados são facilmente separáveis ou que o modelo está sobreajustado.

A principal recomendação para garantir a validade desses modelos é a realização de testes com novos dados e aplicação de técnicas como **validação cruzada**. Isso permitirá verificar se os padrões identificados realmente refletem uma lógica generalizável ou se o modelo apenas aprendeu a classificar corretamente os exemplos já vistos.

As árvores de decisão são ferramentas poderosas para problemas de classificação, oferecendo regras interpretáveis para tomada de decisão. No entanto, devem ser aplicadas com cautela para evitar problemas de **overfitting** e garantir sua aplicabilidade em dados reais.
